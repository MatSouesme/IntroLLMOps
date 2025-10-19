"""Fine-tuning component for Vertex AI pipeline (LoRA on Phi-3) - CPU version."""

from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
	# CUDA-enabled image (though we'll force CPU to avoid GPU quota)
	base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
	packages_to_install=[
		# Versions aligned with session note for Vertex serving compatibility
		"transformers==4.46.3",
		"peft==0.13.2",
		"trl==0.11.4",  # trainer utils
		"accelerate>=1.10.1",
		"datasets>=4.2.0",
		"pandas>=2.3.3",
		"tensorboard>=2.20.0",
		"google-cloud-storage>=2.19.0",
	],
)
def fine_tuning_component(
	train_dataset: Input[Dataset],  # type: ignore
	model: Output[Model],  # type: ignore - output
	training_metrics: Output[Metrics],  # type: ignore - output
	# Hyperparams
	num_train_epochs: int = 1,
	learning_rate: float = 2e-4,
	per_device_train_batch_size: int = 2,  # Reduced for CPU
	gradient_accumulation_steps: int = 8,  # Increased to compensate
	lora_r: int = 8,
	lora_alpha: int = 16,
	lora_dropout: float = 0.1,
) -> None:
	"""Fine-tune Phi-3-mini-4k-instruct with LoRA on the train dataset (CPU-only).

	Saves the adapter model to model.path and logs metrics to training_metrics.
	"""
	import json
	import logging
	import os
	import sys

	import pandas as pd
	import torch
	from datasets import Dataset
	from peft import LoraConfig, get_peft_model
	from transformers import AutoModelForCausalLM, AutoTokenizer
	from trl import SFTConfig, SFTTrainer

	# Setup detailed logging
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		handlers=[logging.StreamHandler(sys.stdout)]
	)
	logger = logging.getLogger(__name__)

	try:
		logger.info("=" * 80)
		logger.info("FINE-TUNING COMPONENT STARTED (CPU-only mode)")
		logger.info("=" * 80)

		model_name = "microsoft/Phi-3-mini-4k-instruct"
		logger.info(f"Model to fine-tune: {model_name}")
		logger.info(f"Hyperparameters: epochs={num_train_epochs}, lr={learning_rate}, batch={per_device_train_batch_size}")
		logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

		# Load training data (messages column with role/content pairs expected)
		logger.info(f"Loading training dataset from: {train_dataset.path}")
		df = pd.read_csv(train_dataset.path)
		logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

		# Expect a 'messages' column with JSON-like structure
		if "messages" in df.columns:
			import ast
			logger.info("Parsing 'messages' column from CSV strings to Python objects")

			def parse_messages(val):
				if isinstance(val, list):
					return val
				try:
					return ast.literal_eval(val)
				except Exception as e:
					logger.warning(f"Failed to parse message: {val[:100]}... Error: {e}")
					return val

			df["messages"] = df["messages"].apply(parse_messages)
			ds = Dataset.from_pandas(df[["messages"]])
			logger.info(f"Dataset created with {len(ds)} samples")
		else:
			# Fallback: build messages from existing columns
			logger.info("Building 'messages' column from existing columns")

			def to_messages(row):
				user = row.get("prompt") or row.get("sentence") or ""
				assistant = row.get("completion") or row.get("translation_extra") or ""
				return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]

			df["messages"] = df.apply(to_messages, axis=1)
			df = df[["messages"]]
			ds = Dataset.from_pandas(df)
			logger.info(f"Dataset created with {len(ds)} samples")

		# LoRA configuration
		logger.info("Configuring LoRA...")
		lora_config = LoraConfig(
			r=lora_r,
			lora_alpha=lora_alpha,
			lora_dropout=lora_dropout,
			bias="none",
			task_type="CAUSAL_LM",
			target_modules=["q_proj", "v_proj"],
		)
		logger.info(f"LoRA config created: {lora_config}")

		# Load tokenizer
		logger.info(f"Loading tokenizer from {model_name}")
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
			logger.info("Set pad_token = eos_token")

		# Load base model (CPU-only, FP32)
		logger.info(f"Loading base model {model_name} for CPU training")
		logger.info("Using CPU-only mode to avoid GPU quota limits")
		base_model = AutoModelForCausalLM.from_pretrained(
			model_name, 
			torch_dtype=torch.float32,  # FP32 for CPU
			low_cpu_mem_usage=True
		)
		logger.info("Applying LoRA to model")
		lora_model = get_peft_model(base_model, lora_config)

		# Count trainable params
		trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
		total = sum(p.numel() for p in lora_model.parameters())
		logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

		# Train/Val split (10% validation)
		logger.info("Splitting dataset into train/val (90/10)")
		split = ds.train_test_split(test_size=0.1, seed=42)
		logger.info(f"Train samples: {len(split['train'])}, Val samples: {len(split['test'])}")

		# Convert chat messages -> single text using tokenizer chat template
		logger.info("Applying chat template to convert messages to text")
		def to_text(ex):
			text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
			return {"text": text}

		train_ds = split["train"].map(
			to_text,
			remove_columns=[c for c in split["train"].column_names if c != "messages"],
		)
		eval_ds = split["test"].map(
			to_text,
			remove_columns=[c for c in split["test"].column_names if c != "messages"],
		)
		logger.info("Chat template applied successfully")

		# SFT Trainer config (CPU-optimized)
		logger.info("Configuring SFT Trainer for CPU")
		tb_dir = training_metrics.path
		os.makedirs(tb_dir, exist_ok=True)
		logger.info(f"TensorBoard logs will be saved to: {tb_dir}")

		sft_config = SFTConfig(
			output_dir=os.path.dirname(model.path),
			num_train_epochs=num_train_epochs,
			per_device_train_batch_size=per_device_train_batch_size,
			gradient_accumulation_steps=gradient_accumulation_steps,
			learning_rate=learning_rate,
			logging_dir=tb_dir,
			logging_steps=5,
			report_to=["tensorboard"],
			save_strategy="no",
			eval_strategy="epoch",
			fp16=False,  # No FP16 on CPU
			use_cpu=True,  # Force CPU training
			dataloader_num_workers=4,  # Multi-threaded loading
		)
		logger.info(f"SFT config: {sft_config}")

		logger.info("Creating SFTTrainer")
		trainer = SFTTrainer(
			model=lora_model,
			tokenizer=tokenizer,
			args=sft_config,
			train_dataset=train_ds,
			eval_dataset=eval_ds,
			dataset_text_field="text",
			formatting_func=None,
		)

		logger.info("=" * 80)
		logger.info("STARTING TRAINING (CPU - this will take longer than GPU)")
		logger.info("=" * 80)
		train_result = trainer.train()
		metrics = train_result.metrics
		logger.info("=" * 80)
		logger.info("TRAINING FINISHED SUCCESSFULLY")
		logger.info(f"Metrics: {metrics}")
		logger.info("=" * 80)

		# Save only the adapters (compact) to output model dir
		logger.info(f"Saving model to: {model.path}")
		os.makedirs(model.path, exist_ok=True)
		trainer.model.save_pretrained(model.path)
		tokenizer.save_pretrained(model.path)
		logger.info("Model and tokenizer saved successfully")

		# Log metrics to Kubeflow
		to_log = {
			"train_runtime": float(metrics.get("train_runtime", 0.0)),
			"train_samples_per_second": float(metrics.get("train_samples_per_second", 0.0)),
			"train_loss": float(metrics.get("train_loss", 0.0)),
			"eval_loss": float(metrics.get("eval_loss", 0.0)),
			"num_train_epochs": float(num_train_epochs),
			"learning_rate": float(learning_rate),
			"batch_size": float(per_device_train_batch_size),
		}
		logger.info(f"Logging metrics to Kubeflow: {to_log}")
		training_metrics.log_metric("train_loss", to_log["train_loss"])
		training_metrics.log_metric("eval_loss", to_log["eval_loss"])
		with open(os.path.join(tb_dir, "aggregated_metrics.json"), "w", encoding="utf-8") as f:
			json.dump(to_log, f)

		logger.info("=" * 80)
		logger.info("FINE-TUNING COMPONENT COMPLETED SUCCESSFULLY")
		logger.info("=" * 80)

	except Exception as e:
		logger.error("=" * 80)
		logger.error("FATAL ERROR IN FINE-TUNING COMPONENT")
		logger.error("=" * 80)
		logger.error(f"Exception type: {type(e).__name__}")
		logger.error(f"Exception message: {str(e)}")
		import traceback
		logger.error("Full traceback:")
		logger.error(traceback.format_exc())
		logger.error("=" * 80)
		raise
