"""Inference component to generate predictions with the fine-tuned model."""

from kfp.dsl import Dataset, Input, Model, Output, component


@component(
	base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
	packages_to_install=[
		"transformers==4.46.3",
		"peft==0.13.2",
		"torch>=2.0.0",
		"pandas>=2.3.3",
		"datasets>=4.2.0",
		"google-cloud-storage>=2.19.0",
	],
)
def inference_component(
	model: Input[Model],  # type: ignore
	test_dataset: Input[Dataset],  # type: ignore
	predictions: Output[Dataset],  # type: ignore
	max_new_tokens: int = 128,
	temperature: float = 0.7,
	top_p: float = 0.9,
) -> None:
	import logging
	import os
	import re

	import pandas as pd
	import torch
	from datasets import Dataset
	from peft import PeftModel
	from transformers import AutoModelForCausalLM, AutoTokenizer

	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	base_model_name = "microsoft/Phi-3-mini-4k-instruct"

	logger.info("Loading tokenizer and base model %s (CPU-only mode)", base_model_name)
	tokenizer = AutoTokenizer.from_pretrained(base_model_name)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
	logger.info("Attaching LoRA adapters from %s", model.path)
	model_ft = PeftModel.from_pretrained(base, model.path)
	model_ft.eval()

	# Load test data
	df = pd.read_csv(test_dataset.path)
	if "messages" in df.columns:
		import ast

		def parse_messages(val):
			if isinstance(val, list):
				return val
			try:
				return ast.literal_eval(val)
			except Exception:
				return val

		df["messages"] = df["messages"].apply(parse_messages)
		ds = Dataset.from_pandas(df[["messages"]])
	else:
		# Rebuild messages if needed
		def to_messages(row):
			user = row.get("prompt") or row.get("sentence") or ""
			ref = row.get("completion") or row.get("translation_extra") or ""
			return [{"role": "user", "content": user}, {"role": "assistant", "content": ref}]

		df["messages"] = df.apply(to_messages, axis=1)
		ds = Dataset.from_pandas(df[["messages"]])

	def build_prompt_from_messages(msgs):
		return tokenizer.apply_chat_template(msgs, tokenize=False)

	def generate(text_prompt: str) -> str:
		inputs = tokenizer(text_prompt, return_tensors="pt").to(model_ft.device)
		with torch.no_grad():
			out = model_ft.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				do_sample=True,
				temperature=temperature,
				top_p=top_p,
				pad_token_id=tokenizer.eos_token_id,
			)
		decoded = tokenizer.decode(out[0], skip_special_tokens=False)
		return decoded

	def extract_response(model_output: str) -> str:
		# Extract content after the assistant tag up to end token
		m = re.search(r"<\|assistant\|>\n?(.*?)(?:<\|end\|>|$)", model_output, flags=re.S)
		return m.group(1).strip() if m else model_output

	rows = []
	for item in ds:
		msgs = item["messages"]
		user_text = next((m["content"] for m in msgs if m.get("role") == "user"), "")
		ref_text = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
		prompt = build_prompt_from_messages(msgs)
		gen = generate(prompt)
		resp = extract_response(gen)
		rows.append({"user_input": user_text, "reference": ref_text, "extracted_response": resp})

	out_df = pd.DataFrame(rows)
	os.makedirs(os.path.dirname(predictions.path), exist_ok=True)
	out_df.to_csv(predictions.path, index=False)
