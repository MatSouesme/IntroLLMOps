"""Model training pipeline definition for Vertex AI."""

from kfp.dsl import pipeline

from src.pipeline_components.data_transformation_component import (
    data_transformation_component,
)
from src.pipeline_components.fine_tuning_component import fine_tuning_component
from src.pipeline_components.inference_component import inference_component
from src.pipeline_components.evaluation_component import evaluation_component


@pipeline(name="mathieu-soul-model-training-pipeline")
def model_training_pipeline(
    raw_dataset_uri: str,
    train_test_split_ratio: float = 0.1,
    # Fine-tune hyper-params (can be overridden at submission time)
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> None:
    """End-to-end pipeline: transform -> fine-tune -> inference -> evaluation."""

    # 1) Data transformation (produces train/test datasets)
    transform_task = data_transformation_component(
        train_test_split_ratio=train_test_split_ratio,
        raw_dataset_uri=raw_dataset_uri,
    )  # type: ignore

    # 2) Fine-tuning step (CPU-only to avoid GPU quota)
    fine_tune_task = fine_tuning_component(
        train_dataset=transform_task.outputs["train_dataset"],
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=2,  # Reduced for CPU
        gradient_accumulation_steps=8,  # Increased to compensate
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )  # type: ignore
    fine_tune_task.set_display_name("Fine-tune Phi-3 with LoRA (CPU)")
    fine_tune_task.set_cpu_limit("16")
    fine_tune_task.set_memory_limit("60G")

    # 3) Inference on test set (CPU-only)
    inference_task = inference_component(
        model=fine_tune_task.outputs["model"],
        test_dataset=transform_task.outputs["test_dataset"],
    )  # type: ignore
    inference_task.set_display_name("Generate predictions on test set (CPU)")
    inference_task.set_cpu_limit("8")
    inference_task.set_memory_limit("30G")

    # 4) Evaluation step (Rouge-L per-sample + aggregate)
    _ = evaluation_component(
        predictions=inference_task.outputs["predictions"],
    )  # type: ignore
