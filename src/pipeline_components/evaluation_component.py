"""Evaluation component computing Rouge metrics on predictions."""

from kfp.dsl import Dataset, Input, Metrics, Output, component


@component(
	base_image="cicirello/pyaction:3.11",
	packages_to_install=[
		"pandas>=2.3.3",
		"rouge-score>=0.1.2",
	],
)
def evaluation_component(
	predictions: Input[Dataset],  # type: ignore
	evaluation_results: Output[Dataset],  # type: ignore
	aggregated_metrics: Output[Metrics],  # type: ignore
) -> None:
	import json
	import logging
	import os

	import pandas as pd
	from rouge_score import rouge_scorer

	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	logger.info("Loading predictions from %s", predictions.path)
	df = pd.read_csv(predictions.path)

	scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
	per_sample = []
	for _, row in df.iterrows():
		user = str(row.get("user_input", ""))
		ref = str(row.get("reference", ""))
		pred = str(row.get("extracted_response", ""))
		scores = scorer.score(ref, pred)
		rouge_l_f = float(scores["rougeL"].fmeasure)
		per_sample.append({
			"user_input": user,
			"reference": ref,
			"extracted_response": pred,
			"rougeL_f": rouge_l_f,
		})

	per_sample_df = pd.DataFrame(per_sample)
	os.makedirs(os.path.dirname(evaluation_results.path), exist_ok=True)
	per_sample_df.to_csv(evaluation_results.path, index=False)

	# Aggregate
	mean_rouge_l = float(per_sample_df["rougeL_f"].mean()) if not per_sample_df.empty else 0.0
	aggregated_metrics.log_metric("rougeL_f", mean_rouge_l)

	# Also write a JSON summary alongside for convenience
	with open(os.path.join(os.path.dirname(evaluation_results.path), "aggregated_metrics.json"), "w", encoding="utf-8") as f:
		json.dump({"rougeL_f": mean_rouge_l}, f)
