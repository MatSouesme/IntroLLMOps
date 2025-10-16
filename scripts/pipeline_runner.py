"""Pipeline compilation & submission for Vertex AI (session 2)."""

import os
import sys

# Permet d'importer `src.*` quand on lance le script depuis `scripts/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.cloud import aiplatform
from kfp import compiler

from src.constants import (
    PROJECT_ID,
    REGION,
    PIPELINE_ROOT_PATH,  # ex: "llmops_project_bucket/pipeline_root"
    RAW_DATASET_URI,     # ex: "gs://llmops_project_bucket/yoda_sentences.csv"
)
from src.pipelines.model_training_pipeline import model_training_pipeline


def main() -> None:
    # 1) Init Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # 2) Nom + chemin du template compilé
    pipeline_name = "mathieu_soul_model_training_pipeline"
    package_path = f"{pipeline_name}.json"

    # 3) Compiler la pipeline -> JSON
    compiler.Compiler().compile(
        pipeline_func=model_training_pipeline,
        package_path=package_path,
        type_check=False,
    )
    print(f"✅ Compiled pipeline -> {package_path}")

    # 4) Paramètres (avec override possible via argument CLI)
    #    Usage: python scripts/pipeline_runner.py gs://bucket/mon_autre.csv
    raw_uri = sys.argv[1] if len(sys.argv) > 1 else RAW_DATASET_URI
    params = {"raw_dataset_uri": raw_uri}

    # 5) Créer et lancer le job
    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=package_path,
        pipeline_root=f"gs://{PIPELINE_ROOT_PATH}",
        parameter_values=params,
        enable_caching=True,  # mets False si tu veux forcer la réexécution
        labels={"owner": "arnaud", "course": "llmops", "env": "dev"},
    )

    # run(sync=True) pour attendre la fin et voir l'état dans le terminal
    print("🚀 Submitting PipelineJob…")
    job.run(sync=True)
    print("🏁 PipelineJob completed.")


if __name__ == "__main__":
    main()
