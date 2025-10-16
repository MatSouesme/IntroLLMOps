# scripts/pipeline_runner.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from google.cloud import aiplatform
from kfp import compiler

def main():
    # 1) Charger les variables d'environnement
    load_dotenv()  # lit .env si présent

    PROJECT = os.getenv("GCP_PROJECT_ID")
    REGION  = os.getenv("GCP_REGION", "europe-west2")
    BUCKET  = os.getenv("GCP_BUCKET_NAME")
    assert PROJECT and REGION and BUCKET, "Variables GCP manquantes (GCP_PROJECT_ID, GCP_REGION, GCP_BUCKET_NAME)."

    # 2) Paramètres du job / entrées
    input_csv = sys.argv[1] if len(sys.argv) > 1 else f"gs://{BUCKET}/yoda_sentences.csv"
    pipeline_root = f"gs://{BUCKET}/pipeline_root"
    labels = {"owner": "arnaud", "course": "llmops", "env": "dev"}

    # 3) Import du pipeline
    from src.pipelines.model_training_pipeline import model_training_pipeline

    # 4) Init Vertex AI
    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

    # 5) Compiler le pipeline -> JSON
    package_path = "pipeline_job.json"
    compiler.Compiler().compile(
        pipeline_func=model_training_pipeline,
        package_path=package_path,
        type_check=False,
    )

    # 6) Soumettre le job
    job = aiplatform.PipelineJob(
        display_name="llmops-yoda-data-prep",
        template_path=package_path,
        pipeline_root=pipeline_root,
        parameter_values={
            "input_csv_gcs_uri": input_csv,
            "text_col": "sentence",
            "target_col": "translation",  # ✅ colonne correcte
            "test_size": 0.1,
            "seed": 42,
        },
        enable_caching=False,
        labels=labels,
    )

    job.run(sync=True)

if __name__ == "__main__":
    main()
