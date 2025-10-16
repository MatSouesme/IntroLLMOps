"""Project-wide constants for Vertex AI pipelines."""

# Identifiants GCP
PROJECT_ID = "peak-haven-475016-d5"
REGION = "europe-west2"

# Bucket GCS utilisé pour la pipeline
BUCKET = "llmops_project_bucket"
PIPELINE_ROOT_PATH = f"{BUCKET}/pipeline_root"  # sans "gs://"

# Dataset d’entrée (ton fichier CSV dans le bucket)
RAW_DATASET_URI = f"gs://{BUCKET}/yoda_sentences.csv"
