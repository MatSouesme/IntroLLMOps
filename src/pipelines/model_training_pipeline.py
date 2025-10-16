from kfp import dsl
from src.pipeline_components.data_transformation_component import data_transformation

@dsl.pipeline(
    name="llmops-data-prep-pipeline",
    description="Pipeline de préparation des données Yoda (transformation + split train/test)"
)
def model_training_pipeline(
    input_csv_gcs_uri: str,
    text_col: str = "sentence",
    target_col: str = "translation",  # ✅ correspond à ton CSV
    test_size: float = 0.1,
    seed: int = 42,
):
    """
    Pipeline Kubeflow simple :
    1️⃣ Télécharge et transforme les données à partir d’un CSV GCS.
    2️⃣ Crée deux fichiers JSONL (train/test) prêts à être utilisés.
    """

    step = data_transformation(
        input_csv_gcs_uri=input_csv_gcs_uri,
        text_col=text_col,
        target_col=target_col,
        test_size=test_size,
        seed=seed,
    )

    step.set_display_name("data-transformation")
    step.set_caching_options(False) 
