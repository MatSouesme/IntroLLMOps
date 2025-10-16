from kfp.dsl import component, OutputPath

@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "google-cloud-storage==2.19.0",
        "loguru==0.7.2",
    ],
)
def data_transformation(
    input_csv_gcs_uri: str,
    train_out_path: OutputPath(str),
    test_out_path: OutputPath(str),
    text_col: str = "sentence",
    target_col: str = "translation",
    test_size: float = 0.1,
    seed: int = 42,
):
    import os, json, random                                 # ← ICI
    from loguru import logger
    import pandas as pd
    from io import BytesIO
    from google.cloud import storage

    def parse_gs_uri(gs_uri: str):
        assert gs_uri.startswith("gs://"), f"URI attendu gs://..., reçu: {gs_uri}"
        p = gs_uri[5:]; bucket, _, blob = p.partition("/"); return bucket, blob

    bucket, blob = parse_gs_uri(input_csv_gcs_uri)
    logger.info(f"Downloading CSV from gs://{bucket}/{blob}")
    data = storage.Client().bucket(bucket).blob(blob).download_as_bytes()
    df = pd.read_csv(BytesIO(data))

    if text_col not in df.columns: raise ValueError(f"Colonne texte '{text_col}' absente. Colonnes: {list(df.columns)}")
    if target_col not in df.columns: raise ValueError(f"Colonne cible '{target_col}' absente. Colonnes: {list(df.columns)}")

    records = [
        [{"role":"user","content":str(row[text_col])},
         {"role":"assistant","content":str(row[target_col])}]
        for _, row in df.iterrows()
    ]
    n = len(records); logger.info(f"Records: {n}")

    rnd = random.Random(seed); idx = list(range(n)); rnd.shuffle(idx)
    cut = int(n*(1-test_size)); train_idx, test_idx = idx[:cut], idx[cut:]
    logger.info(f"Split -> train={len(train_idx)} test={len(test_idx)}")

    for p in (train_out_path, test_out_path): os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(train_out_path,"w",encoding="utf-8") as f:
        for i in train_idx: f.write(json.dumps(records[i], ensure_ascii=False)+"\n")
    with open(test_out_path,"w",encoding="utf-8") as f:
        for i in test_idx: f.write(json.dumps(records[i], ensure_ascii=False)+"\n")

    logger.success("✅ Data transformation completed — Yoda dataset ready!")
