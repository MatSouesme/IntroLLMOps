import os
from google.cloud import aiplatform, storage

project = os.getenv("GCP_PROJECT_ID")
region  = os.getenv("GCP_REGION")
bucket  = os.getenv("GCP_BUCKET_NAME")
assert project and region and bucket, "Variables d'environnement manquantes (GCP_PROJECT_ID, GCP_REGION, GCP_BUCKET_NAME)."

print(f"✅ Initialisation Vertex AI (projet={project}, région={region})")
aiplatform.init(project=project, location=region)

print(f"📦 Liste des objets dans: gs://{bucket}")
client = storage.Client(project=project)
b = client.bucket(bucket)
for i, blob in enumerate(b.list_blobs(page_size=5), 1):
    print(f"{i:02d} - {blob.name}")
print("✅ GCP setup OK !")

