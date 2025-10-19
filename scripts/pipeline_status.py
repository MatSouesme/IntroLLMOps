"""Check Vertex AI Pipeline Job status by ID.

Usage:
    python scripts/pipeline_status.py <job_id>

Example job_id:
    mathieu-soul-model-training-pipeline-20251019134531
"""
from __future__ import annotations
import os
import sys
from google.cloud import aiplatform

PROJECT = os.getenv("GCP_PROJECT_ID") or "peak-haven-475016-d5"
REGION = os.getenv("GCP_REGION") or "europe-west2"

aiplatform.init(project=PROJECT, location=REGION)

if len(sys.argv) < 2:
    print("Usage: python scripts/pipeline_status.py <job_id>")
    sys.exit(2)

job_id = sys.argv[1]
res_name = f"projects/{aiplatform.initializer.global_config.project}/locations/{aiplatform.initializer.global_config.location}/pipelineJobs/{job_id}"

job = aiplatform.PipelineJob.get(res_name)
print(f"Job: {job.resource_name}")
print(f"Display name: {job.display_name}")
print(f"State: {job.state}")
