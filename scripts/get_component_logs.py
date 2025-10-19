"""Retrieve detailed logs for a specific component in a Vertex AI Pipeline run."""
import sys
from google.cloud import logging as cloud_logging

if len(sys.argv) < 3:
    print("Usage: python scripts/get_component_logs.py <pipeline_job_id> <component_name>")
    print("Example: python scripts/get_component_logs.py mathieu-soul-model-training-pipeline-20251019144511 fine-tuning-component")
    sys.exit(1)

job_id = sys.argv[1]
component_name = sys.argv[2]

# Specify project explicitly
PROJECT_ID = "peak-haven-475016-d5"
client = cloud_logging.Client(project=PROJECT_ID)

# Query for logs from the specific component
# CustomJob logs contain the stdout/stderr from the component execution
filter_str = f'''
resource.type="aiplatform.googleapis.com/PipelineJob"
resource.labels.pipeline_job_id="{job_id}"
"{component_name}"
'''

print(f"Fetching logs for component '{component_name}' in job '{job_id}'...")
print("=" * 80)

entries = client.list_entries(filter_=filter_str, order_by=cloud_logging.DESCENDING, max_results=100)

found_logs = False
for entry in entries:
    found_logs = True
    print(f"[{entry.timestamp}] {entry.payload}")
    print("-" * 80)

if not found_logs:
    print("No logs found with this filter. Trying broader search...")
    # Try a broader search
    filter_str2 = f'''
resource.type="aiplatform.googleapis.com/CustomJob"
labels."ml.googleapis.com/pipeline_job_id"="{job_id}"
'''
    print(f"\nUsing filter: {filter_str2}")
    print("=" * 80)
    
    entries2 = client.list_entries(filter_=filter_str2, order_by=cloud_logging.DESCENDING, max_results=200)
    
    for entry in entries2:
        print(f"[{entry.timestamp}] {entry.payload}")
        print("-" * 80)
