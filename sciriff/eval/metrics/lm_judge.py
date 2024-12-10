from openai import OpenAI
import json
import argparse
import requests
import os

# Set your OpenAI API key
API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# Function to retrieve batch job results
def retrieve_batch_job_results(job_id, output_file):
    try:
        # Retrieve the job status and details
        response = client.files.content(job_id)
        # Get the result data if the job is complete
        result = response.content
        results = []
        for line in result.decode().split('\n'):
            if line:
                results.append(json.loads(line))
        # Save the result to the specified output file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error retrieving job results: {e}")

def fetch_batch_file(api_key, batch_id):
    """Fetches and displays details of a specific batch."""
    url = f"https://api.openai.com/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        batch = response.json()
        print("Batch Details:")
        print(batch)
    else:
        print("Failed to fetch batch details")
    return batch['output_file_id']
