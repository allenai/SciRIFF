import sys
from openai import OpenAI

def cancel_job(job_id):
    client = OpenAI()
    try:
        response = client.batches.cancel(job_id)
        print(f"Job {job_id} canceled successfully.")
    except Exception as e:
        print(f"Failed to cancel job {job_id}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cancel_job.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    cancel_job(job_id)
    