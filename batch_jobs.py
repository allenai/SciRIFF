import requests

def fetch_batches(api_key):
    """Fetches all batches from the API and returns them as a list."""
    url = "https://api.openai.com/v1/batches"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print("Failed to fetch batches")
        return []

def display_batches(batches):
    """Displays the batches in a numbered list."""
    print("Select a batch to view details (or 99 to exit):")
    for index, batch in enumerate(batches):
        print(f"{index + 1}. Batch ID: {batch['id']}, Status: {batch['status']}")
    print("99. Exit")

def fetch_batch_details(api_key, batch_id):
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

import os

def main():
    API_KEY = os.getenv("OPENAI_API_KEY")  # Fetch API key from environment variable
    if API_KEY is None:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    while True:
        batches = fetch_batches(API_KEY)
        display_batches(batches)
        
        try:
            choice = int(input("Enter your choice: "))
            if choice == 99:
                break
            elif 1 <= choice <= len(batches):
                fetch_batch_details(API_KEY, batches[choice - 1]['id'])
            else:
                print("Invalid choice, please try again.")
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":
    main()
    