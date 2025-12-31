"""Simple test script for the API."""

import requests
import json


def test_chat():
    url = "http://127.0.0.1:8000/chat"
    payload = {"message": "What is Zyloth Industries?"}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure it's running.")
    except Exception as e:
        print(f"Error: {e}")


def test_health():
    url = "http://127.0.0.1:8000/health"
    try:
        response = requests.get(url, timeout=10)
        print(f"Health Check: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing Health Endpoint...")
    test_health()
    print("\nTesting Chat Endpoint...")
    test_chat()
