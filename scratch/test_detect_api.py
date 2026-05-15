import requests
import json
import os

url = "http://localhost:5001/api/detect"
image_path = "test_detect.jpg"

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found.")
    exit(1)

data = {
    "sensors": json.dumps({"temperature": 30, "humidity": 70, "soil_moisture": 50}),
    "location": json.dumps({"latitude": 17.385, "longitude": 78.486})
}

with open(image_path, "rb") as f:
    files = {"image": f}
    print("Sending request...")
    try:
        response = requests.post(url, data=data, files=files, timeout=30)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
