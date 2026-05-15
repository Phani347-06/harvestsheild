import requests
import json

try:
    print("Testing health check...")
    response = requests.get("http://localhost:5001/", timeout=5)
    print(f"Health check status: {response.status_code}")
    print(f"Health check body: {response.json()}")
except Exception as e:
    print(f"Health check failed: {e}")

url = "http://localhost:5001/api/detect"
image_path = "test_detect.jpg"

data = {
    "sensors": json.dumps({"temperature": 30, "humidity": 70, "soil_moisture": 50}),
    "location": json.dumps({"latitude": 17.385, "longitude": 78.486})
}

with open(image_path, "rb") as f:
    files = {"image": f}
    print("Sending detect request...")
    try:
        response = requests.post(url, data=data, files=files, timeout=30)
        print(f"Detect status: {response.status_code}")
        print(f"Detect body: {response.json()}")
    except Exception as e:
        print(f"Detect failed: {e}")
