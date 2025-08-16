import requests
import json

#URL = "http://localhost:8080/predict"
URL = "https://seg-api-dev-268616946422.us-central1.run.app"
IMAGE_PATH = "testfile.png"   # change to your test image path

with open(IMAGE_PATH, "rb") as f:
    files = {"image": f}
    resp = requests.post(URL, files=files)

if resp.status_code == 200:
    data = resp.json()

    # Save full response for inspection
    with open("result.json", "w") as out:
        json.dump(data, out, indent=2)
    print("✅ Saved full response to result.json\n")

    # Show quick summary
    print("📌 Subjects found:")
    for subj in data.get("subjects", []):
        print(f"  - {subj['name']}")

    print("\n📌 Objects found:")
    for obj in data.get("objects", []):
        print(f"  - {obj['name']}")

else:
    print("❌ Request failed:", resp.status_code, resp.text)
