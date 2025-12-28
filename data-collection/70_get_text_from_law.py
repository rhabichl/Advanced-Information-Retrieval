import requests

BASE_URL = "https://data.bka.gv.at/ris/api/v2.6"

query = "Art23i Abs4 B-VG"

url = f"{BASE_URL}/Rechtssaetze"

params = {
    "Suchbegriff": query,
    "page": 1,
    "pageSize": 10
}

headers = {
    "Accept": "application/json"
}

response = requests.get(url, params=params, headers=headers)
response.raise_for_status()

data = response.json()

for item in data.get("Rechtssaetze", []):
    print("Titel:", item.get("Titel"))
    print("Text:", item.get("Rechtssatz"))
    print("-" * 80)



exit(1)

from pypdf import PdfReader
import os

directory = "law"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    reader = PdfReader(f"./law/{filename}")
    with open(f"./law_txt/{filename.split(".")[0]}.txt", "w") as f:
        for page in reader.pages:
            f.write(page.extract_text())