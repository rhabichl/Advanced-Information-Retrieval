from pypdf import PdfReader
import os

directory = "law"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    reader = PdfReader(f"./law/{filename}")
    with open(f"./law_txt/{filename.split(".")[0]}.txt", "w") as f:
        for page in reader.pages:
            f.write(page.extract_text())