from bs4 import BeautifulSoup
import os

directory = os.fsencode("./data")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".html"): 
        filepath = os.path.join(directory.decode("utf-8"), filename)
        with open(filepath, "r") as f:
            html = f.read()

            soup = BeautifulSoup(html, "html.parser")
            body = soup.body

            body_content = body.decode_contents() if body else ""
            output_file = "./data_proccessed/" + filename
            with open(output_file, "w+", encoding="utf-8") as f:
                f.write(body_content)

            print("Body content extracted to:", output_file)
        continue
    else:
        continue
