import os
import json

filtered_files = []
directory = "data_extracted"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    obj = {}
    if filename.endswith(".json"): 
        with open("./" + directory + "/" + filename, "r") as f:
            obj = json.load(f)
            if len(obj["references"]) != 0:
                filtered_files.append(filename)
    else:
        continue

with open("./filtered_referneces.json", "w") as f:
    f.write(json.dumps(filtered_files, indent=2))
print(len(filtered_files))