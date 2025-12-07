#
# Foramt the Documents so i have a list with all links to download 
#
#
import os
import json

final_downloaad_list = []

directory = os.fsencode("./refs")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".json"): 
        filepath = os.path.join(directory.decode("utf-8"), filename)
        with open(filepath, "r") as f:
            o = json.load(f)
            for element in o["OgdSearchResult"]["OgdDocumentResults"]["OgdDocumentReference"]:
                ID = element["Data"]["Metadaten"]["Technisch"]["ID"]
                DownloadURL = element["Data"]["Dokumentliste"]["ContentReference"]["Urls"]["ContentUrl"][1]["Url"]
                #print("ID: " +ID + "  URL: " + DownloadURL)
                final_downloaad_list.append((ID, DownloadURL))
        continue
    else:
        continue


with open("./download_list.json", "w") as f:
    f.write(json.dumps(final_downloaad_list, indent=2))


