import json
import requests
import time
from tqdm import tqdm


# curl 'https://data.bka.gv.at/ris/api/v2.6/Judikatur?Applikation=Vfgh&DokumenteProSeite=OneHundred&Seitennummer=1'
URL = "https://data.bka.gv.at/ris/api/v2.6/Judikatur?Applikation=Vfgh&DokumenteProSeite=OneHundred&Seitennummer="
# there are 237 pages of documents

for i in tqdm(range(237)):
    r = requests.get(URL+f"{i+1}")
    with open(f'refs/search_result-{i}-237.json', 'wb') as f:
        f.write(r.content)
    time.sleep(3)