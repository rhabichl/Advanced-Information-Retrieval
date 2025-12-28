import pandas as pd
import json
import os
import re

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


df = pd.DataFrame(columns=['id', 'query', 'document'])

directory = "data_proccessed"

index = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    id = filename.removesuffix(".html")
    to_add = []
    with open(f"./{directory}/{file}", "r") as f:
        raw_html = f.read()
    cleantext = re.sub(CLEANR, '', raw_html)
    with open(f"./matchings/{file}.json", "r") as f:
        o = json.load(f)
    if len(o) == 0:
        continue
    for i in o:
        for e in i["matches"]:
            to_add = []
            to_add = [id , cleantext, e["context"]]
            df.loc[index] = to_add
            index += 1


df.to_parquet("./dataset.parquet")
