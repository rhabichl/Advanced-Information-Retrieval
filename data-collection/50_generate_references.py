# prompt is generated with chatgpt 
# Create a perfekt Prompt for an LLM to extarct the reference from a legal document, 
# the legal Document is from the austrian Verfassungsgericht. It is input via a html. 
# The Goal for the LLM is to extarct the references from this text passage in a json document. 
# It is important that the LLM outputs the text exactly as it is written in the text because i need it later char for char. 
# Please optimize the prompt and find a perfect output format. This must be adheared. Here are some examples: [.....] output in markdown format
# 

import os
import json
from tqdm import tqdm
import time  
import asyncio
import aiohttp
from aiofiles import open as aioopen
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENCY = 15



prompt = ""

with open("./prompt.md") as f:
    prompt = f.read()


async def download_file(session, semaphore, filename, pbar):
    async with semaphore:
        try:
            text = ""
            with open(f"./data_proccessed/{filename}", "r") as input:
                text = input.read()

            headers = {
               "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get("TOKEN")}",
            }
            
            model = "gpt-oss-120b"
            api_url=os.environ.get("API_URL")
            payload = {
                "model": model,
                "input": prompt + "\n\n\n [BEGIN INPUT]\n" + text + "\n\n[END INPUT]",
                "stream": False,
                "reasoning": {"effort" : "low"}
            }          
            async with session.post(api_url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                with open("./data_extracted/" + filename + ".json", "w") as f:
                    obj = await resp.json()
                    f.write(json.dumps(json.loads(obj["output"][1]["content"][0]["text"]), indent=2))

        except Exception as e:
            print(f"Failed: {filename} – {e}")

        finally:
            pbar.update(1)


async def main(files):
    total = len(files)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    pbar = tqdm_asyncio(total=total, desc="Extracting", unit="file")

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_file(session, semaphore, filename, pbar)
            for filename in files
        ]
        await asyncio.gather(*tasks)

    pbar.close()


if __name__ == "__main__":
    files = []
    directory = "data_proccessed"
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".html"): 
            files.append(filename)
        else:
            continue

    asyncio.run(main(files))

