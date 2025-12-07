import json
from tqdm import tqdm
import time  
import asyncio
import aiohttp
from aiofiles import open as aioopen
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENCY = 10


async def download_file(session, semaphore, filename, url, pbar):
    async with semaphore:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()

                async with aioopen("./data/" + filename + ".html", "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 32):
                        await f.write(chunk)
                time.sleep(0.1)
        except Exception as e:
            print(f"Failed: {filename} – {e}")

        finally:
            pbar.update(1)


async def main(files):
    total = len(files)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    pbar = tqdm_asyncio(total=total, desc="Downloading", unit="file")

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_file(session, semaphore, filename, url, pbar)
            for filename, url in files
        ]
        await asyncio.gather(*tasks)

    pbar.close()


if __name__ == "__main__":
    with open("./download_list.json", "r") as f:
        files = json.load(f)

    asyncio.run(main(files))
