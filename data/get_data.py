import os
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import hashlib
import time

# ---- CONFIG ----
SERPAPI_KEY = "c7ef258490afcbc2052968b72e9b8a72254e358fe33bcbad5a401b8924ad4412"

queries_apple = [
    "apple",
    "fresh apple",
    "red apple",
    "green apple",
    "apple on white background",
    "apple on black background",
    "apple isolated",
    "apple studio shot",
    "apple photography",
    "single apple",
    "apple photo",
    "raw apple image",
    "apple high detail",
    "apple in natural light",
    "whole apple"
]

queries_banana = [
    "banana",
    "fresh banana",
    "ripe banana",
    "yellow banana",
    "banana on white background",
    "banana on black background",
    "banana isolated",
    "banana studio shot",
    "banana photography",
    "single banana",
    "banana photo",
    "raw banana image",
    "banana high detail",
    "banana in natural light",
    "whole banana"
]

queries_carrot = [
    "carrot",
    "fresh carrot",
    "whole carrot",
    "carrot on white background",
    "carrot on black background",
    "carrot isolated",
    "carrot studio shot",
    "carrot photography",
    "single carrot",
    "carrot photo",
    "raw carrot image",
    "carrot high detail",
    "carrot natural light",
    "orange carrot",
    "clean carrot photo"
]

queries_cucumber = [
    "cucumber",
    "fresh cucumber",
    "whole cucumber",
    "cucumber on white background",
    "cucumber on black background",
    "cucumber isolated",
    "cucumber studio shot",
    "cucumber photography",
    "single cucumber",
    "cucumber photo",
    "raw cucumber image",
    "cucumber high detail",
    "cucumber natural light",
    "green cucumber",
    "clean cucumber photo"
]

queries_orange = [
    "orange",
    "fresh orange",
    "whole orange",
    "orange on white background",
    "orange on black background",
    "orange isolated",
    "orange studio shot",
    "orange photography",
    "single orange",
    "orange photo",
    "raw orange image",
    "orange high detail",
    "orange in natural light",
    "ripe orange",
    "clean orange photo"
]



def create_file(fruit_name):
    
    FOLDER = "fruit_images/" + fruit_name
    if fruit_name == "apple":
        queries = queries_apple
    elif fruit_name == "banana":
        queries = queries_banana
    elif fruit_name == "orange":
        queries = queries_orange
    elif fruit_name == "cucumber":
        queries = queries_cucumber
    elif fruit_name == "carrot":
        queries = queries_carrot
    else:
        print(f"[ERROR] Unknown fruit name: {fruit_name}")
        return
    IMAGES_PER_QUERY = 100          # how many images to try per query
    REQUEST_TIMEOUT = 12
    SLEEP_BETWEEN_QUERIES = 1.0    # be nice to the API / avoid rate-limits

    # ---- CREATE FOLDER ----
    os.makedirs(FOLDER, exist_ok=True)

    # ---- DUPLICATE CHECK ----
    def sha256_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    seen_hashes = set()

    def download_and_save_image(url: str) -> bool:
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200 or not r.content:
                return False

            # Try open as image
            img = Image.open(BytesIO(r.content))
            img = img.convert("RGB")

            h = sha256_bytes(r.content)
            if h in seen_hashes:
                return False
            seen_hashes.add(h)

            out_path = os.path.join(FOLDER, f"{h[:12]}.jpg")
            img.save(out_path, "JPEG", quality=95)
            return True
        except Exception:
            return False

    def fetch_yandex_image_urls(query: str, max_results: int) -> list[str]:
        params = {
            "engine": "yandex_images",
            "text": query,          # yandex_images uses "text" commonly; "q" also works sometimes
            "api_key": SERPAPI_KEY,
            "num": min(max_results, 200)  # serpapi limit per page varies; keep safe
        }

        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"[WARN] SerpAPI error for query='{query}': HTTP {resp.status_code}")
            return []

        data = resp.json()
        results = data.get("images_results", [])
        urls = []

        for item in results:
            # Try multiple possible fields SerpAPI can return
            if "original" in item and item["original"]:
                urls.append(item["original"])
            elif "source" in item and item["source"]:
                urls.append(item["source"])
            elif "thumbnail" in item and item["thumbnail"]:
                urls.append(item["thumbnail"])

        return urls[:max_results]

    # ---- MAIN: DOWNLOAD FROM YANDEX (via SerpAPI) ----
    total_saved = 0

    for q in queries:
        urls = fetch_yandex_image_urls(q, IMAGES_PER_QUERY)
        if not urls:
            print(f"[INFO] No URLs for query: {q}")
            continue

        saved_this_query = 0
        for url in tqdm(urls, desc=f"Query: {q}", unit="img"):
            if download_and_save_image(url):
                total_saved += 1
                saved_this_query += 1

        print(f"[OK] Saved {saved_this_query} images for query: {q}")
        time.sleep(SLEEP_BETWEEN_QUERIES)

    print(f"\nDONE. Total saved: {total_saved} images in ./{FOLDER}/")


if __name__ == "__main__":
    fruits = ["cucumber"]
    for fruit in fruits:
        print(f"Starting download for: {fruit}")
        create_file(fruit)
        print(f"Finished download for: {fruit}\n")