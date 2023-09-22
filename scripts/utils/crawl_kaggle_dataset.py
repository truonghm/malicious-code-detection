import argparse
import concurrent.futures
import hashlib
import json
import os
import signal

import bs4
import pandas as pd
import pymongo
import requests
from src.utils.logging import logger

TIMEOUT_SECS = 3
shutdown = False  # Flag to signal workers to stop working


def handle_shutdown(signum, frame):
    global shutdown
    shutdown = True


signal.signal(signal.SIGINT, handle_shutdown)


def extract_js(url):
    if shutdown:  # Check shutdown flag before making the request
        return None

    try:
        response = requests.get(url, timeout=TIMEOUT_SECS)
        if response.status_code != 200 or not response.text:
            raise requests.exceptions.RequestException("Invalid response")
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        scripts = soup.find_all("script")
        js_code = ""
        for script in scripts:
            if script.contents:
                js_code += script.contents[0]
        return js_code
    except requests.exceptions.RequestException as e:
        raise


def extract_js_and_save(idx, url, label, super_label):
    if shutdown:  # Check shutdown flag
        return None

    if not url.startswith("http") and not url.startswith("https"):
        new_url = "http://" + url
    else:
        new_url = url

    try:
        js_code = extract_js(new_url)
    except Exception as e:
        logger.error(f"{idx}, Failed,{super_label},{label},{new_url},{url}")
        return None

    if js_code == "":
        return None

    logger.info(f"{idx}, Success,{super_label},{label},{new_url},{url}")
    return {"url": new_url, "label": label, "super_label": super_label, "js_code": js_code}


def main():
    global shutdown
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--super_label", type=str)
    args = parser.parse_args()
    skip = args.skip
    limit = args.limit
    input_file = args.input
    output_dir = args.output
    super_label = args.super_label

    url_df = pd.read_csv(
        input_file,
        usecols=[0, 1, 2, 3],
        header=0,
        names=["idx", "url", "label", "super_label"],
        skiprows=range(1, skip + 1),  # skip the first 'skip' rows (assuming there's a header row too)
        nrows=limit,
    )  # read 'limit' rows
    if super_label:
        url_df = url_df[url_df["super_label"] == super_label]

    # url_df_slice = url_df.iloc[skip : skip + limit]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        args = [(idx, url, label, super_label) for idx, url, label, super_label in url_df.itertuples(index=False)]

        try:
            for res in executor.map(lambda x: extract_js_and_save(*x), args):
                if shutdown:  # Check shutdown flag
                    break

                if res:
                    file_name = hashlib.sha256(res["js_code"].encode("utf-8")).hexdigest()
                    file_path = os.path.join(output_dir, res["super_label"], file_name)
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))

                    if os.path.exists(file_path):
                        continue

                    with open(file_path, "w") as f:
                        f.write(res["js_code"])
        except KeyboardInterrupt:
            # This part is executed after Ctrl+C is pressed
            print("\nReceived shutdown signal, stopping workers...")
            shutdown = True


if __name__ == "__main__":
    main()
