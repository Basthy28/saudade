import os
import json
import time
import argparse
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from dotenv import load_dotenv

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))
EXTRACTION_DIR = DATA / "01_extraction"


# Default headers
DEFAULT_UA = "SAUDADE-RankingBot/1.0 (https://github.com/Basthy28/saudade) requests"

# Shared session configured in rank_entities
API_SESSION = None

checkpoint_file_lock = None # will be set in init_worker


def setup_session(token=None):
    """Sets up a requests session with retries and authentication if provided."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("✓ Authenticated session configured.")
    else:
        print("⚠️ No token configured. Using fallback User-Agent rate limits.")

    session.headers.update(headers)
    return session


def get_backlinks(title, session):
    """Gets total number of backlinks for a given title using MediaWiki Action API."""
    url = "https://pt.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "backlinks",
        "bltitle": title,
        "bllimit": "max",
        "blnamespace": 0  # Only count links from other Wikipedia articles (not user pages, etc)
    }

    total_backlinks = 0
    try:
        while True:
            # MediaWiki Action API may fail with valid oauth tokens if the user account isn't mapped on this specific wiki.
            response = session.get(url, params=params, headers={"Authorization": None}, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "query" in data and "backlinks" in data["query"]:
                total_backlinks += len(data["query"]["backlinks"])

            if "continue" in data:
                params.update(data["continue"])
            else:
                break
    except Exception as e:
        print(f"[Warning] Failed to get backlinks for {title}: {e}")
        return -1
    return total_backlinks


def get_pageviews(title, session):
    """Gets total pageviews for the last year using Wikimedia Analytics REST API."""
    # Getting for the year 2023 (or an entire recent 12-month period)
    # Using 20230101 to 20231231 as an example period
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/pt.wikipedia.org/all-access/all-agents/{title}/monthly/20230101/20231231"
    
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 404:
            return 0 # No pageviews found
        response.raise_for_status()
        data = response.json()

        total_views = 0
        if "items" in data:
            for item in data["items"]:
                total_views += item.get("views", 0)
        return total_views
    except Exception as e:
        print(f"[Warning] Failed to get pageviews for {title}: {e}")
        return -1


def process_entity(args):
    """Process a single entity (loads json, gets metrics)"""
    json_path, token, output_csv = args
    session = setup_session(token) # Local session for the worker
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        title = data.get("title")
        if not title:
            return None
            
        # Optional: Add sleep to respect rate limits if no token is used
        if not token:
            time.sleep(0.5) 
            
        backlinks = get_backlinks(title, session)
        
        if not token:
            time.sleep(0.5)
            
        pageviews = get_pageviews(title, session)
        
        result = {
            "title": title,
            "path": json_path,
            "backlinks": backlinks if backlinks >= 0 else 0,
            "pageviews": pageviews if pageviews >= 0 else 0
        }
        
        # Write immediately to CSV (Checkpointing)
        if output_csv:
            df_row = pd.DataFrame([result])
            df_row.to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')
            
        return result
        
    except Exception as e:
        print(f"[Error] Failed processing {json_path}: {e}")
        return None

def find_entity_files(base_dir):
    """Recursively finds all entity_data.json files."""
    json_files = []
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[Error] Base directory not found: {base_dir}")
        return json_files

    print(f"Scanning for entity_data.json in {base_dir} ...")
    for file_path in base_path.rglob("entity_data.json"):
        json_files.append(str(file_path))
        
    print(f"Found {len(json_files)} entities.")
    return json_files

def main():
    load_dotenv()  # Load variables from .env file
    
    parser = argparse.ArgumentParser(description="Rank Portuguese Wiki Entities by Popularity")
    parser.add_argument("--data-dir", type=str, default=str(EXTRACTION_DIR / "newdata" / "temporal_relevant"), help="Base directory containing the entity data")
    parser.add_argument("--output", type=str, default=str(EXTRACTION_DIR / "entity_ranking.csv"), help="Output CSV path")
    parser.add_argument("--token", type=str, default=os.getenv("WIKIMEDIA_ACCESS_TOKEN"), help="Wikimedia Access Token (Defaults to .env)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSV output")
    
    args = parser.parse_args()
    
    files = find_entity_files(args.data_dir)
    
    if args.resume and os.path.exists(args.output):
        try:
            existing_df = pd.read_csv(args.output)
            processed_paths = set(existing_df['path'].tolist())
            files = [f for f in files if f not in processed_paths]
            print(f"Resuming... found {len(processed_paths)} already processed entities. {len(files)} remaining.")
        except pd.errors.EmptyDataError:
            print("Output file exists but is empty. Starting fresh.")
    else:
        # Initialize output CSV with headers
        pd.DataFrame(columns=["title", "path", "backlinks", "pageviews"]).to_csv(args.output, index=False, encoding='utf-8')
        print("Initialized new output file.")
    
    if args.limit:
        files = files[:args.limit]
        print(f"Limiting to {args.limit} files for testing.")
        
    if not files:
        print("No new files to process. Finished!")
        return
        
    print(f"Starting metric extraction with {args.workers} workers...")
    
    # Prepare arguments for multiprocessing
    worker_args = [(f, args.token, args.output) for f in files]
    
    results = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_entity, worker_args), total=len(worker_args)):
            if result:
                results.append(result)
                
    # Final ranking pass over all data
    print("\nCalculating final rankings...")
    try:
        df = pd.read_csv(args.output)
        
        if df.empty:
            print("No valid data collected.")
            return
            
        df = df.sort_values(by=["pageviews", "backlinks"], ascending=[False, False])
        
        # Remove old rank column if it exists to cleanly recreate it
        if "rank" in df.columns:
            df = df.drop(columns=["rank"])
            
        df.insert(0, "rank", range(1, len(df) + 1))
        
        # Save final ranked CSV
        df.to_csv(args.output, index=False, encoding='utf-8')
        print(f"\nFinal sorted ranking saved to {args.output}")
        
    except Exception as e:
        print(f"Error reading or writing final CSV: {e}")

if __name__ == "__main__":
    main()
