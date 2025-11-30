import os
import time
import json
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import typer

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer()

def http_get(url: str, params: Optional[dict] = None, retries: int = 3, timeout: int = 20):
    """HTTP GET with retries and timeout."""
    headers = {"User-Agent": "Mozilla/5.0 (NFL QUANT Fetcher)"}
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"[retry {i+1}/{retries}] {e}")
            time.sleep(1.5 * (i + 1))

def fetch_csv_to_parquet(url: str, out_name: str):
    """Streams CSV → DataFrame → Parquet."""
    print(f"[fetch] {url}")
    df = pd.read_csv(url)
    raw_path = RAW_DIR / f"{out_name}.csv"
    proc_path = PROC_DIR / f"{out_name}.parquet"
    df.to_csv(raw_path, index=False)
    df.to_parquet(proc_path, index=False)
    print(f"[saved] {raw_path} and {proc_path}")
    return df

def fetch_json(url: str, out_name: str):
    """Fetch JSON data and save to files."""
    print(f"[fetch] {url}")
    r = http_get(url)
    data = r.json()
    raw_path = RAW_DIR / f"{out_name}.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {raw_path}")
    return data


@app.command()
def csv(
    url: str = typer.Argument(..., help="CSV URL to fetch"),
    name: str = typer.Argument(..., help="Output name for the file")
):
    """Fetch CSV data from URL."""
    try:
        df = fetch_csv_to_parquet(url, name)
        print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error fetching CSV: {e}")
        raise typer.Exit(1)

@app.command()
def json_cmd(
    url: str = typer.Argument(..., help="JSON URL to fetch"),
    name: str = typer.Argument(..., help="Output name for the file")
):
    """Fetch JSON data from URL."""
    try:
        data = fetch_json(url, name)
        if isinstance(data, dict):
            print(f"✅ JSON loaded: {len(data)} keys")
        elif isinstance(data, list):
            print(f"✅ JSON loaded: {len(data)} items")
        else:
            print(f"✅ JSON loaded: {type(data)}")
    except Exception as e:
        print(f"❌ Error fetching JSON: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()