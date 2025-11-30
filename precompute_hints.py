#!/usr/bin/env python3
"""
Pre-compute hints for Phrase Golf games.

This script should be run daily before 3am Pacific (when the game day changes).
It computes similarity scores for all hint phrases against upcoming target phrases
and caches them for fast retrieval.

Usage:
    python precompute_hints.py              # Compute for today and tomorrow
    python precompute_hints.py --days 7     # Compute for next 7 days
    python precompute_hints.py --game 150   # Compute for specific game number
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import numpy as np
import pytz
from tqdm import tqdm

# Model setup (same as application.py)
REPO_ID = "ChristianAzinn/e5-large-v2-gguf"
FILENAME_IN_REPO = "e5-large-v2.Q5_K_M.gguf"

print("Loading model...")
LOCAL_FILENAME = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_IN_REPO)
# Load model for embeddings - sequential processing is most reliable
model = Llama(
    model_path=LOCAL_FILENAME, 
    embedding=True, 
    n_ctx=512,          # Context size matches model training
    n_batch=512,        # Batch size for token processing
    n_threads=8,        # Use 8 CPU threads
    verbose=False       # Reduce verbose output
)

def embed(text):
    """Embed a single text."""
    return model.embed("query: " + text)

# Load targets
def load_targets():
    ret = []
    with open('targets.tsv', 'r') as targets_file:
        for line in targets_file:
            ret.append(line.strip())
    return ret

targets = load_targets()
print(f"Loaded {len(targets)} targets")

# Load hint phrases
def load_hint_phrases():
    phrases = set()
    
    # Load words from google-10000-english-no-swears.txt
    with open('google-10000-english-no-swears.txt', 'r') as f:
        for line in f:
            word = line.strip()
            if word:
                phrases.add(word)
    
    # Load targets
    phrases.update(targets)
    
    return sorted(list(phrases))

hint_phrases = load_hint_phrases()
print(f"Loaded {len(hint_phrases)} hint phrases")

# Game logic (same as application.py)
START_DATE = datetime.date(2024, 7, 2)

def get_game_num_for_today():
    now = datetime.datetime.now(pytz.timezone('US/Pacific'))
    three_hours_ago = now - datetime.timedelta(seconds=3*60*60)
    day = three_hours_ago.date()
    return (day - START_DATE).days

def get_target_for_game_num(game_num):
    hash_object = hashlib.md5(str(game_num).encode())
    hash_hex = hash_object.hexdigest()
    hash_int = int(hash_hex, 16)
    return targets[hash_int % len(targets)]

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Hints cache
HINTS_CACHE_DIR = "hints_cache"
os.makedirs(HINTS_CACHE_DIR, exist_ok=True)

def get_hints_cache_path(game_num):
    return os.path.join(HINTS_CACHE_DIR, f"game_{game_num}.json")

def compute_hints_for_game(game_num):
    """Compute and cache hints for a specific game number."""
    cache_path = get_hints_cache_path(game_num)
    
    # Check if already exists
    if os.path.exists(cache_path):
        print(f"  Cache already exists for game {game_num}, skipping...")
        return
    
    target = get_target_for_game_num(game_num)
    print(f"  Game {game_num}: Target is '{target}'")
    print(f"  Computing target embedding...")
    
    target_embedding = embed(target)
    
    print(f"  Computing embeddings for {len(hint_phrases)} phrases...")
    
    # Compute embeddings sequentially with progress bar
    hints = []
    for phrase in tqdm(hint_phrases, desc="  Processing", unit=" phrases"):
        phrase_embedding = embed(phrase)
        sim = cosine_similarity(target_embedding, phrase_embedding)
        hints.append({"phrase": phrase, "similarity": float(sim)})
    
    # Sort by similarity (descending)
    hints.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Save to cache
    print(f"  Saving cache to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(hints, f)
    
    print(f"  ✓ Completed game {game_num}")
    print(f"    Top 5 hints: {[h['phrase'] for h in hints[:5]]}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Pre-compute hints for Phrase Golf games')
    parser.add_argument('--days', type=int, help='Number of days ahead to compute (default: 2)')
    parser.add_argument('--game', type=int, help='Specific game number to compute')
    args = parser.parse_args()
    
    if args.game is not None:
        # Compute specific game
        print(f"Computing hints for game {args.game}")
        compute_hints_for_game(args.game)
    else:
        # Compute for today and N days ahead
        days = args.days if args.days is not None else 2
        today_game = get_game_num_for_today()
        print(f"Today's game number: {today_game}")
        print(f"Computing hints for {days} games starting from today")
        print()
        
        for offset in range(days):
            game_num = today_game + offset
            compute_hints_for_game(game_num)
    
    print("All done!")

if __name__ == "__main__":
    main()

