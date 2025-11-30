from flask import Flask, render_template
import datetime
import hashlib
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import logging.handlers
import numpy as np
import pytz
import re
import urllib
import threading
import json
import os

def setup_logging():
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Handler 
    LOG_FILE = '/tmp/phrasegolf.log'
    handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=5)
    handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add Formatter to Handler
    handler.setFormatter(formatter)

    # add Handler to Logger
    logger.addHandler(handler)

    return logger

logger = setup_logging()

REPO_ID = "ChristianAzinn/e5-large-v2-gguf"
FILENAME_IN_REPO = "e5-large-v2.Q5_K_M.gguf"

LOCAL_FILENAME = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_IN_REPO)

model = Llama(model_path=LOCAL_FILENAME, embedding=True)

model_lock = threading.Lock()

def embed(text):
    with model_lock:
        return model.embed("query: " + text)

TARGETS_FILE = "targets.tsv"

targets = None

def load_targets():
    ret = []
    with open('targets.tsv', 'r') as targets_file:
        for line in targets_file:
            ret.append(line.strip())
    return ret

targets = load_targets()

# Load hint phrases (words + targets combined)
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
logger.info(f"Loaded {len(hint_phrases)} hint phrases")

# Cache directory for pre-computed hints
HINTS_CACHE_DIR = "hints_cache"
os.makedirs(HINTS_CACHE_DIR, exist_ok=True)

START_DATE = datetime.date(2024,7,2)

def get_game_num_for_today():
    now = datetime.datetime.now(pytz.timezone('US/Pacific'))
    three_hours_ago = now - datetime.timedelta(seconds=3*60*60)
    day = three_hours_ago.date()
    return (day - START_DATE).days

def get_target_for_game_num(game_num):
    # hash game_num as a decimal string
    hash_object = hashlib.md5(str(game_num).encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert the hash to an integer
    hash_int = int(hash_hex, 16)
    
    # Use the integer to select a target from the list
    return targets[hash_int % len(targets)]

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def similarity(game_num, guess):
    logger.info("Computing similarity for %s" % guess)
    target = get_target_for_game_num(game_num)
    target_embedding = embed(target)
    guess_embedding = embed(guess)
    return cosine_similarity(target_embedding, guess_embedding)

def letter_pattern(game_num):
    target = get_target_for_game_num(game_num)
    return re.sub(r'\S', '□', target)

def get_hints_cache_path(game_num):
    return os.path.join(HINTS_CACHE_DIR, f"game_{game_num}.json")

def get_or_compute_hints(game_num):
    """Get pre-computed hints for a game, computing them if necessary."""
    cache_path = get_hints_cache_path(game_num)
    
    # Check if cache exists
    if os.path.exists(cache_path):
        logger.info(f"Loading cached hints for game {game_num}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    # Compute similarities for all hint phrases
    logger.info(f"Computing hints for game {game_num} ({len(hint_phrases)} phrases)...")
    target = get_target_for_game_num(game_num)
    target_embedding = embed(target)
    
    hints = []
    for i, phrase in enumerate(hint_phrases):
        if i % 500 == 0:
            logger.info(f"  Progress: {i}/{len(hint_phrases)}")
        phrase_embedding = embed(phrase)
        sim = cosine_similarity(target_embedding, phrase_embedding)
        hints.append({"phrase": phrase, "similarity": float(sim)})
    
    # Sort by similarity (descending)
    hints.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Save to cache
    logger.info(f"Saving hints cache for game {game_num}")
    with open(cache_path, 'w') as f:
        json.dump(hints, f)
    
    return hints

def get_hint_better_than(game_num, best_similarity):
    """Get a hint that's better than the user's best similarity."""
    hints = get_or_compute_hints(game_num)
    target = get_target_for_game_num(game_num)
    
    # Find the first hint better than best_similarity that isn't the answer
    for hint in hints:
        if hint["phrase"] != target and hint["similarity"] > best_similarity:
            return hint
    
    # If no hint is better (edge case), return the best non-answer hint
    for hint in hints:
        if hint["phrase"] != target:
            return hint
    
    return None

application = Flask(__name__)

@application.route('/')
def index():
    game = get_game_num_for_today()
    return render_template('index.html',
                           game=game,
                           letter_pattern=letter_pattern(game))

application.add_url_rule('/game/<game>/guess/<guess>', 'similarity',
                         (lambda game, guess:
    {"similarity": similarity(game, urllib.parse.unquote(guess))}))

application.add_url_rule('/game/<game>/hint/<best_similarity>', 'hint',
                         (lambda game, best_similarity:
    get_hint_better_than(int(game), float(best_similarity))))

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
