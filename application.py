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

def embed(text):
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

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
