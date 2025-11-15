from flask import Flask, render_template, request, jsonify
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
import secrets
import time

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

# Session management for multiplayer
sessions = {}  # session_id -> {'game_num': int, 'guesses': [...], 'last_activity': timestamp}
sessions_lock = threading.Lock()

SESSION_TIMEOUT = 24 * 60 * 60  # 24 hours in seconds

def cleanup_old_sessions():
    """Remove sessions that haven't been active in SESSION_TIMEOUT seconds"""
    with sessions_lock:
        current_time = time.time()
        expired_sessions = [
            sid for sid, data in sessions.items()
            if current_time - data['last_activity'] > SESSION_TIMEOUT
        ]
        for sid in expired_sessions:
            del sessions[sid]
            logger.info(f"Cleaned up expired session: {sid}")

def create_session(game_num):
    """Create a new shared session and return its ID"""
    session_id = secrets.token_urlsafe(16)
    with sessions_lock:
        sessions[session_id] = {
            'game_num': game_num,
            'guesses': [],
            'last_activity': time.time()
        }
    logger.info(f"Created session {session_id} for game {game_num}")
    return session_id

def get_session_guesses(session_id):
    """Get all guesses for a session"""
    with sessions_lock:
        if session_id not in sessions:
            return None
        sessions[session_id]['last_activity'] = time.time()
        return sessions[session_id]['guesses'].copy()

def add_session_guess(session_id, guess, similarity):
    """Add a guess to a session"""
    with sessions_lock:
        if session_id not in sessions:
            return False
        sessions[session_id]['guesses'].insert(0, {
            'guess': guess,
            'similarity': similarity,
            'timestamp': time.time()
        })
        sessions[session_id]['last_activity'] = time.time()
    logger.info(f"Added guess to session {session_id}: {guess}")
    return True

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

@application.route('/session/create', methods=['POST'])
def create_session_endpoint():
    """Create a new shared session"""
    data = request.get_json()
    game_num = data.get('game_num')
    if game_num is None:
        return jsonify({'error': 'game_num required'}), 400
    
    session_id = create_session(int(game_num))
    cleanup_old_sessions()  # Clean up old sessions when creating new ones
    return jsonify({'session_id': session_id})

@application.route('/session/<session_id>/guesses', methods=['GET'])
def get_session_guesses_endpoint(session_id):
    """Get all guesses for a session"""
    guesses = get_session_guesses(session_id)
    if guesses is None:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'guesses': guesses})

@application.route('/session/<session_id>/guess', methods=['POST'])
def add_session_guess_endpoint(session_id):
    """Add a guess to a session"""
    data = request.get_json()
    guess = data.get('guess')
    if not guess:
        return jsonify({'error': 'guess required'}), 400
    
    # Get the game number for this session
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        game_num = sessions[session_id]['game_num']
    
    # Calculate similarity
    sim = similarity(game_num, guess)
    
    # Add to session
    if not add_session_guess(session_id, guess, sim):
        return jsonify({'error': 'Failed to add guess'}), 500
    
    return jsonify({'similarity': sim})

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
