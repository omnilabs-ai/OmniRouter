"""Shared configuration constants for the OmniLLM API."""
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('firebase-credentials.json')
app = firebase_admin.initialize_app(cred)
db = firestore.client()
VALID_API_KEYS = set()

def update_api_keys(keys_snapshot, changes, read_time):
    """Update the VALID_API_KEYS set when changes occur in Firestore."""
    global VALID_API_KEYS
    VALID_API_KEYS = {key.id for key in keys_snapshot}

initial_keys = db.collection('api_keys').get()
update_api_keys(initial_keys, None, None)
api_keys_watch = db.collection('api_keys').on_snapshot(update_api_keys)

PROVIDERS = {}
MAX_TOKENS = 1000000
