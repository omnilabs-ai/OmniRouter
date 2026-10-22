"""Shared configuration constants for the OmniLLM API."""
import os

import firebase_admin
from firebase_admin import credentials, firestore

# The service-account file is never committed. Point FIREBASE_CREDENTIALS_PATH at it, or leave
# it unset to use Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS, gcloud auth,
# or the attached service account when running on Google Cloud).
_cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
cred = credentials.Certificate(_cred_path) if _cred_path else credentials.ApplicationDefault()
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
MAX_TOKENS = 100000
