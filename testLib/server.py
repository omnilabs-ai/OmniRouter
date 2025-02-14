import uvicorn
import logging
from serverRouter.router import app
import os #added to get env variable

print("server OMNI_API_KEY is:", os.getenv("OMNI_API_KEY"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

print("server.py executed") #print to show server has executed