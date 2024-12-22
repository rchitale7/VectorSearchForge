from flask import Flask, jsonify
from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler
from waitress import serve

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console handler (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = RotatingFileHandler(
    '/app/logs/remote-index-build-service.log',
    maxBytes=1024 * 1024,
    backupCount=5
)
file_handler.setLevel(logging.INFO)

# Format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        "message": "Hello from Vector Index Build Service Coordinator!",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=6006)

