import traceback

from flask import Flask, jsonify, request
from datetime import datetime
import json

from models import data_model
import uuid
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve

# Create logger
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

# Console handler (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = RotatingFileHandler(
    '/app/logs/waitress.log',
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

app = Flask(__name__)

# Sample in-memory store
indexing_jobs = {}


@app.route('/')
def hello():
    return jsonify({
        "message": "Hello from Vector Index Build Service!",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/status/<string:job_id>')
def status(job_id: str):
    job = indexing_jobs.get(job_id)
    if job is None:
        return jsonify({"error": f"Job not found with Id {job_id}"}), 404
    return jsonify({
        "status": job.status
    })


@app.route('/jobs', methods=['GET'])
def get_jobs():
    return json.dumps(indexing_jobs, default=lambda o: o.__dict__, indent=4)


@app.route('/create_index', methods=['POST'])
def create_index():
    try:
        logger.info(f"Received request: %s ", request.json)
        create_index_request = data_model.build_create_index_request(request.json)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Invalid request"}), 400

    indexing_job_id = str(uuid.uuid4())
    indexing_jobs[indexing_job_id] = create_index_request
    # TODO: start the create index process in async fashion
    return jsonify({"jobId": indexing_job_id}), 201


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5005)
