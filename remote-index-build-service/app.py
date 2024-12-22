import traceback

from flask import Flask, jsonify, request
from datetime import datetime
import json

from index_builder.indexing_service import IndexingService
from models import data_model
import uuid
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

indexing_service = IndexingService()


@app.route('/')
def hello():
    return jsonify({
        "message": "Hello from Vector Index Build Service!",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/job/<string:job_id>')
def job(job_id: str):
    job_deatils = indexing_service.get_job_status(job_id)
    if job_deatils is None:
        return jsonify({"error": f"Job not found with Id {job_id}"}), 404
    return jsonify({
        "status": job_deatils.status,
        "result": job_deatils.result,
        "error": job_deatils.error
    })


@app.route('/jobs', methods=['GET'])
def get_jobs():
    jobs = indexing_service.get_jobs()
    return json.dumps(jobs, default=lambda o: o.__dict__, indent=4)


@app.route('/create_index', methods=['POST'])
def create_index():
    try:
        logger.info(f"Received request: %s ", request.json)
        create_index_request = data_model.build_create_index_request(request.json)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Invalid request"}), 400

    indexing_job_id = str(uuid.uuid4())

    job_details = indexing_service.create_job(indexing_job_id, create_index_request)
    indexing_service.start_job(job_details.id, create_index_request)
    return jsonify({"job_id": job_details.id, "status": job_details.status}), 201


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=6005)
