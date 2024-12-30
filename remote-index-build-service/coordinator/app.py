from flask import Flask, jsonify, request
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
from client.worker_client import WorkerService, Worker, RegisterWorkerRequest
import traceback
import os

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

def get_worker_from_seed_file(seed_file):
    with open(seed_file, 'r') as f:
        data = json.load(f)
        workers = []
        for worker in data:
            workers.append(Worker(worker['host'], worker['port']))
        logging.info(f"Workers are: {workers}")
        return workers

domain = os.getenv('DOMAIN', 'dev')
if domain == 'dev':
    logger.info("Running in dev mode")
    workers = get_worker_from_seed_file("workers_seed.json")
else:
    logger.info("Running in prod mode, workers will be added by user")
    workers = []

workerservice = WorkerService(workers=workers)

@app.route('/')
def hello():
    return jsonify({
        "message": "Hello from Vector Index Build Service Coordinator!",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/create_index', methods=['POST'])
def create_index():
    try:
        input = request.json
        response = workerservice.create_index(input)
        logger.info(f"Response is: {response}")
        return json.dumps(response, default=lambda o: o.__dict__, indent=4)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/jobs', methods=['GET'])
def get_jobs():
    try:
        return workerservice.get_jobs()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/job/<string:job_id>')
def job(job_id: str):
    try:
        jobs = workerservice.get_job(job_id)
        return json.dumps(jobs, default=lambda o: o.__dict__, indent=4), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/register_worker', methods=['POST'])
def register_worker():
    try:
        logger.info(f"Received request: %s ", request.json)
        register_worker_request = RegisterWorkerRequest.build_register_worker_request(request.json)
        workerservice.register_worker(register_worker_request)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Invalid request {request.json}"}), 400
    return jsonify({"message": "Worker registered successfully"}), 201

@app.route('/workers', methods=['GET'])
def get_all_worker():
    workers_list = {
        "workerList": workerservice.get_all_worker()
    }
    return json.dumps(workers_list, default=lambda o: o.__dict__, indent=4), 200


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=6006)

