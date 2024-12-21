import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CreateIndexRequest:
    bucketName: str
    objectLocation: str
    numberOfVectors: int
    dimensions: int

def build_create_index_request(data: dict) -> CreateIndexRequest:
    if not all(key in data for key in ['bucket_name', 'object_location', 'number_of_vectors', 'dimensions']):
        raise ValueError("Missing required fields in JSON data")
    return CreateIndexRequest(
        bucketName=data['bucket_name'],
        objectLocation=data['object_location'],
        numberOfVectors=int(data['number_of_vectors']),
        dimensions=int(data['dimensions'])
    )