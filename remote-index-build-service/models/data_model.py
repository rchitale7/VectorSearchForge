import logging
from dataclasses import dataclass
import json

@dataclass
class CreateIndexRequest:
    bucketName: str
    objectLocation: str
    numberOfVectors: int
    dimensions: int

def build_create_index_request(data: dict) -> CreateIndexRequest:
    if not all(key in data for key in ['bucketName', 'objectLocation', 'numberOfVectors', 'dimensions']):
        raise ValueError("Missing required fields in JSON data")

    return CreateIndexRequest(
        bucketName=data['bucketName'],
        objectLocation=data['objectLocation'],
        numberOfVectors=int(data['numberOfVectors']),
        dimensions=int(data['dimensions'])
    )