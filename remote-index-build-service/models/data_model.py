import logging
from dataclasses import dataclass
from enum import Enum

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

class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def enumList(cls):
        return list(map(lambda c: c, cls))


class IndexTypes(ExtendedEnum):
    CPU = 'cpu'
    GPU = 'gpu'

    @staticmethod
    def from_str(labelstr: str) -> 'IndexTypes':
        if labelstr in 'cpu':
            return IndexTypes.CPU
        elif labelstr in 'gpu':
            return IndexTypes.GPU
        else:
            raise NotImplementedError
