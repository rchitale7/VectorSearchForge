from dataclasses import dataclass
import json

@dataclass
class CreateIndexRequest:
    bucketName: str
    objectLocation: str
    numberOfVectors: int
    dimensions: int

def build_create_index_request(json_data: str) -> str:
    """
    Process a JSON string containing a createIndex request and return a formatted string
    representing the CreateIndexRequest object.

    The function should:
    1. Parse the JSON string into a Python dictionary.
    2. Create an instance of the CreateIndexRequest class using the parsed data.
    3. Return a formatted string representation of the CreateIndexRequest object.

    Args:
    json_data (str): A JSON string containing the createIndex request data.

    Returns:
    str: A formatted string representing the CreateIndexRequest object.

    Raises:
    json.JSONDecodeError: If the input is not a valid JSON string.
    ValueError: If the required fields are missing from the JSON data.
    """
    try:
        data = json.loads(json_data)
        if not all(key in data for key in ['bucketName', 'objectLocation', 'numberOfVectors', 'dimensions']):
            raise ValueError("Missing required fields in JSON data")

        request = CreateIndexRequest(
            bucketName=data['bucketName'],
            objectLocation=data['objectLocation'],
            numberOfVectors=int(data['numberOfVectors']),
            dimensions=int(data['dimensions'])
        )

        return (f"CreateIndexRequest(bucketName='{request.bucketName}', "
                f"objectLocation='{request.objectLocation}', "
                f"numberOfVectors={request.numberOfVectors}, "
                f"dimensions={request.dimensions})")
    except json.JSONDecodeError:
        raise json.JSONDecodeError("Invalid JSON string", json_data, 0)