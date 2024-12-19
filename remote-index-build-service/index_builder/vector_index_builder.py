from dataclasses import dataclass

from index_builder.cpu.create_cpu_index import create_index
from models.data_model import CreateIndexRequest
from vector_data_accessor.accessor import VectorsDataset


def build_index(createIndexRequest: CreateIndexRequest):
    print("Building index...")
    print(createIndexRequest)
    dataset = VectorsDataset.get_vector_dataset(createIndexRequest)
    hnsw_params = {}
    index_file = f"/tmp/{createIndexRequest.objectLocation}.faiss.cpu"
    stats = create_index(dataset, hnsw_params, "l2", index_file)
    print(f"Stats for the create Index request: {createIndexRequest} is : {stats}")