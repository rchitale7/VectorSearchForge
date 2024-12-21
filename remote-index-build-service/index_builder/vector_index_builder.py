from index_builder.cpu.create_cpu_index import create_index
from models.data_model import CreateIndexRequest
from vector_data_accessor.accessor import VectorsDataset
import logging
from timeit import default_timer as timer

logger = logging.getLogger(__name__)

def build_index(createIndexRequest: CreateIndexRequest):
    logger.info(f"Building index... with input: {createIndexRequest}")
    t1 = timer()
    dataset = VectorsDataset.get_vector_dataset(createIndexRequest)
    t2 = timer()
    stats = {
        "download_stats": {
            "time": t2 - t1, "unit": "seconds"
        }
    }
    hnsw_params = {}

    index_file = f"/tmp/{createIndexRequest.objectLocation}.faiss.cpu"
    create_index_stats = create_index(dataset, hnsw_params, "l2", index_file)
    logger.info(f"Stats for the create Index request: {createIndexRequest} is : {stats}")

    stats = {**stats, **create_index_stats}
    return stats