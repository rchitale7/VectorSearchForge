from index_builder.cpu.create_cpu_index import create_index
from models.data_model import CreateIndexRequest
from utils.decorators.timer import timer_func
from vector_data_accessor.accessor import VectorsDataset
from s3.s3client import upload_file, cleanup_temp_file
import logging
from timeit import default_timer as timer

logger = logging.getLogger(__name__)

@timer_func
def build_index_and_upload_index(createIndexRequest: CreateIndexRequest):
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
    t1 = timer()
    upload_file(index_file, f"{createIndexRequest.objectLocation}.faiss.cpu",  createIndexRequest.bucketName)
    cleanup_temp_file(index_file)
    t2 = timer()
    stats["upload_stats"] = {
        "time": t2 - t1, "unit": "seconds"
    }
    logger.info(f"Index file uploaded for request: {createIndexRequest}")
    stats["create_index"] = create_index_stats
    return stats