from index_builder.cpu.create_cpu_index import create_index
from models.data_model import CreateIndexRequest, IndexTypes
from utils.decorators.timer import timer_func
from vector_data_accessor.accessor import VectorsDataset
from s3.s3client import upload_file, cleanup_temp_file
import logging
from timeit import default_timer as timer
import os

logger = logging.getLogger(__name__)
# Since this is an env property lets init this during the start of the service
build_type = os.getenv('INDEX_BUILD_TYPE', 'cpu')
index_type = IndexTypes.from_str(build_type.lower())

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
    index_file, create_index_stats = create_index(dataset, createIndexRequest)
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

@timer_func
def create_index(dataset: VectorsDataset, createIndexRequest:CreateIndexRequest):
    index_file = f"/tmp/{createIndexRequest.objectLocation}.faiss"
    space_type = "l2"
    create_index_stats = {}
    if index_type == IndexTypes.CPU:
        from index_builder.cpu.create_cpu_index import create_index
        index_file = index_file + f".{index_type.value}"
        hnsw_params = {}
        create_index_stats = create_index(dataset, hnsw_params, space_type, index_file)
    elif index_type == IndexTypes.GPU:
        index_file = index_file + f".{index_type.value}"
        indexing_params = {}
        from index_builder.gpu.create_gpu_index import create_index
        create_index_stats = create_index(dataset, indexing_params, space_type, index_file)
        # from index_builder.gpu.create_gpu_index import create_index
        # hnsw_params = {}
        # create_index_stats = create_index(dataset, hnsw_params, "l2", index_file)
    logger.info(f"Stats for the create Index request: {createIndexRequest} is : {create_index_stats}")
    return index_file, create_index_stats