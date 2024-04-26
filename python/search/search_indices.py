import os.path

import faiss
import sys
import logging

from timeit import default_timer as timer
from python.decorators.timer import timer_func
from python.utils.common_utils import recall_at_r

def runIndicesSearch(xq, graphFile:str, param:dict, gt) -> dict:
    index:faiss.Index = loadGraphFromFile(graphFile)
    hnswParameters = faiss.SearchParametersHNSW()
    hnswParameters.efSearch = 100 if param.get('ef_search') is None else param['ef_search']
    k = 100 if param.get('K') is None else param['K']
    
    @timer_func
    def search(xq, k, params):
        D, ids = index.search(xq, k, params=params)
        return ids
    # K is always set to 100
    t1 = timer()
    I = search(xq, 100, hnswParameters)
    t2 = timer()

    recall_at_k = recall_at_r(I, gt, k, k, len(xq))
    recall_at_1 = recall_at_r(I, gt, 1, 1, len(xq))
    logging.info(f"Recall at {k} : is {recall_at_k}")
    logging.info(f"Recall at 1 : is {recall_at_1}")
    # deleting the index to avoid OOM
    # We don't need to set own_fileds = true as this will be automatically set by faiss while reading the index.
    del index
    return {
        "searchTime": t2 - t1,
        "units": "seconds",
        f"recall_at_{k}": recall_at_k,
        "recall_at_1": recall_at_1,
        "total_queries": len(xq)
    }



def loadGraphFromFile(graphFile: str) -> faiss.Index:
    if os.path.isfile(graphFile) is False:
        logging.error(f"The path provided: {graphFile} is not a file")
        sys.exit(0)

    return faiss.read_index(graphFile)