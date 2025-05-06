import numpy as np
import math
import faiss
from timeit import default_timer as timer

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/files/vector_search.log'),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser()
parser.add_argument('filename')   
args = parser.parse_args()

dataset = np.load(args.filename)

logging.info("Dataset loaded from file {}, size {:.2f} GiB".format(args.filename, dataset.size*dataset.dtype.itemsize/(1<<30)))


res = faiss.StandardGpuResources()
res.noTempMemory()

metric = faiss.METRIC_L2

cagraIndexConfig = faiss.GpuIndexCagraConfig()
cagraIndexConfig.intermediate_graph_degree = 64
cagraIndexConfig.graph_degree = 32
cagraIndexConfig.device = 0
cagraIndexConfig.store_dataset = False
cagraIndexConfig.refine_rate = 1.0
cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ

# My addition
cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
cagraIndexIVFPQConfig.kmeans_trainset_fraction = 0.5
cagraIndexIVFPQConfig.kmeans_n_iters = 20
cagraIndexIVFPQConfig.pq_dim = int(1536/8)
cagraIndexIVFPQConfig.pq_bits = 8
cagraIndexIVFPQConfig.n_lists = int(math.sqrt(dataset.size))
cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig

cagraIndexSearchIVFPQConfig = faiss.IVFPQSearchCagraConfig()
cagraIndexSearchIVFPQConfig.n_probes = 20
#cagraIndexConfig.ivf_pq_search_params = cagraIndexSearchIVFPQConfig
    

logging.info("Creating CAGRA Index with IVF_PQ")
cagraIVFPQIndex = faiss.GpuIndexCagra(res, dataset.shape[1], metric, cagraIndexConfig)
t1 = timer()
cagraIVFPQIndex.train(dataset)
t2 = timer()
logging.info(f'Index Build executed in {(t2 - t1):.6f}s')
logging.info("Created CAGRA index")