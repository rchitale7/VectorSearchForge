import yaml
import logging
import sys
import os

from python.data_types.data_types import IndexTypes
from python.dataset import dataset_utils

logging.basicConfig(level=logging.INFO)


def runWorkload(workloadName: dict, indexType: IndexTypes):
    allWorkloads = readAllWorkloads()
    workloadToExecute = allWorkloads[indexType.value][workloadName]
    logging.info(workloadToExecute)
    dataset_file = dataset_utils.downloadDataSetForWorkload(workloadToExecute)
    doIndexing(workloadToExecute, dataset_file, indexType)


def doIndexing(workloadToExecute: dict, datasetFile: str, indexType: IndexTypes):
    logging.info("Run Indexing...")
    d, xb, ids = dataset_utils.prepare_indexing_dataset(datasetFile, workloadToExecute.get('normalize'))
    workloadToExecute["dimension"] = d
    for param in workloadToExecute['indexing-parameters']:
        prepare_env_for_indexing(workloadToExecute, indexType, param)
        timingMetrics = None
        logging.info(f"================ Running configuration: {param} ================")
        if indexType == IndexTypes.CPU:
            from python.cpu.create_cpu_index import indexData as indexDataInCpu
            indexDataInCpu(d, xb, ids, file_to_write=workloadToExecute["graph_file"])

        if indexType == IndexTypes.GPU:
            param["pq_dim"] = int(d / param['compression_factor'])
            from python.gpu.create_gpu_index import indexData as indexDataInGpu
            timingMetrics = indexDataInGpu(d, xb, ids, param, workloadToExecute["graph_file"])
        logging.info(f"===== Timing Metrics : {timingMetrics} ====")
        logging.info(f"================ Completed configuration: {param} ================")


def prepare_env_for_indexing(workloadToExecute: dict, indexType:IndexTypes, param:dict):
    if os.path.isdir("graphs") == False:
        os.makedirs("graphs")
    d = workloadToExecute["dimension"]
    if indexType == IndexTypes.CPU:
        workloadToExecute["graph_file"] = os.path.join("graphs", f"{workloadToExecute['dataset_name']}_{d}.{indexType.value}_efconst_{param['ef_construction']}.graph")

    if indexType == IndexTypes.GPU:
        workloadToExecute["graph_file"] = os.path.join("graphs", f"{workloadToExecute['dataset_name']}_{d}.{indexType.value}_compressionFactor_{param['compression_factor']}.graph")
    
    if os.path.exists(workloadToExecute["graph_file"]):
        logging.info(f"Removing file : {workloadToExecute['graph_file']}")
        os.remove(workloadToExecute["graph_file"])
    



def readAllWorkloads():
    with open("./python/benchmarks.yml") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit()
