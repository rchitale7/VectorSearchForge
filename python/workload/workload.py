import yaml
import logging
import sys
import os

from python.data_types.data_types import IndexTypes, WorkloadTypes
from python.dataset import dataset_utils
from python.search import search_indices

logging.basicConfig(level=logging.INFO)


def runWorkload(workloadName: dict, indexType: IndexTypes, workloadType: WorkloadTypes):
    allWorkloads = readAllWorkloads()
    workloadToExecute = allWorkloads[indexType.value][workloadName]
    logging.info(workloadToExecute)
    dataset_file = dataset_utils.downloadDataSetForWorkload(workloadToExecute)
    if workloadType == WorkloadTypes.INDEX_AND_SEARCH or workloadType == WorkloadTypes.INDEX:
        doIndexing(workloadToExecute, dataset_file, indexType)

    if workloadType == WorkloadTypes.INDEX_AND_SEARCH or workloadType == WorkloadTypes.SEARCH:
        doSearch(workloadToExecute, dataset_file, indexType)

def doIndexing(workloadToExecute: dict, datasetFile: str, indexType: IndexTypes):
    logging.info("Run Indexing...")
    d, xb, ids = dataset_utils.prepare_indexing_dataset(datasetFile, workloadToExecute.get('normalize'))
    workloadToExecute["dimension"] = d
    for param in workloadToExecute['indexing-parameters']:
        prepare_env_for_indexing(workloadToExecute, indexType, param)
        timingMetrics = None
        logging.info(f"================ Running configuration: {param} ================")
        if indexType == IndexTypes.CPU:
            from python.indexing.cpu.create_cpu_index import indexData as indexDataInCpu
            indexDataInCpu(d, xb, ids, file_to_write=workloadToExecute["graph_file"])

        if indexType == IndexTypes.GPU:
            param["pq_dim"] = int(d / param['compression_factor'])
            from python.indexing.gpu.create_gpu_index import indexData as indexDataInGpu
            timingMetrics = indexDataInGpu(d, xb, ids, param, workloadToExecute["graph_file"])
        logging.info(f"===== Timing Metrics : {timingMetrics} ====")
        logging.info(f"================ Completed configuration: {param} ================")


def doSearch(workloadToExecute: dict, datasetFile: str, indexType: IndexTypes):
    logging.info("Running Search...")
    d, xq, gt = dataset_utils.prepare_search_dataset(datasetFile, workloadToExecute.get('normalize'))
    workloadToExecute["dimension"] = d
    for indexingParam in workloadToExecute["indexing-parameters"]:
        put_graph_file_name_in_workload(workloadToExecute, d, indexType, indexingParam)
        for searchParam in workloadToExecute['search-parameters']:
            logging.info(f"=== Running search for index config: {indexingParam} and search config: {searchParam}===")
            searchTimingMetrics = search_indices.runIndicesSearch(xq, workloadToExecute['graph_file'], searchParam, gt)
            logging.info(f"===== Timing Metrics : {searchTimingMetrics} ====")
            logging.info(f"=== Completed search for index config: {indexingParam} and search config: {searchParam}===")
            logging.info(f"=======")


def prepare_env_for_indexing(workloadToExecute: dict, indexType:IndexTypes, param:dict):
    if os.path.isdir("graphs") == False:
        os.makedirs("graphs")
    d = workloadToExecute["dimension"]
    put_graph_file_name_in_workload(workloadToExecute, d, indexType, param)
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

def put_graph_file_name_in_workload(workloadToExecute:dict, d:int, indexType:IndexTypes, param:dict):
    if indexType == IndexTypes.CPU:
        workloadToExecute["graph_file"] = os.path.join("graphs", f"{workloadToExecute['dataset_name']}_{d}.{indexType.value}_efconst_{param['ef_construction']}.graph")

    if indexType == IndexTypes.GPU:
        workloadToExecute["graph_file"] = os.path.join("graphs", f"{workloadToExecute['dataset_name']}_{d}.{indexType.value}_compressionFactor_{param['compression_factor']}.graph")