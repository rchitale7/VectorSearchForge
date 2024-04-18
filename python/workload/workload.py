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
    d, xb, ids = dataset_utils.prepare_indexing_dataset(datasetFile, workloadToExecute['normalize'])
    for param in workloadToExecute['indexing-parameters']:
        if indexType == IndexTypes.CPU:
            workloadToExecute["graph_file"] = os.path.join("graphs", f"{workloadToExecute['dataset_name']}_{d}.{indexType.value}_{param['ef_construction']}.graph")
            logging.info(workloadToExecute["graph_file"])
            from python.cpu.create_cpu_index import indexData as indexDataInCpu
            indexDataInCpu(d, xb, ids, workloadToExecute["graph_file"])

        if indexType == IndexTypes.GPU:
            from python.gpu.create_gpu_index import indexData as indexDataInGpu
            indexDataInGpu(d, xb, ids)


def readAllWorkloads():
    with open("./python/benchmarks.yml") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit()
