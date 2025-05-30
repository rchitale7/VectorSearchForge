import yaml
import logging
import sys
import os
from typing import List

from benchmarking.data_types.data_types import IndexTypes, WorkloadTypes
from benchmarking.dataset import dataset_utils
from benchmarking.memory_profiler.gpu_memory_profiler import GPUMemoryMonitor
from benchmarking.search import search_indices
from benchmarking.utils.common_utils import ensureDir
import json
import time
from tqdm import tqdm
import numpy as np
import copy

from core.index_builder.index_builder_utils import (
    calculate_ivf_pq_n_lists,
)

from core.common.models import VectorsDataset, SpaceType
from benchmarking.service.faiss_index_build_service import FaissIndexBuildService


def runWorkload(
    workloadNames: List[str], indexTypeStr: str, workloadType: WorkloadTypes
):
    allWorkloads = readAllWorkloads()
    indexTypesList = []
    if indexTypeStr == "all":
        indexTypesList = IndexTypes.enumList()
    else:
        indexTypesList.append(IndexTypes.from_str(indexTypeStr))

    for indexType in indexTypesList:
        # if workloadNames are empty, default to running all workloads
        if len(workloadNames) == 0:
            for currentWorkloadName in allWorkloads[indexType.value]:
                executeWorkload(
                    workloadName=currentWorkloadName,
                    workloadToExecute=allWorkloads[indexType.value][
                        currentWorkloadName
                    ],
                    indexType=indexType,
                    workloadType=workloadType,
                )
        else:
            for workloadName in workloadNames:
                executeWorkload(
                    workloadName=workloadName,
                    workloadToExecute=allWorkloads[indexType.value][workloadName],
                    indexType=indexType,
                    workloadType=workloadType,
                )


def executeWorkload(
    workloadName: str,
    workloadToExecute: dict,
    indexType: IndexTypes,
    workloadType: WorkloadTypes,
):
    workloadToExecute["indexType"] = indexType.value
    logging.info(workloadToExecute)
    dataset_file = dataset_utils.downloadDataSetForWorkload(workloadToExecute)
    allMetrics = {f"{workloadName}": {}}
    if (
        workloadType == WorkloadTypes.INDEX_AND_SEARCH
        or workloadType == WorkloadTypes.INDEX
    ):
        indexingMetrics = doIndexing(
            workloadToExecute, dataset_file, indexType, workloadType
        )
        allMetrics[workloadName] = {
            "workload-details": indexingMetrics["workload-details"],
            "indexingMetrics": indexingMetrics["indexing-metrics"],
        }

    if (
        workloadType == WorkloadTypes.INDEX_AND_SEARCH
        or workloadType == WorkloadTypes.SEARCH
    ):
        searchMetrics = doSearch(
            workloadToExecute, dataset_file, indexType, workloadType
        )
        allMetrics[workloadName]["searchMetrics"] = searchMetrics["search-metrics"]
        allMetrics[workloadName]["workload-details"] = searchMetrics["workload-details"]

    logging.info(json.dumps(allMetrics))
    persistMetricsAsJson(workloadType, allMetrics, workloadName, indexType)


def persistMetricsAsJson(
    workloadType: WorkloadTypes,
    allMetrics: dict,
    workloadName: str,
    indexType: IndexTypes,
):
    dir_path = ensureDir(f"results/{workloadName}")
    with open(f"{dir_path}/{workloadType.value}_{indexType.value}.json", "w") as file:
        json.dump(allMetrics, file, indent=4)


def doIndexing(
    workloadToExecute: dict,
    datasetFile: str,
    indexType: IndexTypes,
    workloadType: WorkloadTypes,
):
    logging.info("Run Indexing...")
    d, xb, ids = dataset_utils.prepare_indexing_dataset(
        datasetFile,
        workloadToExecute.get("normalize"),
        workloadToExecute.get("indexing-docs"),
    )

    vectors_dataset = VectorsDataset(xb, ids)

    workloadToExecute["dimension"] = d
    workloadToExecute["vectorsCount"] = len(xb)

    space_type = (
        SpaceType("l2")
        if workloadToExecute.get("space-type") is None
        else SpaceType(workloadToExecute.get("space-type"))
    )

    parameters_level_metrics = []
    for param in tqdm(workloadToExecute["indexing-parameters"]):
        if indexType == IndexTypes.GPU:
            for compression in workloadToExecute['compression']:
                param = copy.deepcopy(param)
                compression = int(compression)
                if compression != 0:
                    param["ivf_pq_params"]["pq_dim"] = int(workloadToExecute["dimension"] / compression)
                else:
                    param["ivf_pq_params"]["pq_dim"] = 0

                graph_file = prepare_env_for_indexing(workloadToExecute, indexType, param)
                timingMetrics = None
                metrics = {"indexing-param": param}
                logging.info(
                    f"================ Running configuration: {param} ================"
                )

                monitor = GPUMemoryMonitor(graph_file)

                try:
                    monitor.start_monitoring()

                    faiss_index_build_service = FaissIndexBuildService()
                    timingMetrics = faiss_index_build_service.build_index(
                        param,
                        workloadToExecute["search-parameters"][0],
                        vectors_dataset,
                        workloadToExecute,
                        graph_file,
                    )
                    metrics["indexing-timingMetrics"] = timingMetrics
                    time.sleep(3)

                finally:
                    monitor.stop_monitoring()
                    logging.info(json.dumps(monitor.memory_logs))
                    logging.info(json.dumps(monitor.cpu_memory_logs))
                    logging.info(json.dumps(monitor.ram_used_mb))
                    max_mem, start_mem, end_mem = monitor.log_metrics()
                    cpu_max, cpu_start, cpu_end = monitor.log_cpu_metrics()
                    metrics["memory_metrics"] = {
                        "timestamps": monitor.timestamps,
                        "gpu_memory_logs": monitor.memory_logs,
                        "start_time": monitor.start_time,
                        "gpu_id": monitor.gpu_id,
                        "ram_used_kb": monitor.ram_used_mb,
                        "interval": monitor.interval,
                        "peak_gpu_mem": max_mem - start_mem,
                        "leftover_gpu_mem": end_mem - start_mem,
                        "peak_cpu_mem": cpu_max - cpu_start,
                        "leftover_cpu_mem": cpu_end - cpu_start
                    }

                logging.info(f"===== Timing Metrics : {timingMetrics} ====")
                logging.info(
                    f"================ Completed configuration: {param} ================"
                )
                parameters_level_metrics.append(metrics)
                logging.info("Sleeping for 5 sec for better metrics capturing")
                time.sleep(5)

        else:
            graph_file = prepare_env_for_indexing(workloadToExecute, indexType, param)
            metrics = {"indexing-param": param}
            logging.info(
                f"================ Running configuration: {param} ================"
            )
            from benchmarking.indexing.cpu.create_cpu_index import (
                indexData as indexDataInCpu,
            )

            timingMetrics = indexDataInCpu(
                d, xb, ids, param, space_type, file_to_write=graph_file
            )

            logging.info(f"===== Timing Metrics : {timingMetrics} ====")
            logging.info(
                f"================ Completed configuration: {param} ================"
            )
            parameters_level_metrics.append(metrics)
            logging.info("Sleeping for 5 sec for better metrics capturing")
            time.sleep(5)

    del vectors_dataset
    del xb
    del ids
    return {
        "workload-details": workloadToExecute,
        "indexing-metrics": parameters_level_metrics,
    }



def doSearch(
    workloadToExecute: dict,
    datasetFile: str,
    indexType: IndexTypes,
    workloadType: WorkloadTypes,
):
    logging.info("Running Search...")
    d, xq, gt = dataset_utils.prepare_search_dataset(
        datasetFile, workloadToExecute.get("normalize")
    )
    workloadToExecute["dimension"] = d
    workloadToExecute["queriesCount"] = len(xq)
    parameters_level_metrics = []
    for indexingParam in workloadToExecute["indexing-parameters"]:
        if indexType == IndexTypes.GPU:
            for compression in workloadToExecute['compression']:
                indexingParam = copy.deepcopy(indexingParam)
                if compression != 0:
                    indexingParam["ivf_pq_params"]["pq_dim"] = int(workloadToExecute["dimension"] / compression)
                else:
                    indexingParam["ivf_pq_params"]["pq_dim"] = 0

                dir_path = ensureDir("graphs")
                d = workloadToExecute["dimension"]
                graph_file = get_graph_file(workloadToExecute, d, indexType, indexingParam, dir_path)


                for searchParam in workloadToExecute["search-parameters"]:
                    logging.info(
                        f"=== Running search for index config: {indexingParam} and search config: {searchParam}==="
                    )
                    searchTimingMetrics = search_indices.runIndicesSearch(
                        xq, graph_file, searchParam, gt
                    )
                    logging.info(f"===== Timing Metrics : {searchTimingMetrics} ====")
                    logging.info(
                        f"=== Completed search for index config: {indexingParam} and search config: {searchParam}==="
                    )
                    logging.info(f"=======")
                    parameters_level_metrics.append(
                        {
                            "indexing-params": indexingParam,
                            "search-timing-metrics": searchTimingMetrics,
                            "search-params": searchParam,
                        }
                    )
                    logging.info("Sleeping for 5 sec for better metrics capturing")
                    time.sleep(5)
        else:
            dir_path = ensureDir("graphs")
            d = workloadToExecute["dimension"]
            graph_file = get_graph_file(workloadToExecute, d, indexType, indexingParam, dir_path)

            for searchParam in workloadToExecute["search-parameters"]:
                logging.info(
                    f"=== Running search for index config: {indexingParam} and search config: {searchParam}==="
                )
                searchTimingMetrics = search_indices.runIndicesSearch(
                    xq, graph_file, searchParam, gt
                )
                logging.info(f"===== Timing Metrics : {searchTimingMetrics} ====")
                logging.info(
                    f"=== Completed search for index config: {indexingParam} and search config: {searchParam}==="
                )
                logging.info(f"=======")
                parameters_level_metrics.append(
                    {
                        "indexing-params": indexingParam,
                        "search-timing-metrics": searchTimingMetrics,
                        "search-params": searchParam,
                    }
                )
                logging.info("Sleeping for 5 sec for better metrics capturing")
                time.sleep(5)

    del xq
    del gt
    return {
        "workload-details": workloadToExecute,
        "search-metrics": parameters_level_metrics,
    }


def prepare_env_for_indexing(
    workloadToExecute: dict, indexType: IndexTypes, param: dict
):
    dir_path = ensureDir("graphs")
    d = workloadToExecute["dimension"]
    graph_file = get_graph_file(workloadToExecute, d, indexType, param, dir_path)
    if os.path.exists(graph_file):
        logging.info(f"Removing file : {graph_file}")
        os.remove(graph_file)
    return graph_file


def readAllWorkloads():
    with open("/benchmarking/benchmarks.yml") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit()


def get_graph_file(
        workloadToExecute: dict, d: int, indexType: IndexTypes, param: dict, dir_path: str
):
    str_to_build = f"{workloadToExecute['dataset_name']}_{d}.{indexType.value}"
    sorted_param_keys = sorted(param.keys())
    for key in sorted_param_keys:
        value = str(param[key])
        special_chars_to_remove = " {}',"
        for character in special_chars_to_remove:
            value = value.replace(character, "")
        value = value.replace(":", "_")
        str_to_build += f"_{key}_{value}"
    str_to_build += ".graph"
    return os.path.join(dir_path, str_to_build)
