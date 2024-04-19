import logging
import os
from urllib.request import urlretrieve
from python.decorators.timer import timer_func
from python.dataset.dataset import HDF5DataSet, Context
import numpy as np
import logging


def downloadDataSetForWorkload(workloadToExecute: dict) -> str:
    download_url = workloadToExecute["download_url"]
    dataset_name = workloadToExecute["dataset_name"]

    return downloadDataSet(download_url, dataset_name)


def downloadDataSet(download_url: str, dataset_name: str) -> str:
    logging.info("Downloading dataset...")
    destination_path = os.path.join("dataset", f"{dataset_name}.hdf5")
    if not os.path.exists(destination_path):
        logging.info(f"downloading {download_url} -> {destination_path}...")
        urlretrieve(download_url, destination_path)
        logging.info(f"downloaded {download_url} -> {destination_path}...")
    return destination_path


@timer_func
def prepare_indexing_dataset(datasetFile: str, normalize: bool = None) -> tuple[int, np.ndarray, list]:
    logging.info(f"Reading data set from file: {datasetFile}")
    index_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.INDEX)
    xb: np.ndarray = index_dataset.read(index_dataset.size())
    d: int = len(xb[0])
    ids = [i for i in range(len(xb))]
    if normalize:
        logging.info("Doing normalization...")
        xb = xb / np.linalg.norm(xb)
        logging.info("Completed normalization...")

    logging.info("Dataset info : ")
    logging.info(f"Dimensions: {d}")
    logging.info(f"Total Vectors: {len(xb)}")
    logging.info(f"Normalized: {normalize}")

    return d, xb, ids


@timer_func
def prepare_search_dataset(datasetFile: str, normalize: bool = None) -> tuple[int, np.ndarray, HDF5DataSet]:
    logging.info(f"Reading data set from file: {datasetFile}")
    search_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.QUERY)
    xq: np.ndarray = search_dataset.read(search_dataset.size())
    gt:HDF5DataSet = HDF5DataSet(datasetFile, Context.NEIGHBORS)
    d: int = len(xq[0])
    logging.info("Dataset info : ")
    logging.info(f"Dimensions: {d}")
    logging.info(f"Total Vectors: {len(xq)}")
    logging.info(f"Normalized: {normalize}")
    if normalize:
        logging.info("Doing normalization...")
        xq = xq / np.linalg.norm(xq)
        logging.info("Completed normalization...")
    return d, xq, gt