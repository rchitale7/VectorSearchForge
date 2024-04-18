import logging
import os
from urllib.request import urlretrieve
from python.decorators.timer import timer_func
from python.dataset.dataset import HDF5DataSet, Context
import numpy as np
import logging


def downloadDataSetForWorkload(workloadToExecute: dict):
    download_url = workloadToExecute["download_url"]
    dataset_name = workloadToExecute["dataset_name"]

    return downloadDataSet(download_url, dataset_name)


def downloadDataSet(download_url: str, dataset_name: str):
    logging.info("Downloading dataset...")
    destination_path = os.path.join("dataset", f"{dataset_name}.hdf5")
    if not os.path.exists(destination_path):
        logging.info(f"downloading {download_url} -> {destination_path}...")
        urlretrieve(download_url, destination_path)
        logging.info(f"downloaded {download_url} -> {destination_path}...")
    return destination_path


@timer_func
def prepare_indexing_dataset(datasetFile: str, normalize: bool = None):
    logging.info(f"Reading data set from file: {datasetFile}")
    index_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.INDEX)
    xb: np.ndarray = index_dataset.read(index_dataset.size())
    d: int = len(xb[0])
    logging.info(f"Dimensions: {d} for dataset file: {datasetFile}")
    logging.info(f"Dataset size: {len(xb)}")
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
