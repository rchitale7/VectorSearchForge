FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

USER root
RUN conda install rmm
USER appuser
RUN pip install h5py cupy-cuda12x tqdm

WORKDIR /cuvs_benchmarks
COPY . /cuvs_benchmarks

ENV PYTHONPATH=${PYTHONPATH}:/tmp/faiss/build/faiss/python:/
RUN ["python", "/cuvs_benchmarks/test_imports.py"]
CMD ["python", "main.py"]