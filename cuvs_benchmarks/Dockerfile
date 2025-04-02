FROM opensearchstaging/remote-vector-index-builder:base-latest

USER root
RUN apt-get update && apt-get upgrade -y && apt-get install libjemalloc2
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
USER appuser

WORKDIR /cuvs_benchmarks

RUN pip install h5py cupy-cuda12x tqdm
COPY . /cuvs_benchmarks

ENV PYTHONPATH=${PYTHONPATH}:/tmp/faiss/build/faiss/python:/
RUN ["python", "/cuvs_benchmarks/test_imports.py"]
USER root

USER appuser
CMD ["python", "main.py"]