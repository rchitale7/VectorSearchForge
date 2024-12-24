FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install base utilities and also the dependencies for faiss
RUN apt-get update \
    && apt-get install -y build-essential gfortran libblas-dev libopenblas-base zlib1g zlib1g-dev git tmux libssl-dev \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser 

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN chown -R appuser:appuser /opt
#RUN chmod -R 755 /opt

USER appuser
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
#install some necessary dependencies
RUN conda install -c pytorch -c nvidia -c rapidsai -c conda-forge python=3.11 cmake make -y
RUN cmake --version

# special for 12.1.1
ENV CUDA_ARCHS "70-real;72-real;75-real;80;86-real"

# install base packages for X86_64
RUN conda install -y -q -c conda-forge gxx_linux-64 sysroot_linux-64

# install all the dependencies that we think we will need for installing faiss
RUN conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libraft=24.06 cuda-nvcc swig numpy scipy pytest gflags -y

COPY faiss-test.py /tmp
COPY faiss /tmp


CMD ["sh", "-c", "nvidia-smi && sleep 36000"]
