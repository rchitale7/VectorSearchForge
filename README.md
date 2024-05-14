# VectorSearchForge

## Setting Up Environment on a GPU
0. Install miniconda on the machine using link: https://docs.anaconda.com/free/miniconda/#quick-command-line-install
1. Get latest conda dependencies list from https://github.com/rapidsai/raft file `conda/environments/all_cuda-122_arch-x86_64.yaml` in your local, or if you need all the correct dependenices you can use the `linux_x64_cuda12_raft24_06.yml`
2. Run the below command to create then env with name raft_246, you can use any name here.
```
conda env create --name raft_246 --file all_cuda-122_arch-x86_64-24-06.yaml
```
or
```
conda env create --name raft_246 --file linux_x64_cuda12_raft24_06.yml
```

3.(Not required if you use `linux_x64_cuda12_raft24_06.yml` in step 2) Install right version of `libraft` in the env. Use `rapidsai-nightly` to get latest build and `rapidsai` for stable builds. Use `Optional-Libraft-Version` to set version of libraft you need.

```
conda install -y -q libraft=<Optional-Libraft-Version> -c rapidsai-nightly  -c conda-forge
```
4. Activate the env.
```
conda activate raft_246
```

## Updating Submodule
```
git submodule update --remote
```

## Build Package on GPU
### Common for both C++ and Python
Replace ENV_NAME with conda env name in the command.

```
cmake -B build . -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_RAFT=ON  -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=ON -GNinja -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DFAISS_ENABLE_GPU=ON -DCUDAToolkit_ROOT="/home/ubuntu/miniconda3/envs/<ENV_NAME>/lib"
```

### Python

```
pip install numpy  swig==4.0.0 h5py psutil
```
for versions >=4.2.0 the faiss swig build will fail. GH issue: https://github.com/facebookresearch/faiss/issues/3239

```
ninja -C build -j10 install faiss swigfaiss

```

```
cd build/external/faiss/faiss/python && python3 setup.py build
```
Now go to the root of the project.
```
export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/"
```

## Setting up env for CPU Machine
0. Run the below commands to install the correct packages. Some of these things are optional(like tmux)
```
sudo yum install gcc-c++ gcc g++ tmux git zlib-devel openblas-devel gfortran python3-devel -y
```
1. Install miniconda on the machine using link: https://docs.anaconda.com/free/miniconda/#quick-command-line-install
2. Run the below command to create then env with name faiss-cpu, you can use any name here.
```
conda env create --name faiss-cpu --file linux_arm_cpu.yml
```
3. Now activate the env
```
conda activate faiss-cpu
```
## Build Package on CPU
### Common for both C++ and Python
Replace ENV_NAME with conda env name in the command.

```
cmake -B build . -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_RAFT=OFF"
```


### Python

```
pip install numpy  swig==4.0.0 h5py psutil
```
for versions >=4.2.0 the faiss swig build will fail. GH issue: https://github.com/facebookresearch/faiss/issues/3239

```
make -C build -j faiss swigfaiss
```

```
cd build/external/faiss/faiss/python && python3 setup.py build
```
Now go to the root of the project.
```
export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/"
```



## Running Benchmarks

### Setup
```
export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/"
```

### Indexing GPU
```
python python/main.py --workload=sift-128 --index_type=gpu --workload_type=index
```

### Indexing CPU
```
python python/main.py --workload=sift-128 --index_type=cpu --workload_type=index
```

### Run Both indexing and search 

*GPU*
```
python python/main.py --workload=gist-960 --index_type=gpu --workload_type=index_and_search
```

*CPU*
```
python python/main.py --workload=gist-960 --index_type=cpu --workload_type=index_and_search
```

### Exporting All results as CSV
```
python python/results.py --workload=all --index_type=gpu --workload_type=index_and_search
```
After this command the results will be stored under `results/all/all_results.csv`



## Old 
### Setup C++
#### Building all the CPP files on CPU and Run simple test
```
cmake -B cmake-build-debug -DCMAKE_BUILD_TYPE=Release -GNinja .  -DFAISS_ENABLE_PYTHON=OFF
```

```
cmake --build cmake-build-debug --target faiss-test -j 10
```

```
./cmake-build-debug/cpp/faiss-test
```

### CPU Python
```conda create -n faiss-cpu  python=3.11.8```

#### Not working For CPU
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -GNinja .  -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_GPU=OFF -GNinja
```

```
/Applications/CLion.app/Contents/bin/ninja/mac/x64/ninja -C cmake-build-debug -j4 install

```

```
cd build/external/faiss/faiss/python && python3 setup.py build
```

```
cd cmake-build-debug/external/faiss/faiss/python
```

### Working for CPU
```
cmake -B build . -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_RAFT=OFF
make -C build -j faiss swigfaiss

cd build/external/faiss/faiss/python && python3 setup.py build

export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/:$(/usr/local/dcgm/bindings/python3)"

```