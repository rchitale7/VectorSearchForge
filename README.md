# VectorSearchForge

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

#### Building all the CPP files on GPU
```

cmake -B build . -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_RAFT=ON  -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=ON -GNinja -DCMAKE_CUDA_ARCHITECTURES=80  -DFAISS_ENABLE_GPU=ON -DCUDAToolkit_ROOT="/home/ubuntu/miniconda3/envs/faiss-gpu/lib"
```

```
cmake --build build --target cagra-gpu-index -j 10
```

### Updating Submodule
```
git submodule update --remote
```

### CPU Python
```conda create -n faiss-cpu  python=3.8```

#### For GPU
```
pip install numpy  swig==4.0.0 h5py psutil
```
for versions >=4.2.0 the faiss swig build will fail. GH issue: https://github.com/facebookresearch/faiss/issues/3239


```
cmake -B build . -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_RAFT=ON  -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=ON -GNinja -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DFAISS_ENABLE_GPU=ON
```

```
ninja -C build -j10 install faiss swigfaiss

```

```
cd build/external/faiss/faiss/python && python3 setup.py build

export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/"
```


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

export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/"

```