# VectorSearchForge

### Setup C++
#### Building all the CPP files on CPU and Run simple test
```
cmake -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug -GNinja .  -DFAISS_ENABLE_PYTHON=OFF
```

```
cmake --build cmake-build-debug --target faiss-test -j 10
```

```
./cmake-build-debug/cpp/faiss-test
```

#### Building all the CPP files on GPU
```
cmake -B build . -DFAISS_ENABLE_RAFT=ON -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=OFF -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
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
cmake -B build -DCMAKE_BUILD_TYPE=Release -GNinja .  -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DFAISS_ENABLE_GPU=ON -GNinja
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
cmake -B build .
make -C build -j faiss swigfaiss

cd build/external/faiss/faiss/python && python3 setup.py build

export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/"



```