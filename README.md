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
cmake -B build . -DFAISS_ENABLE_RAFT=ON -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=OFF -GNinja
```

```
cmake --build build --target cagra-gpu-index -j 10
```

### Updating Submodule
```
git submodule update --remote
```
