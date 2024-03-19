# VectorSearchForge

### Setup C++
#### Building all the CPP files on CPU
```
cmake -DCMAKE_BUILD_TYPE=Debug -G Ninja . -B cmake-build-debug
```

#### Building all the CPP files on GPU
```
cmake -DCMAKE_BUILD_TYPE=Debug -G Ninja . -B cmake-build -DFAISS_ENABLE_RAFT=ON -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=OFF
```

#### Running Simple Faiss HNSW Test
```
cmake --build cmake-build-debug --target faiss-test -j 10
```

```
./cmake-build-debug/faiss-test
```

### Updating Submodule
```
git submodule update --remote
```
