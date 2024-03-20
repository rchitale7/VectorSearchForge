### Faiss Test
set(FAISS_TEST_HNSW faiss-test)
add_executable(${FAISS_TEST_HNSW} faiss-test.cpp)
target_link_libraries(${FAISS_TEST_HNSW} faiss OpenMP::OpenMP_CXX)
#target_include_directories(${FAISS_TEST_HNSW} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)
set_target_properties(${FAISS_TEST_HNSW} PROPERTIES POSITION_INDEPENDENT_CODE ON)


### Test for Id mapping
add_executable(run-gpu-graph-on-cpu EXCLUDE_FROM_ALL run-gpu-graph-on-cpu.cpp)
target_link_libraries(run-gpu-graph-on-cpu PRIVATE faiss OpenMP::OpenMP_CXX)