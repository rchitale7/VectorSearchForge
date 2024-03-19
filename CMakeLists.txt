cmake_minimum_required(VERSION 3.23.1)

project(VectorSearchForge)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(FAISS_TEST_HNSW faiss-test)

set(BUILD_TESTING OFF)          # Avoid building faiss tests
set(BLA_STATIC ON)              # Statically link BLAS
set(FAISS_OPT_LEVEL generic)    # Keep optimization level generic
set(FAISS_ENABLE_PYTHON OFF)

if (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
    endif()
endif()

find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Check if faiss exists
find_path(FAISS_REPO_DIR NAMES faiss PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)

# If not, pull the updated submodule
if (NOT EXISTS ${FAISS_REPO_DIR})
    message(STATUS "Could not find faiss. Pulling updated submodule.")
    execute_process(COMMAND git submodule update --init -- external/faiss WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)

# Running the CAGRA Indices
if(FAISS_ENABLE_RAFT)
    set(FAISS_ENABLE_GPU ON)
    set(FAISS_ENABLE_RAFT ON)
    find_package(CUDAToolkit REQUIRED)
    ## Find the raft package
    find_package(raft COMPONENTS compiled distributed)
    add_executable(cagra-gpu-index EXCLUDE_FROM_ALL cagra-gpu-index.cpp)
    ## Add the raft dependencies
    target_link_libraries(cagra-gpu-index PRIVATE faiss CUDA::cudart raft::raft raft::compiled)

#    add_executable(check-id-map-working EXCLUDE_FROM_ALL check-id-map-working.cpp)
#    target_link_libraries(check-id-map-working PRIVATE faiss CUDA::cudart raft::raft raft::compiled)
else ()
    set(FAISS_ENABLE_GPU OFF)
endif()


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/faiss EXCLUDE_FROM_ALL)

add_executable(${FAISS_TEST_HNSW} ${CMAKE_CURRENT_SOURCE_DIR}/cpp/faiss-test.cpp)
target_link_libraries(${FAISS_TEST_HNSW} faiss OpenMP::OpenMP_CXX)
target_include_directories(${FAISS_TEST_HNSW} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)
set_target_properties(${FAISS_TEST_HNSW} PROPERTIES POSITION_INDEPENDENT_CODE ON)