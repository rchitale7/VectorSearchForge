cmake_minimum_required(VERSION 3.23.1)

project(VectorSearchForge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(BUILD_TESTING OFF)          # Avoid building faiss tests
set(BLA_STATIC ON)              # Statically link BLAS
set(FAISS_OPT_LEVEL generic)    # Keep optimization level generic
set(FAISS_ENABLE_GPU OFF)

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

# This add faiss as a subdirectory for all the files.

add_subdirectory(cpp)
# Running the GPU Indices
if(FAISS_ENABLE_RAFT)
    set(FAISS_ENABLE_GPU ON)
    find_package(CUDAToolkit REQUIRED)
    ## Find the raft package
    find_package(raft COMPONENTS compiled distributed)

    set(CAGRA_GPU_INDEX cagra-gpu-index)

    add_executable(${CAGRA_GPU_INDEX} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/cpp/gpu/cagra-gpu-index.cpp)
    ## Add the raft dependencies
    target_link_libraries(${CAGRA_GPU_INDEX} PRIVATE faiss CUDA::cudart raft::raft raft::compiled)
    #target_include_directories(${CAGRA_GPU_INDEX} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)
    set_target_properties(${CAGRA_GPU_INDEX} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/faiss EXCLUDE_FROM_ALL)