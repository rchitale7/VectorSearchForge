//
// Created by Verma, Navneet on 3/14/24.
//


#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <cstddef>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <optional>
#include <vector>
#include "faiss/IndexIDMap.h"

#include <raft/core/resource/cuda_stream.hpp>
#include "faiss/IndexHNSW.h"
#include "faiss/index_io.h"
#include <iostream>

struct Options {
    Options() {
        numTrain = 6000000;
        dim = 768;

        graphDegree = 32;
        intermediateGraphDegree = 64;
        buildAlgo = faiss::gpu::graph_build_algo::IVF_PQ;

        numQuery = 1;
        k = 5;

        device = faiss::gpu::getNumDevices() - 1;
    }

    std::string toString() const {
        std::stringstream str;
        str << "CAGRA device " << device << " numVecs " << numTrain << " dim "
            << dim << " graphDegree " << graphDegree
            << " intermediateGraphDegree " << intermediateGraphDegree
            << "buildAlgo " << static_cast<int>(buildAlgo) << " numQuery "
            << numQuery << " k " << k;

        return str.str();
    }

    int numTrain;
    int dim;
    size_t graphDegree;
    size_t intermediateGraphDegree;
    faiss::gpu::graph_build_algo buildAlgo;
    int numQuery;
    int k;
    int device;
};

int main() {
    Options opt;

    std::vector<float> trainVecs;
    trainVecs.reserve(opt.numTrain * opt.dim);

    for(long long i = 0 ; i < opt.numTrain; i++) {
        for(long long j = 0 ; j < opt.dim; j++) {
            trainVecs.push_back(i + j);
        }
    }

    std::vector<faiss::idx_t> ids;
    for(long long i = 0 ; i < opt.numTrain; i++) {
        ids.push_back(opt.numTrain - i);
    }

    std::cout<<"Vector are: "<<trainVecs.size()/opt.dim<< std::endl;

//    for(int i = 0 ; i < opt.numTrain ; i ++) {
//        std::cout<<"Id " << ids[i] << " [";
//        for(int j = 0 ; j < opt.dim; j++) {
//            std::cout<<trainVecs[(i * opt.dim) + j] << ", ";
//        }
//        std::cout<<"]"<<std::endl;
//    }

    // train gpu index
    faiss::gpu::StandardGpuResources res;
    //res.noTempMemory();

    faiss::gpu::GpuIndexCagraConfig config;
    config.device = opt.device;
    config.graph_degree = opt.graphDegree;
    config.intermediate_graph_degree = opt.intermediateGraphDegree;
    config.build_algo = opt.buildAlgo;
    config.store_dataset = false;
    faiss::gpu::IVFPQBuildCagraConfig ivfpqBuildCagraConfig;
    ivfpqBuildCagraConfig.kmeans_n_iters = 10;
    ivfpqBuildCagraConfig.n_lists = (int) sqrt((double) opt.numTrain);
    ivfpqBuildCagraConfig.pq_bits = 8;
    // 32x compression
    ivfpqBuildCagraConfig.pq_dim = opt.dim / 32 ;

    config.ivf_pq_params = &ivfpqBuildCagraConfig;
    std::cout<<"Building graph: " << std::endl;
    faiss::gpu::GpuIndexCagra gpuIndex(
            &res, opt.dim, faiss::METRIC_L2, config);
    gpuIndex.train(opt.numTrain, trainVecs.data());
    //faiss::IndexIDMap idMapIndex = faiss::IndexIDMap(&gpuIndex);
    //std::cout<<"Adding ids: " << std::endl;
    // Train the index
    //idMapIndex.add_with_ids(opt.numTrain, trainVecs.data(), ids.data());
    trainVecs.clear();
    std::cout<<"Added ids: " << std::endl;
    faiss::IndexHNSWCagra cpuCagraIndex;
    std::cout<<"Converting to CPU graph: " << std::endl;
    //faiss::gpu::GpuIndexCagra *mappedGpuIndex = dynamic_cast<faiss::gpu::GpuIndexCagra*>(gpuIndex);
    gpuIndex.copyTo(&cpuCagraIndex);

    //idMapIndex.index = &cpuCagraIndex;

    auto *indexToBeWritten = dynamic_cast<faiss::Index*>(&cpuCagraIndex);
    faiss::write_index(indexToBeWritten, "/tmp/cagraindex-test-with-ids.txt");
    std::cout<<"index is written"<<std::endl;
    return 0;
}
