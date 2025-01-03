## Introduction
The folder contains some scripts to run the OSB targeted to an Opensearch cluster.

## Running benchmarks
1. Make sure that you have python installed with version == 3.11
2. Now install the opensearch benchmarks via pip
    ```
    pip install opensearch-benchmark
    ```
3. Now you can run the script to trigger the benchmarks
    ```
    export ENDPOINT= OS endpoint with http and port present
    export PARAMS_FILE= Full path of params file
    export ACCESS_KEY= AWS access key for connecting to s3 bucket
    export SECRET_KEY= AWS secret key for connecting to s3 bucket
    export REMOTE_INDEX_BUILD_COORDINATOR= enpoint without http to be used for hitting remote index build service coordinator
    export PORT= Port to be used with remote index build service coordinator
    
    
    bash run_benchmarks.sh -e $ENDPOINT -p $PARAMS_FILE -a <ACCESS_KEY> -s <SECRET_KEY> -c <REMOTE_INDEX_BUILD_COORDINATOR> -t <PORT>
    ```