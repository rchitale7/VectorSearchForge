### Benchmarking

This folder contains the code needed to benchmark GPU vs CPU build, using a docker container
on a GPU machine. This code does not use `opensearch-benchmark`, it uses a custom `main.py`
to run the experiments. 

### How to run the code

1. Make sure you `cd` into the `benchmarking` directory. 
2. Build the docker image on local computer: `docker build -t <your-dockerhub-repo>:<tag> . `
3. Push the docker image to your dockerhub repo: `docker push <your-dockerhub-repo>:<tag>`
4. Provision EC2 instance
    - If you are benchmarking GPU Faiss code
        - Use the `Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)`
        - Use g5.2xlarge instances (although for the cohere 10M vector 768 vector dataset, you may need a bigger instance)
    - If you are benchmarking CPU Faiss code, use the AL2023 AMI and r5.2xlarge
5. Connect to the EC2 instance in a terminal, then do `sudo bash`
6. Pull the docker image: `docker pull <your-dockerhub-repo>:<tag>`
   - One such docker image is here: `rchitale7/remote-index-build-service:vector-benchmark`
7. Create a directory in your root directory for docker mountpoint: `mkdir /docker-mountpoint`
8. Change the permissions on the directory: `chmod 777 /docker-mountpoint`
9. Create a file to hold environment variables. Call this file 'env_variables' 
The environment variables are:
   - `workload`: name of the dataset (such as `sift-128`). Can be a comma separated list of datasets, 
   to benchmark with multiple datasets
   - `workload_type`: Can be `INDEX`, `INDEX_AND_SEARCH`, or `INDEX`. Defaults to `INDEX_AND_SEARCH`
   - `index_type`: Can be `gpu`, `cpu`, or `all`
   - `run_id`: Sub-folder to store results. Defaults to `default_run_id`
   - For example, to run the GPU Faiss benchmarks with sift-128 for indexing and searching, 
   the environment variables file will look like:
   ```
   index_type=gpu
   workload=sift-128,gist-960
   ```
10. Run the docker container: `docker run -e env_variables -v /docker-mountpoint:/benchmarking/files --gpus all <your-dockerhub-repo>:<tag>`
    - One such docker image is here: `rchitale7/remote-index-build-service:vector-benchmark`
    - You can run the docker container in the background with `-d` option
    - Note that downloading the sift-128 and gist-960 datasets may fail. In that case, manually download 
    the files to the `/docker_mountpoint/{run_id}/dataset` folder
11. All files will be stored in `/docker_mountpoint/{run_id}` directory