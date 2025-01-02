#!/bin/bash

# Exit on any error
set -xe

run_indexing() {
  echo "Running Indexing using ${1} and params: ${2}"
  opensearch-benchmark execute-test \
      --target-hosts $1     \
      --workload vectorsearch     \
      --workload-params $2     \
      --pipeline benchmark-only     \
      --kill-running-processes \
      --test-procedure=no-train-test-index-only
}

run_force_merge() {
  echo "Running Force Merge using ${1} and params: ${2}"
  opensearch-benchmark execute-test \
      --target-hosts $1     \
      --workload vectorsearch     \
      --workload-params $2     \
      --pipeline benchmark-only     \
      --kill-running-processes \
      --test-procedure=force-merge-index
}

run_search() {
  echo "Running Search using ${1} and params: ${2}"
  opensearch-benchmark execute-test \
      --target-hosts $1     \
      --workload vectorsearch     \
      --workload-params $2     \
      --pipeline benchmark-only     \
      --kill-running-processes \
      --test-procedure=search-only
}

enable_graph_builds() {
  echo "Flushing the index..."
  curl --request GET --url $1/target_index/_flush

  echo "Enabling Graph Builds..."
  curl --request PUT \
  --url $1/target_index/_settings \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/10.2.0' \
  --data '{
  "index.knn.advanced.approximate_threshold": "0"
  }'
}

setup_cluster() {
  echo "Setting up cluster..."
  curl --request PUT \
    --url $1/_cluster/settings \
    --header 'Content-Type: application/json' \
    --header 'User-Agent: insomnia/10.2.0' \
    --data '{
      "persistent": {
          "knn.remote.index.build.enabled": "true",
          "knn.remote.index.build.max_docs": 100000,
          "knn.s3.access.key": "'$2'",
          "knn.s3.secret.key": "'$3'",
          "knn.s3.token.key": null,
          "knn.s3.bucket": "remote-index-poc-bucket",
          "knn.remote.index.build.service.endpoint": "'$4'",
          "knn.remote.index.build.service.port": "'$5'"
      }
  }'
}

# Function to display usage
usage() {
    cat << EOF
Usage: $(basename $0) <options>

Options:
    -e, --endpoint            AWS endpoint
    -p, --params-file        Parameters file
    -a, --access-key         AWS access key
    -s, --secret-key         AWS secret key
    -c, --coordinator-endpoint    Coordinator endpoint
    -t, --coordinator-port       Coordinator port
    -h, --help              Show this help message

Example:
    $(basename $0) -e https://aws.endpoint.com -p params.json -a AKIAXXXXXX -s secretkey -c coordinator.example.com -t 8080
EOF
    exit 1
}

# Function to handle parameters
run_benchmark() {
    local endpoint=""
    local params_file=""
    local access_key=""
    local secret_key=""
    local coordinator_endpoint=""
    local coordinator_port=""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--endpoint)
                endpoint="$2"
                shift 2
                ;;
            -p|--params-file)
                params_file="$2"
                shift 2
                ;;
            -a|--access-key)
                access_key="$2"
                shift 2
                ;;
            -s|--secret-key)
                secret_key="$2"
                shift 2
                ;;
            -c|--coordinator-endpoint)
                coordinator_endpoint="$2"
                shift 2
                ;;
            -t|--coordinator-port)
                coordinator_port="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Validate required parameters
    if [[ -z "$endpoint" ]] || [[ -z "$params_file" ]] || [[ -z "$access_key" ]] || \
       [[ -z "$secret_key" ]] || [[ -z "$coordinator_endpoint" ]] || [[ -z "$coordinator_port" ]]; then
        echo "Error: Missing required parameters"
        usage
    fi

    # Export the variables
    export ENDPOINT="$endpoint"
    export PARAMS_FILE="$params_file"
    export ACCESS_KEY="$access_key"
    export SECRET_KEY="$secret_key"
    export COORDINATOR_ENDPOINT="$coordinator_endpoint"
    export COORDINATOR_PORT="$coordinator_port"

    setup_cluster $ENDPOINT $ACCESS_KEY $SECRET_KEY $COORDINATOR_ENDPOINT $COORDINATOR_PORT

    run_indexing  $ENDPOINT $PARAMS_FILE

    enable_graph_builds $ENDPOINT

    run_force_merge  $ENDPOINT $PARAMS_FILE

    run_search $ENDPOINT $PARAMS_FILE
}

# Main execution
if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
    # Script is being run directly
    if [[ $# -eq 0 ]]; then
        usage
    fi
    run_benchmark "$@"
fi
