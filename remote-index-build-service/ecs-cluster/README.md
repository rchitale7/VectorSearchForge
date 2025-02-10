## Introduction
This package provide the details on how to set up an ECS cluster for remote index build service.

## Setting up ECS cluster

### Prerequisite
1. Ensure that coordinator and worker images are already present in the ECR.

### Building Cluster
1. Export your AWS credentials on the command line.
2. `npm install`

#### Bootstrap Cluster
```
cdk bootstrap aws://<aws-account>/<aws-region> --context region=<aws-region> --context account=<aws-account> --context prefix=<prefix> \
--context worker-image="<ECR image URI of the worker>" \
--context security-group=<Security Group to be used> --context vpc=<VPC Id to be used> \
--context coordinator-image="<ECR image URI of the coordinator>" \
--context taskRole=<IAM role for the tasks> \
--context executionRole=<IAM role for the execution>
```

#### Deploy Cluster
```
cdk deploy "*" --context region=<aws-region> --context region=<aws-region> --context account=<aws-account> --context prefix=<prefix> \
--context worker-image="<ECR image URI of the worker>" \
--context security-group=<Security Group to be used> --context vpc=<VPC Id to be used> \
--context coordinator-image="<ECR image URI of the coordinator>" \
--context taskRole=<IAM role for the tasks> \
--context executionRole=<IAM role for the execution>
```

### Worker and Coordinator Images
Use the coordinator and worker folders present in remote-index-build-service to create the docker images and then use 
[this](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html) SOP to update those image on the ECR. 