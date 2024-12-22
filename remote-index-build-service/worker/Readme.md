# Remote Index Build Service(Dataplane/Workers)
## Overview
The Data Plane component is a crucial part of the Remote Index Build Service that handles the actual processing and building of vector indices. It's responsible for the execution layer of index building operations, working in conjunction with the control plane to manage vector search capabilities.

## Deployment

### Build the Docker image via docker compose
```bash
docker compose build
```
### Run the container via docker compose
```bash
docker compose up
```
This will spin up a container listening on the port: `6005` 

### Run the container via docker compose in detach mode
```bash
docker compose up -d
```
### Check the logs of the docker compose
```bash
docker compose logs -f
```
### Bring docker compose down
```bash
docker compose down
```
### See running container
```bash
docker compose ps
```
