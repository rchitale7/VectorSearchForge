# Build the Docker image
docker build -t remote-index-build-service .

# Run the container
docker run -p 5000:5000 -v $(pwd)/logs:/app/logs remote-index-build-service 
