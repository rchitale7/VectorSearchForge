## Creating the conda env
conda create --name cuvs-test --file environment.yml

## Running the file

python main.py


## Image build
docker build -t open-ai-bug:latest .

## Run the code
