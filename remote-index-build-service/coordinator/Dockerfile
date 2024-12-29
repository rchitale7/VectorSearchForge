FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create user and set permissions
RUN useradd -m appuser 
RUN chown -R appuser:appuser /app

RUN mkdir -p /app/logs && \
    chmod -R 777 /app/logs