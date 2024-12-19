# # Use Python 3.10 slim image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#     && rm -rf /var/lib/apt/lists/*

# # Create a non-root user
# RUN adduser --disabled-password --gecos '' appuser

# # Copy requirements first
# COPY requirements.txt .

# # Create logs directory
# RUN mkdir -p /app/logs

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application
# COPY . .

# # Change ownership to non-root user
# RUN chown -R appuser:appuser /app

# # Switch to non-root user
# #USER appuser

# # Expose port
# #EXPOSE 5000

# # Command to run the application
# #CMD ["python", "app.py"]

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