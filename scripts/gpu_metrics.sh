#!/bin/bash

# Function to capture GPU metrics and save them to CSV
capture_gpu_metrics() {
    # Get current date and time
    timestamp=$(date +"%Y-%m-%d %T")

    # Get GPU utilization
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # Get GPU memory utilization
    gpu_mem_util=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits)

    # Get GPU temperature
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)

    # Get GPU power consumption
    gpu_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)

    # Print captured metrics to Console
    echo "${timestamp},${gpu_util}%,${gpu_mem_util}%,${gpu_temp}°C,${gpu_power}W"
    # Print captured metrics to CSV file
    echo "${timestamp},${gpu_util}%,${gpu_mem_util}%,${gpu_temp}°C,${gpu_power}W" >> scripts/gpu_metrics.csv

}

echo "Timestamp,GPU Utilization,GPU Memory Utilization,GPU Temperature,GPU Power Consumption"

# Create CSV file and add header if it doesn't exist
if [ ! -f "gpu_metrics.csv" ]; then
    echo "Timestamp,GPU Utilization,GPU Memory Utilization,GPU Temperature,GPU Power Consumption" > scripts/gpu_metrics.csv
fi

# Continuously capture GPU metrics and append to CSV file
while true; do
    capture_gpu_metrics
    sleep 1  # Adjust the interval as needed (in seconds)
done
