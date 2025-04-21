from py3nvml import nvidia_smi
import psutil
import time
import threading
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import logging


class GPUMemoryMonitor:
    def __init__(self, gpu_id=0, interval: float = 0.1):
        self.interval = interval
        self.gpu_id = gpu_id
        self.memory_logs = []
        self.start_time = None
        self.ram_used_mb: List[float] = []
        self.timestamps: List[float] = []
        self.is_monitoring = False
        self._monitor_thread = None
        self.process = psutil.Process()
        
        # Initialize NVML
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)

    def _get_memory_info(self):
        """Get current GPU memory usage in MB"""
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            'used': info.used / 1024 / 1024,  # Convert to MB
            'free': info.free / 1024 / 1024,
            'total': info.total / 1024 / 1024
        }

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            self.timestamps.append(time.time())
            memory_info = self._get_memory_info()
            self.memory_logs.append({
                'used_memory': memory_info['used'],
                'free_memory': memory_info['free'],
                'total_memory': memory_info['total']
            })
            self.ram_used_mb.append(self._get_process_memory_mb())
            time.sleep(self.interval)  # 100ms sampling rate

    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in KB"""
        return self.process.memory_info().rss / (1024**2)  # Convert bytes to MB

    def start_monitoring(self):
        """Start GPU memory monitoring"""
        self.start_time = time.time()
        self.is_monitoring = True
        self.memory_logs = []
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop GPU memory monitoring"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def log_metrics(self):
        df = pd.DataFrame(self.memory_logs)
        max_memory = df['used_memory'].max()
        min_memory = df['used_memory'].min()
        logging.info(f"Max GPU Memory: ,{max_memory}")
        logging.info(f"Min GPU Memory: ,{min_memory}")
        logging.info(f"Net GPU Memory used:, {max_memory-min_memory}")

    def __del__(self):
        """Cleanup NVML on object destruction"""
        try:
            nvidia_smi.nvmlShutdown()
        except:
            pass