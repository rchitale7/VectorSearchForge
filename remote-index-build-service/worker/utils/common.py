import math
import os

def get_omp_num_threads():
    return max(math.floor(os.cpu_count()-2), 1)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)