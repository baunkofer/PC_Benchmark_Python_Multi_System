import time
import numpy as np
import multiprocessing
import platform
import os
import csv
import sys

sys.dont_write_bytecode = True  # Disable frozen modules

try:
    import cupy as cp  # GPU Benchmark (falls CUDA-fähige GPU vorhanden ist)
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def cpu_worker(_):
    total = 0
    for _ in range(10**6):
        total += 1
    return total

def single_thread_cpu_benchmark():
    print("Starting Single Thread CPU Benchmark")
    start_time = time.time()
    total = cpu_worker(0)
    end_time = time.time()
    return end_time - start_time

def multi_thread_cpu_benchmark():
    print("Multithread Single Thread CPU Benchmark")
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(cpu_worker, range(num_cores))
    end_time = time.time()
    return end_time - start_time

def ram_benchmark():
    print("Starting RAM Benchmark")
    start_time = time.time()
    arr = np.random.random(size=(1000, 1000))
    mean = np.mean(arr)
    std = np.std(arr)
    end_time = time.time()
    return end_time - start_time

def ssd_benchmark():
    print("Starting SSD Benchmark")
    file_path = "benchmark_test_file.bin"
    data = os.urandom(1000 * 1024 * 1024)  # 100 MB zufällige Daten
    
    # Schreibtest
    print("Writing to SSD")
    start_time = time.time()
    with open(file_path, "wb") as f:
        f.write(data)
    write_time = time.time() - start_time
    
    # Lesetest
    print("Reading from SSD")
    start_time = time.time()
    with open(file_path, "rb") as f:
        _ = f.read()
    read_time = time.time() - start_time
    
    os.remove(file_path)
    return write_time, read_time

def gpu_benchmark():

    print("Starting GPU Benchmark")
    if not GPU_AVAILABLE:
        print("No GPU available.")
        return None
    
    start_time = time.time()
    arr = cp.random.random((500000, 500000))
    mean = cp.mean(arr)
    std = cp.std(arr)
    cp.cuda.Device(0).synchronize()  # Warten auf Fertigstellung der GPU-Berechnung
    end_time = time.time()
    return end_time - start_time

def run_benchmark(test_function, runs=3):
    results = [test_function() for _ in range(runs)]
    return min(results), max(results), sum(results) / runs

def main():
    computer_name = platform.node()
    system_info = f"{platform.system()} {platform.release()}"
    cpu_cores = multiprocessing.cpu_count()
    
    benchmarks = {
        "Single-Thread CPU": run_benchmark(single_thread_cpu_benchmark),
        "Multi-Thread CPU": run_benchmark(multi_thread_cpu_benchmark),
        "RAM": run_benchmark(ram_benchmark),
        "SSD Write": run_benchmark(lambda: ssd_benchmark()[0]),
        "SSD Read": run_benchmark(lambda: ssd_benchmark()[1])
    }
    
    if GPU_AVAILABLE:
        benchmarks["GPU"] = run_benchmark(gpu_benchmark)
    else:
        print("Keine GPU verfügbar.")
    
    csv_filename = f"benchmark_results_{computer_name}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Computer Name", "System", "CPU Cores", "Benchmark", "Best Time (sec)", "Worst Time (sec)", "Average Time (sec)"])
        
        for benchmark, (best, worst, avg) in benchmarks.items():
            writer.writerow([computer_name, system_info, cpu_cores, benchmark, best, worst, avg])
    
    print(f"Benchmark results saved to {csv_filename}")
    
if __name__ == "__main__":
    main()
