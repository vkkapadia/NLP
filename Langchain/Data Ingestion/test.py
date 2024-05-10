import dataingestion
import faiss

# Create a standard L2 index on the CPU
index_cpu = faiss.IndexFlatL2(128)

# Create GPU resources
gpu_res = faiss.StandardGpuResources()

# Move the index to GPU, specifying the GPU device ID (0 for first GPU)
index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)

dataingestion.run_etl()