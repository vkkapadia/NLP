import faiss

# Specify the dimensionality of your vectors
d = 128

# Create a standard L2 index on the CPU
index_cpu = faiss.IndexFlatL2(d)

# Create GPU resources
gpu_res = faiss.StandardGpuResources()

# Move the index to GPU, specifying the GPU device ID (0 for first GPU)
index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)

# Add some dummy vectors to the index
import numpy as np
vectors = np.random.random((1000, d)).astype('float32')
index_gpu.add(vectors)

# Perform a search on the GPU index
k = 5  # Number of nearest neighbors
distances, indices = index_gpu.search(vectors, k)

print("Search completed successfully.")
print("Sample distances:", distances[:5])
print("Sample indices:", indices[:5])
