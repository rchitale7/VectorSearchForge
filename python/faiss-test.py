import faiss
from faiss import IndexHNSWCagra
import numpy as np

faissIndex = faiss.read_index("/Volumes/workplace/VectorSearchForge/cagraindex-test-with-ids.txt")

xq = np.array([[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]]).astype('float32')

k = 10
D, I = faissIndex.search(xq, k)

# D and I are 2D arrays
print(D[0][0]) # returns top element distance
print(I[0][0]) # returns top element Id

