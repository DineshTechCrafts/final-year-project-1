import numpy as np
gt = np.load(r"LiverCancer-MultiAgent-Retrieval\data\embeddings\ground_truth\slice_embeddings.npy")
print("Shape:", gt.shape)
print("Any all-zero rows (should be some, for tumor-absent slices):", (gt[:, 2, :].sum(axis=1) == 0).sum())
print("Non-zero embeddings look random or structured?", gt[0, 0, :10])  # first 10 values of first slice's full-image embedding