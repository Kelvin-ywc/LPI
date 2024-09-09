import numpy as np

mat = np.loadtxt('../MID/task_sim_matrix.txt')
threshold = 0.4
mat = (mat>threshold).astype(int)
print(mat)
