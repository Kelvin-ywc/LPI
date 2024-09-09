import numpy as np

# task_vector = np.loadtxt('../MID/tasks_array.txt')
# print(task_vector)

# 12个1024维的向量
vectors = np.loadtxt('../MID/tasks_array.txt')  # 填入实际的向量值

# 初始化一个12x12的矩阵，用于存储相似度
cosine_similarity_matrix = np.zeros((12, 12))

# 计算余弦相似度
for i in range(12):
    for j in range(12):
        # embedding1 = vectors[i]
        # embedding2 = vectors[j]
        # cosine_similarity_matrix[i, j] = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        cosine_similarity_matrix[i, j] = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))

# 输出余弦相似度矩阵
print(cosine_similarity_matrix)

np.savetxt('../MID/task_sim_matrix.txt', cosine_similarity_matrix)