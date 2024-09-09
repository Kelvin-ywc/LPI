import numpy as np
import torch
def cal_dif( v1, v2):
    return np.sum((v1 - v2) ** 2)

v1 = [1,2,3]
v2 = [4,5,6]
v1 = torch.tensor(v1)
v2 = torch.tensor(v2)
# print(cal_dif(v1, v2))
print(torch.sum((v1-v2)**2))