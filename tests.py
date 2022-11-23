from Grid import Grid
import numpy as np

arr = np.array([[0,1,0],[0,0,0]])
arr2 = arr[4:]
# print(arr2)
# print(np.nonzero(arr)[0])
# if np.nonzero(arr2)[0].size != 0:
#     print(np.nonzero(arr2)[0].size)
#     print(np.nonzero(arr2)[0][0])


rows = np.array([0,0,0,0])

print(rows[rows>10].size)
