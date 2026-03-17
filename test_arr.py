import numpy as np

arr = np.array([1, 2])
try:
    print(np.int64(arr))
except Exception as e:
    print("np.int64", type(e))

