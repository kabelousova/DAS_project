import numpy as np
import matplotlib.pyplot as plt

# array = np.array([1,2,3])
# print(array[array >= 2])
# print(np.mean(array[array >= 2]))
#
def cross_cor(segm1, segm2, len_window=10):
    n = np.shape(segm1)[1]
    max_sum = 0
    arr_sum = []
    arg_max = -1
    for i in range(1, n):
        # print(i, n - i)
        # print(segm1[0, 0:i], segm2.T[n - i: n, 0], sep="\n")
        # print(np.shape(segm1))
        # print(np.shape(segm2.T))
        mysum = float((segm1[0, 0:i] * segm2.T[n - i: n, 0]))
        if mysum > max_sum:
            max_sum = mysum
            arg_max = i
        # print("sum =", mysum)
        arr_sum.append(mysum)
    # print(8*"=")
    for i in range(0, n):
        # print(i, n - i)
        # print(segm1[0, i:n], segm2.T[i: n, 0], sep="\n")
        mysum = float((segm1[0, i:n] * segm2.T[i: n, 0]))
        if mysum > max_sum:
            max_sum = mysum
            arg_max = i
        # print("sum =", mysum)
        arr_sum.append(mysum)

    return np.argmax(arr_sum), max_sum, arr_sum

rng = np.random.default_rng()
# array = np.matrix(rng.integers(low=-10, high=10, size=25))
# array1 = array
# array2 = array
array1 = np.matrix(rng.integers(low=-10, high=10, size=25))
array2 = np.matrix(rng.integers(low=-10, high=10, size=25))
# array1 = np.matrix(np.array([1,0,1,0,1,1]))
# array2 = np.matrix([0,1,0,1,0,1])
# print(array1)
# print(array2)
# print(np.shape(array1))
# print(np.shape(array2))
arg, max, arr = cross_cor(array1, array2)
print(arg)
plt.plot(arr)
plt.show()
