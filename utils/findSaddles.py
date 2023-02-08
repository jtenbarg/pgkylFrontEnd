import numpy as np

def saddles(mat : np.ndarray) -> list:

    """
    returns the list of all saddle points of the input matrix

    """

    (N, M) = mat.shape

    jMax = np.argmax(mat, axis = 1) # index of col for max in each row
    iMin = np.argmin(mat, axis = 0) # index of row for min in each col

    IJMax = [(i,jMax[i]) for i in range(N)] # list of indexes of max of each row
    IJMin = [(iMin[j],j) for j in range(M)] # list of indexes of min of each col

    maxRowMinCol = list(set(IJMax) & set(IJMin)) # max of row, min of col


    iMax = np.argmax(mat, axis = 0) # index of row for max in each col
    jMin = np.argmin(mat, axis = 1) # index of col for min in each row

    IJMax = [(iMax[j],j) for j in range(M)] # list of indexes of max of each col
    IJMin = [(i,jMin[i]) for i in range(N)] # list of indexes of min of each row

    minRowMaxCol = list(set(IJMax) & set(IJMin)) # min of row, max of col


    return maxRowMinCol + minRowMaxCol
