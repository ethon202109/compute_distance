import numpy as np
from numpy import linalg as la
# 行向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[ 0,     5.196]
# [ 5.196, 0    ]]
def compute_squared_EDM_method(X):
    # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
    n, m = X.shape
    # 因为有n个向量，距离矩阵是n x n
    D = np.zeros([n, n])
    # 迭代求解向量的距离
    for i in range(n):
        for j in range(i + 1, n):
            # la.norm()求向量都范数，默认是2范数
            D[i, j] = la.norm(X[i, :] - X[j, :])
            D[j, i] = D[i, j]
    return D


print(compute_squared_EDM_method(np.array([[1, 2, 3], [4, 5, 6]])))


# 列向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[0,     1.414, 2.828]
# [1.414, 0,     1.414]
# [2.828, 1.414, 0    ]]
def compute_squared_EDM_method(X):
    # 获得矩阵都行和列，因为是列向量，因此一共有m个向量
    n, m = X.shape
    # 因为有m个向量，距离矩阵是m x m
    D = np.zeros([m, m])
    # 迭代求解向量的距离
    for i in range(m):
        for j in range(i + 1, m):
            # la.norm()求向量都范数，默认是2范数（注意这里是列向量)
            D[i, j] = la.norm(X[:, i] - X[:, j])
            D[j, i] = D[i, j]  # *1
    return D

###########################
# 上述运算可以使用点积（即矩阵内积）来计算：
# 这里是列向量
# D[i,j] = np.sqrt(np.dot(X[:,i]-X[:,j],(X[:,i]-X[:,j]).T))
############################
# 行向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[ 0,     5.196]
# [ 5.196, 0    ]]
def compute_squared_EDM_method2(X):
    # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
    n, m = X.shape
    # 因为有n个向量，距离矩阵是n x n
    D = np.zeros([n, n])
    # 迭代求解向量的距离
    for i in range(n):
        for j in range(i + 1, n):
            # 因为是行向量，这里是行索引
            d = X[i, :] - X[j, :]
            # 向量內积运算,并进行求根
            D[i, j] = np.sqrt(np.dot(d, d))
            D[j, i] = D[i, j]
    return D


# 列向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[0,     1.414, 2.828]
# [1.414, 0,     1.414]
# [2.828, 1.414, 0    ]]
def compute_squared_EDM_method2(X):
    # 获得矩阵都行和列，因为是列向量，因此一共有m个向量
    n, m = X.shape
    # 因为有m个向量，距离矩阵是m x m
    D = np.zeros([m, m])
    # 迭代求解向量的距离
    for i in range(m):
        for j in range(i + 1, m):
            # 因为是列向量，这里是列索引
            d = X[:, i] - X[:, j]
            # 向量內积运算，并求根运算
            D[i, j] = np.sqrt(np.dot(d, d))
            D[j, i] = D[i, j]
    return D


###########################
# 第三种方法：避免循环内的点积运算,只做加减运算
############################

# 行向量
# 行向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[ 0,     5.196]
# [ 5.196, 0    ]]
def compute_squared_EDM_method3(X):
    # 获得矩阵的行和列，因为是行向量，因此一共有n个向量
    n, m = X.shape
    # 计算Gram 矩阵
    G = np.dot(X, X.T)
    # 初始化距离矩阵，因为有n个向量，距离矩阵是n x n
    D = np.zeros([n, n])
    # 迭代求解
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.sqrt(G[i, i] - 2 * G[i, j] + G[j, j])
            D[j, i] = D[i, j]
    return D


# 列向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[0,     1.414, 2.828]
# [1.414, 0,     1.414]
# [2.828, 1.414, 0    ]]
def compute_squared_EDM_method3(X):
    # 获得矩阵都行和列，因为是列向量，因此一共有m个向量
    n, m = X.shape
    # 计算Gram 矩阵
    G = np.dot(X.T, X)
    # 初始化距离矩阵， # 因为有m个向量，距离矩阵是m x m
    D = np.zeros([m, m])
    # 迭代求解
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = np.sqrt(G[i, i] - 2 * G[i, j] + G[j, j])
            D[j, i] = D[i, j]
    return D

###########################
# 第四种方法：避免循环
############################

# 行向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[ 0,     5.196]
# [ 5.196, 0    ]]
def compute_squared_EDM_method4(X):
    # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
    n, m = X.shape
    # 计算Gram 矩阵
    G = np.dot(X, X.T)
    # 因为是行向量，n是向量个数,沿y轴复制n倍，x轴复制一倍
    H = np.tile(np.diag(G), (n, 1))
    return np.sqrt(H + H.T - 2 * G)


# 列向量
# [[1,2,3],
# [4,5,6]]
# 得到
# [[0,     1.414, 2.828]
# [1.414, 0,     1.414]
# [2.828, 1.414, 0    ]]
def compute_squared_EDM_method4(X):
    # 获得矩阵都行和列，因为是列向量，因此一共有m个向量
    n, m = X.shape
    # 计算Gram 矩阵
    G = np.dot(X.T, X)
    # 因为是列向量，n是向量个数,沿y轴复制m倍，x轴复制一倍
    H = np.tile(np.diag(G), (m, 1))
    return np.sqrt(H + H.T - 2 * G)


##################################
# 第五种方法：利用scipy求距离矩阵（推荐用法）
###################################

# 默认是针对行向量进行操作
# 向量矩阵为：
# [[1,2],
#  [3,4],
#  [5,6]
#  [7,8]]

# 距离矩阵为：
# [[0,     2.828, 5.656, 8.485],
#  [2.828, 0,     2.828, 5.656],
#  [5.656, 2.828, 0,     2.828],
#  [8.485, 5.656, 2.828, 0    ]]

# distA距离列表为（上三角矩阵展开成一个列表）:
# [2.828, 5.656, 8.485, 2.828, 5.656, 2.828]

# distB距离矩阵为:
# [[0,     2.828, 5.656, 8.485],
#  [2.828, 0,     2.828, 5.656],
#  [5.656, 2.828, 0,     2.828],
#  [8.485, 5.656, 2.828, 0    ]]
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

A = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
# A是一个向量矩阵：euclidean代表欧式距离
distA = pdist(A, metric='euclidean')
# 将distA数组变成一个矩阵
distB = squareform(distA)



#######################
#两矩阵求距离矩阵
#########################

# 行向量:A （3行2列）
# [[1,2],
# [3,4],
# [5,6]]

# 行向量:B （2行2列）
# [[1,2],
# [3,4]]

# 得到矩阵C（3行2列），由A->B的距离，Cij代表A中都第i个向量到B中第j向量都距离
# [[0,      2.828],
# [2.828 , 0    ],
# [5.656 , 2.828]]
def compute_distances_no_loops(A, B):
    # A 有m个向量
    m = np.shape(A)[0]
    # B 有n个向量
    n = np.shape(B)[0]
    # 求得矩阵M为 m*n维（针对行向量）
    M = np.dot(A, B.T)
    # 对于H,我们只需要A . A^T的对角线元素
    # np.square(A)是A中都每一个元素都求平方
    # np.square(A).sum(axis=1) 是将每一行都元素都求和，axis是按行求和（原因是行向量）
    # np.matrix() 是将一个列表转为矩阵，该矩阵为一行多列
    # 求矩阵都转置，为了变成一列多行
    # np.tile是复制，沿Y轴复制1倍（相当于没有复制），再沿X轴复制n倍
    H = np.tile(np.matrix(np.square(A).sum(axis=1)).T, (1, n))
    # 对于H,我们只需要B . B^T的对角线元素
    # np.square(B)是B中都每一个元素都求平方
    # np.square(B).sum(axis=1) 是将每一行都元素都求和，axis是按行求和（原因是行向量）
    # np.matrix() 是将一个列表转为矩阵，该矩阵为一行多列
    # np.tile是复制，沿Y轴复制m倍（相当于没有复制），再沿X轴复制1倍
    K = np.tile(np.matrix(np.square(B).sum(axis=1)), (m, 1))
    # H对M在y轴方向上传播,即H加和到M上的第一行,K对M在x轴方向上传播,即K加和到M上的每一列
    return np.sqrt(-2 * M + H + K)


# 行向量:A （2行3列）.3个向量
# [[1,2,3],
# [4,5,6]]

# 行向量:B （2行2列），2个向量
# [[1,2],
# [3,4]]

# 得到矩阵C（3行2列），由A->B都距离 Cij代表A中都第i个向量到B中第j向量都距离
# [[1    , 1    ],
# [2.236, 1    ],
# [3.605, 2.236]]
def compute_distances_no_loops(A, B):
    # A 有m个向量（针对列向量）
    m = np.shape(A)[1]
    # B 有n个向量（针对列向量）
    n = np.shape(B)[1]
    # 求得矩阵M为 m*n维
    # 求得矩阵M为 m*n维
    M = np.dot(A.T, B)
    # 对于H,我们只需要A . A^T的对角线元素,下面的方法高效求解(只计算对角线元素)
    # 沿Y轴复制1倍（相当于没有复制），再沿X轴复制n倍
    H = np.tile(np.matrix(np.square(A).sum(axis=0)).T, (1, n))
    # 结果K为n维行向量.要将其元素运用到矩阵M的每一列,需要将其转置为行向量
    K = np.tile(np.matrix(np.square(B).sum(axis=0)), (m, 1))
    # H对M在y轴方向上传播,即H加和到M上的第一行,K对M在x轴方向上传播,即K加和到M上的每一列
    return np.sqrt(-2 * M + H + K)


########################
#2利用`scipy`求距离矩阵（推荐用法）
##############################
# 行向量:A （3行2列）
# [[1,2],
# [3,4],
# [5,6]]

# 行向量:B （2行2列）
# [[1,2],
# [3,4]]

# 得到矩阵C（3行2列），由A->B的距离，Cij代表A中都第i个向量到B中第j向量都距离
# [[0,      2.828],
# [2.828 , 0    ],
# [5.656 , 2.828]]
import numpy as np
from scipy.spatial.distance import cdist

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
B = np.array([[1, 2],
              [3, 4]])
dist = cdist(A, B, metric='euclidean')
