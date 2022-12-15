import numpy as np

dt = np.dtype(np.int32)
dt = np.dtype([('age',np.int8)])
print(dt)

student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student)
print(student)
print(student['name'])

a = np.arange(10)
print(a)
print("")
b= np.arange(10, 20)
print(b)
print("")
c = np.hstack((a, b)) # hstack表示横向连接
print(c)
print("")
d = np.vstack((a, b)) # vstack表示纵向连接
print(d)
print("")
e = np.stack((a, b), axis=0) # axis=0 横向看(纵向连接)
print(e)
print("")
e = np.stack((a, b), axis=1) # 纵向看(横向连接)
print(e)

