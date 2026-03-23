import matrix

m1 = matrix.Matrix(3, 3, 0.0)
m2 = matrix.Matrix.identity(3)

m3 = m1 + m2
print(m3)

m2.randomize(-1.0, 1.0)
print(m2.getRow(0))

col = m2.getCol(1)
m2.setRow(0, [1.0, 2.0, 3.0])

vec = matrix.Matrix.fromVector([1.0, 2.0, 3.0], True)
print(vec.transpose())