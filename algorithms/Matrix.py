import math
import numpy


class Matrix:
    def __init__(self, rows, columns, name):
        self.name = name
        self.rows = rows
        self.columns = columns
        self.matrix = [[0 for x in range(columns)] for y in range(rows)]

    def __repr__(self):
        string = ''
        name_length = len("{} = ".format(self.name))
        for x in range(len(self.matrix)):
            if x == math.ceil(self.rows / 2) - 1:
                string += "{} = ".format(self.name)
            else:
                string += " " * name_length
            string += self.row_string(self.matrix[x])
            string += '\n'
        return string

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.dot_product(other)
        elif isinstance(other, (int, float)):
            return self.multiply_by_value(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def row_string(self, rows):
        row = '|'
        for x in rows:
            row += str(x)
            row += ' '
        row = row[:-1]
        row += '|'
        return row

    def multiply_by_value(self, value):
        for x in range(self.rows):
            for y in range(self.columns):
                self.matrix[x][y] = self.matrix[x][y] * value

    def dot_product(self, m):
        result = Matrix(self.rows, m.columns, '{} * {}'.format(self.name, m.name))
        result.matrix = numpy.dot(numpy.array(self.matrix), numpy.array(m.matrix)).tolist()
        return result

    def insert_at(self, row, column, value):
        if not (row <= self.rows and column < self.columns): return None
        self.matrix[row][column] = value
        return value


x = Matrix(2, 3, 'X')
x.insert_at(0, 0, 1)
x.insert_at(0, 1, 2)
x.insert_at(0, 2, 3)
x.insert_at(1, 0, 4)
x.insert_at(1, 1, 5)
x.insert_at(1, 2, 6)

y = Matrix(3, 2, 'Y')
y.insert_at(0, 0, 10)
y.insert_at(0, 1, 11)
y.insert_at(1, 0, 20)
y.insert_at(1, 1, 21)
y.insert_at(2, 0, 30)
y.insert_at(2, 1, 31)

print(x * y)
