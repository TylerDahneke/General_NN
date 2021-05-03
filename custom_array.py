import random as r
import math as m


# BEGIN DESC.

#
#  classes:
#      matrix -> ( PARAMETERS    :
#                  built         : list[list[]] -> storage for matrix
#                  dim           : list[int(), int()] -> 2 by 2 input, required if "built" is not filled
#                  with_constant : bool() -> if true, calls self.with_constant()
#                  prep_inp      : bool() -> if true, calls self.prep_inp()
#                 )
#             -> ( OBJECT CALLS  :
#                  .get_dim(_)   : RETURNS list[int(), int()] -> gets dimension of current matrix
#                  .multiply(e)  : RETURNS list[list[]] -> given other matrix, e, returns "built" for product matrix
#                  .trans()      : RETURNS list[list[]] -> gets transposed "built" of "contents"
#                  .prep_inp()   : RETURNS None -> applies activation function to "contents". modifies first row of
#                                                 "contents" to have int(1) at end.
#                  .prep_weight(): RETURNS None -> adds one row of repeating float to "contents" of len x of "contents"
#
#   definitions:
#       create_matrix(dim), dim: list[int(), int()] -> RETURNS "built" of random.random() of dimension dim
#
#       activation_function(x), x: int() -> RETURNS logarithmic_activation_function(x)
#
#       deriv_activation_function(x), x: int() -> given x is output of activation_function(y), RETURNS derivative in
#                                                 terms of in y
#
#
#
# END DESC.
# GLOBALS

# END GLOBALS
#


def create_matrix(dim):
    col, row = dim
    outer = []
    inner = []
    for _ in range(col):
        for _ in range(row):
            inner.append(r.random())
        outer.append(inner)
        inner = []
    return outer


def activation_function(x):
    return 1 / (1 + m.pow(m.e, -x))


def deriv_activation_function(x):
    return x * (1 - x)


class matrix:

    def __init__(self, built=None, dim=None, with_constant=False, prep_inp=False):
        if built is None:
            self.contents = create_matrix(dim)
        else:
            self.contents = built
        if prep_inp:
            self.prep_inp()
        if with_constant:
            self.prep_weight()

    def __repr__(self):
        r_str = ''
        for row in self.contents:
            r_str += f'{row}\n'
        return r_str

    def get_dim(self):
        return [len(self.contents), len(self.contents[0])]

    def multiply(self, other):
        outer = []
        for row in self.contents:
            inner = []
            for col in other.trans():
                sum = 0
                for pos in range(len(row)):
                    sum += row[pos] * col[pos]
                inner.append(sum)
            outer.append(inner)
        return outer

    def trans(self):
        outer = []
        inner = []
        for x_pos in range(len(self.contents[0])):
            for y_pos in range(len(self.contents)):
                inner.append(self.contents[y_pos][x_pos])
            outer.append(inner)
            inner = []
        return outer

    def prep_inp(self):
        self.contents[0] = [activation_function(x) for x in self.contents[0]]
        self.contents[0].append(1)

    def drop_prep_inp(self):
        self.contents[0].pop()

    def prep_weight(self):
        col, row = self.get_dim()
        self.contents.append([r.random()] * row)


if __name__ == '__main__':
    ph = matrix(built=[[1, 2, 3], [3, 4, 5]])
    s_ph = matrix(built=[[1], [1]])
    out = ph.multiply(s_ph)
