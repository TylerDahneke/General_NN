from custom_array import *

# BEGIN DESC.

#  CLASSES:
#     general_nn ->  PARAMETERS
#                      dim           : list[int(), int()] -> 2 by 2 input, required if "built" is not filled
#                      built         : bool() -> takes "TestCase" and creates weight list, required if lacking dim
#                      with_constant : bool() -> if true, adds constant row to all weight matricies. doesn't matter if
#
#                 -> OBJECT CALLS
#                      .run_input(inp, ret=F)    : -> if ret, RETURNS .translate_answer() given inp that matches
#                                                  dimension fills self.nodes to save node values
#                      .backwards_propagate(ans) : -> takes answer and creates cost function. takes partial derivative
#                                                     of each weight and applies gradient descent.
#                      .bw_prop_helper(d, cnt)   : -> RETURNS list[list[]] -> given other matrix, e, returns "built" for
#                                                      product matrix
#                      .translate_ans(ans)       : -> RETURNS list[int(), int()] -> gets ans with format of 1 in max pos
#                      .reset_nodes()            : -> sets "self.nodes" = []
#
#  DEFINITIONS:
#     build_weight_matricies(m_list, w_const), m_list: list[list[]], w_const: bool();
#                                        -> RETURNS list[obj(Matrix)] called "self.weights"
#
#     fill_nn(dim, w_const), dim: list[int()], w_const: bool();
#                                        -> RETURNS list[obj(Matrix)] called "self.weights". all points random.
#
#
#
# END DESC.
# GLOBALS
# TestCase is output of dim=[2,3], learningrate=5, iterations=50,000
TestCase = [[[-158.8673270829059, 15.075132406033047, 1300.4086274105994],
             [72.34605511413292, 58.06848791011592, -646.0480049141382],
             [73.25081667590938, -70.64878842253808, -648.6392405309638]]

            ]
LearningRate = 5


# END GLOBALS
#


def build_weight_matricies(matrix_list, with_constant=False):
    outer = []
    for matr in matrix_list:
        outer.append(matrix(built=matr, with_constant=with_constant))
    return outer


def fill_nn(dim, with_constant=False):
    counter = 0
    outer = []

    while counter < len(dim) - 1:
        ph = matrix(dim=[dim[counter], dim[counter + 1]], with_constant=with_constant)

        outer.append(ph)
        counter += 1
    return outer


class general_nn:

    def __init__(self, dim=None, built=False, with_constant=True):
        if built:
            self.weights = build_weight_matricies(TestCase, with_constant=with_constant)
        else:
            self.weights = fill_nn(dim, with_constant=with_constant)
        self.nodes = []

    def __repr__(self):
        return str(self.weights)

    def run_input(self, inp, ret=False):
        inp = matrix(built=[inp], prep_inp=True)
        self.nodes.append(inp)
        for weight_matr in self.weights:
            inp = matrix(built=inp.multiply(weight_matr), prep_inp=True)
            self.nodes.append(inp)
        inp.drop_prep_inp()
        if ret:
            self.reset_nodes()
            return self.translate_ans(inp.contents[0])

    def backwards_propagate(self, ans):
        delta_j_list = []
        if self.nodes:
            for delta_pos in range(len(ans)):
                node_val = self.nodes[-1].contents[0][delta_pos]
                delta_j = (node_val - ans[delta_pos]) * deriv_activation_function(node_val)
                delta_j_list.append(delta_j)
            return self.backwards_prop_helper(delta_j_list, len(self.weights))

    def backwards_prop_helper(self, delta_list, counter):
        if counter:
            curr_weights = self.weights[counter - 1].contents
            curr_nodes = self.nodes[counter - 1].contents[0]
            dimen = len(curr_weights), len(curr_weights[0])
            if dimen == (4, 5):
                x = 1
            for curr_node_pos in range(len(curr_nodes)):
                for next_node_pos in range(len(delta_list)):
                    curr_weights[curr_node_pos][next_node_pos] += - LearningRate * curr_nodes[curr_node_pos] * \
                                                                  delta_list[next_node_pos]

            delta_j_list = []
            for delta_pos in range(len(delta_list) - 1):
                node_val = curr_nodes[delta_pos]
                delta_j = (node_val - delta_list[delta_pos]) * deriv_activation_function(node_val)
                delta_j_list.append(delta_j)
            self.backwards_prop_helper(delta_j_list, counter - 1)

    def translate_ans(self, ans):
        max_pos = 0
        curr_pos = 0
        for val in ans:
            if val > ans[max_pos]:
                max_pos = curr_pos
            curr_pos += 1
        r_list = [0] * len(ans)
        r_list[max_pos] = 1
        return r_list

    def reset_nodes(self):
        self.nodes = []


if __name__ == '__main__':
    o5 = 0.9214430516601156
    o6 = 0.9927841574582754
    first_lam = (o5 - 1) * o5 * (1 - o5)
    second_lam = (o6 - 0) * o6 * (1 - o6)
    nn = general_nn(dim=[2], built=True)
    nn.backwards_propagate([1, 0])
