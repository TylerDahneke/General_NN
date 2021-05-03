import file_management
import neural_network
import random as r


correct = 0
incorrect = 0

def translate_ans(inp_str):
    first, inp_str = int(inp_str[:inp_str.find(' ')]), inp_str[inp_str.find(' ') + 1:]
    second, inp_str = int(inp_str[:inp_str.find(' ')]), inp_str[inp_str.find(' ') + 1:]
    if inp_str == 'R':
        return [first, second], [1, 0]
    if inp_str == 'G':
        return [first, second], [0, 1]


file_lines = file_management.get_lines('raw_tests.txt')
nn = neural_network.general_nn(dim=[2, 3, 2])
# nn = neural_network.general_nn([2, 4, 3], built=perfect, with_constant=False)
r.shuffle(file_lines)
for counter in range(100):
    for line in file_lines:
        inp, ans = translate_ans(line)
        nn.run_input(inp)
        nn.backwards_propagate(ans)
        nn.reset_nodes()
print(nn)
inp = input('Try this NN?')
r.shuffle(file_lines)
for line in file_lines:
    inp, ans = translate_ans(line)
    nn_guess = nn.run_input(inp, ret=True)
    if nn_guess == ans:
        correct += 1
    else:
        print(nn_guess, ans)
        incorrect += 1
print(f'{correct} / {incorrect + correct}')
