import file_management
import neural_network
import random as r

# GLOBALS


ITERATIONS = 1000


# ENDGLOBALS

def translate_ans(inp_str):
    poss_ans = ['R', 'G', 'B']
    first, inp_str = int(inp_str[:inp_str.find(' ')]), inp_str[inp_str.find(' ') + 1:]
    second, inp_str = int(inp_str[:inp_str.find(' ')]), inp_str[inp_str.find(' ') + 1:]
    r_list = [0] * len(poss_ans)
    r_list[poss_ans.index(inp_str)] = 1
    return [first, second], r_list


def main(gen_nn, test_file):
    print(test_against_file_name(test_file, gen_nn))


def test_against_file_name(file_name, nn):
    correct_guesses = 0
    incorrect_guesses = 0
    file_lines = file_management.get_lines(file_name)
    r.shuffle(file_lines)
    for line in file_lines:
        inp, ans = translate_ans(line)
        nn_guess = nn.run_input(inp, ret=True)
        if nn_guess == ans:
            correct_guesses += 1
        else:
            incorrect_guesses += 1
    return correct_guesses, incorrect_guesses + correct_guesses


def train_with_file_name(file_name, nn):
    file_lines = file_management.get_lines(file_name)
    r.shuffle(file_lines)
    for counter in range(ITERATIONS):
        for line in file_lines:
            inp, ans = translate_ans(line)
            nn.run_input(inp)
            nn.backwards_propagate(ans)
            nn.reset_nodes()
        if not (counter % (ITERATIONS // 10)):
            print(counter)
    return nn


if __name__ == '__main__':
    training_file = 'raw_tests.txt'
    test_file = 'raw_tests.txt'
    # gen_nn = neural_network.general_nn(built=True)
    gen_nn = neural_network.general_nn(dim=[2, 3])
    # train_with_file_name(training_file, gen_nn)
    main(gen_nn, test_file)

#
