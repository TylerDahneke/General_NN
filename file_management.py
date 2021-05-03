import os


def get_lines(file_inp):
    file = open(file_inp, 'r')
    lines = [i.strip() for i in file.readlines()]
    file.close()
    return lines

def write_lines(file_out, lines_list):
    file = open(file_out, 'w')
    for line in lines_list:
        file.write(line + '\n')
    file.close()