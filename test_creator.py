import random as r
import file_management as fm

lines = []
for _ in range(300):
    x, y = r.randint(0, 2), r.randint(8, 10)
    lines.append(f'{x} {y} R')

for _ in range(300):
    x, y = r.randint(3, 5), r.randint(11, 13)
    lines.append(f'{x} {y} G')
for _ in range(300):
    x, y = r.randint(6, 8), r.randint(14, 16)
    lines.append(f'{x} {y} B')





fm.write_lines('raw_tests.txt', lines)
