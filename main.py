from structure import get_structure
from text_analysis.text_getter import alphabet

# Task 1
fhandle = open('structure-result.txt', 'w')
folder = 'text_analysis/src/'
srcs = ['eneida', 'misto']

for src in srcs:
    fname = folder + src + '.txt'
    fhandle.write('---{}---\n'.format(src.upper()))
    fhandle.write("\nWithout apostrophe:\n")
    for N in range(2, 5):
        fhandle.write('[N = {}]\n'.format(N))
        fhandle.write(get_structure(fname, N, alphabet))

    fhandle.write("\nWith apostrophe:\n")
    for N in range(2, 5):
        fhandle.write('[N = {}]\n'.format(N))
        fhandle.write(get_structure(fname, N, alphabet + ["'"]))

