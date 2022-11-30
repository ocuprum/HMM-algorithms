import numpy as np
from evaluation.fba import FBA
from learning.bwa import BWA
from structure import get_structure, define_key
from text_analysis.text_getter import alphabet, clean_text, sym_to_num, num_to_sym
from decoding.stationary_dist import get_stationary_dist
from caesar_cipher import caesar_cipher, caesar_decipher
from generator.generator import generate_model
from decoding.viterbi import altViterbi

'''
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

fhandle.close()'''


# Task 2
fname = 'text_analysis/src/misto.txt'

# Отримання спостережень
if ' ' in alphabet: alphabet.remove(' ')
obs = 100000
ctext = clean_text(fname, alphabet)[:obs]
observation = sym_to_num(ctext)

# Знаходження матриці перехідних ймовірностей за допомогою біграм
N = len(alphabet)
transition = np.matrix(np.zeros((N, N)))
for sym in range(len(observation)-1):
    i = observation[sym]
    j = observation[sym+1]
    transition[i, j] += 1
transition += 5

# Знаходження стаціонарного розподілу 
norm = transition.sum(axis=1)
for i in range(N):
    transition[i] /= norm[i]
sd = np.array(get_stationary_dist(transition))[0]

# Функція визначення кількості правильно декодованих символів
def percentage(original, plaintext):
    count = 0
    l = len(original)
    for i in range(l):
        if plaintext[i] == original[i]:
            count += 1
    
    return '{}%'.format(count / l * 100)

# Шифрування частини тексту шифром Віженера 
start_obs = 2000
plaintext = sym_to_num(ctext[:start_obs])
key = 23
ciphertext = np.array(caesar_cipher(plaintext, key, alphabet))

d, _, output = generate_model(len(alphabet), len(alphabet))
bwa = BWA(d, transition, output, ciphertext)
bwa.learn(iters=50)

# Визначення ключа 
new_key = define_key(bwa.output, alphabet)

original = ctext[start_obs:start_obs+10000]
# ДЕКОДУВАННЯ - БВ
new_ciphertext = np.array(caesar_cipher(sym_to_num(original), key, alphabet))
bplaintext = ''
for sym in new_ciphertext:
    bplaintext += alphabet[new_key[sym]]

fhandle = open('decoding-result.txt', 'w')
fhandle.write('Original text:\n{}\n\n\n'.format(original[:500]))
fhandle.write('-------------BW-------------\n')
fhandle.write('Plaintext:\n{}\n\n\n'.format(bplaintext[:500]))            
perc = percentage(original, bplaintext)
fhandle.write('Співпадіння:\n{}\n\n'.format(perc))

# ДЕКОДУВАННЯ - ВІТЕРБІ
fhandle.write('-------------VITERBI-------------\n')
v = altViterbi(d, transition, bwa.output, new_ciphertext)
vplaintext = ''.join(num_to_sym(v.alt_decode()))
fhandle.write('Plaintext:\n{}\n\n\n'.format(vplaintext[:500]))
perc = percentage(original, vplaintext)
fhandle.write('Співпадіння:\n{}\n\n'.format(perc))




'''d = np.array([0.05, 0.2, 0.75])
A = np.array([[0.2, 0.3, 0.5], 
                [0.2, 0.2, 0.6],
                [0, 0.2, 0.8]])
            
B = np.array([[0.7, 0.2, 0.1], 
                [0.3, 0.4, 0.3],
                [0, 0.1, 0.9]])

y = np.array([0, 2, 1, 0, 2])

v = altViterbi(d, A, B, y)
print(v.alt_decode())
print(v.psi)'''
