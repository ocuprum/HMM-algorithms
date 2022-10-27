from generator.generator import generate_model
from text_analysis.text_getter import sym_to_num, clean_text
from learning.bwa import BWA

def define_group(mtrx, alphabet):
    groups = {}
    max_probs = mtrx.T.argmax(1)
    for sym in range(len(alphabet)):
        group = max_probs[sym, 0]
        groups[group] = groups.get(group, []) + [alphabet[sym]]
    
    return groups

def get_structure(fname, N, alphabet, obs=50000):
    ctext = clean_text(fname, alphabet)[:obs]

    distribution, transition, output = generate_model(N, len(alphabet))
    observation = sym_to_num(ctext, alphabet)

    model = BWA(distribution, transition, output, observation)
    model.learn()

    groups = define_group(model.output, alphabet)

    result = ''
    for group, symbols in groups.items():
        result += 'Group {}: {}\n'.format(group, symbols)
    return result