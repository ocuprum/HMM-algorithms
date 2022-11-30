#Підрахунок кількості потрібних частинок у тексті
def symbol_frequency(text, sort: bool=False):
    frequency = {}
    for sym in text:
        frequency[sym] = frequency.get(sym, 0) + 1

    if sort:
        return sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    return frequency

def bigram_frequency_with_intersection(text):
    bigrams = [text[i] + text[i+1] for i in range(len(text)-1)]
    return symbol_frequency(bigrams, sort=True)