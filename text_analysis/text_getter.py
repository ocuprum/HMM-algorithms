from typing import List

#Можна додати або видалити символ(и) за потребою
alphabet = ['а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є','ж', 'з', 'и', 'і',
            'ї', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 
            'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я', ' ']

def text_getter(text:str, alphabet:List = alphabet) -> str:
    text = text.lower()

    dellist = set(symbol for symbol in text if symbol not in alphabet)
    for symbol in dellist:
        text = text.replace(symbol, '')

    return text

def clean_text(src, alphabet=alphabet):
    try:
        fhandle = open(src)
    except:
        print('File cannot be opened: {}'.format(src))
        exit()

    raw_text = fhandle.read()
    fhandle.close()

    text = text_getter(raw_text, alphabet)
    return text

def sym_to_num(text, alphabet=alphabet):
    obs_num = []
    for sym in text:
        obs_num.append(alphabet.index(sym))
    
    return obs_num