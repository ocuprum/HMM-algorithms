from text_analysis.text_getter import sym_to_num

def caesar_cipher(plaintext, key, alphabet):
    ciphertext = []
    sym_amount = len(alphabet)
    for sym in plaintext:
        ciphertext.append((sym + key) % sym_amount)
    return ciphertext

def caesar_decipher(ciphertext, key, alphabet):
    plaintext = caesar_cipher(ciphertext, len(alphabet) - key, alphabet)
    return plaintext

