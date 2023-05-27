import os
import fixed_parameters as FP
from rwHelper import *

def buildVocabForAlphabet(alphabet, vocab_path):
    with open(vocab_path, 'w') as f:
        f.write(''.join(alphabet))

def main():
    data_path = {'AMP':'data/AMP_seq.txt','uniprot':'data/uniprot.txt', 're-uniprot':'data/re-uniprot.txt'}
    vocab_path = {'AMP':'./vocab/AMP_seq.vocab','uniprot':'./vocab/uniprot.vocab','re-uniprot':'./vocab/re-uniprot.vocab'}

    for page in ['re-uniprot']:
        data_list = lineTxtHelper(data_path[page]).readLines()
        text = ''.join(data_list)
        alphabet = sorted(list(set(list(text))),key=str)
        alphabet.extend([v for k,v in FP.SPACIAIL_TAG.items()])
        alphabet = sorted(alphabet)
        print(f'alphabet len({len(alphabet)})')
        buildVocabForAlphabet(alphabet, vocab_path[page])
    


if __name__ == "__main__":
    main()