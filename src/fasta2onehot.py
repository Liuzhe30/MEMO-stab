# convert fasta sequences into onehot code
import numpy as np

eyes = np.eye(20)
protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
    'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
    'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
    'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}

class fasta2onehot():
    def convert(self, fastastring, maxlen):
        onehot_matrix = np.zeros([maxlen, 20], float)
        for i in range(len(fastastring)):
            onehot = protein_dict[fastastring[i]]
            onehot_matrix[i] = onehot
        return onehot_matrix
