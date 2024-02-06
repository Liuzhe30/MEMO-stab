AA_dict = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

class fetchPDBSequence():
    def fetch(self, file_path, chain):
        with open(file_path) as r:
            sequence = ''
            pos = []
            line = r.readline().strip()
            while line:
                if(line[0:6].strip() != 'ATOM'):
                    line = r.readline()
                if(line[0:6].strip() == 'ATOM' and line.split()[2] == 'CA' and line.split()[4] == chain):
                    AA = line.split()[3]
                    sequence += AA_dict[AA]
                    pos.append(int(line.split()[5])) # may not be continuous
                line = r.readline()
            return sequence, pos

if __name__ == '__main__':

    f = fetchPDBSequence()
    sequence, pos = f.fetch('../datasets/raw/mCSM_membrane/pdb_stability/1QJP.pdb', 'A')
    print(sequence, pos)
    print(len(sequence))