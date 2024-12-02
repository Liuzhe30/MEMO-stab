with open('celegans_protein.csv', 'r') as f:
    lines = f.readlines()
    with open('celegans_protein.fasta', 'w') as w:
        for line in lines:
            id, seq = line.strip().split(',')
            w.write(f'>{id}\n')
            w.write(f'{seq}\n')