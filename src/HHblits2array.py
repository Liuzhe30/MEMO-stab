# convert hhblits file into numpy array
import numpy as np

class HHblits2array():
    def convet(self, filename, fasta_len):
        with open(filename) as hhm_file:     
            hhm_matrix = np.zeros([fasta_len, 30], float)
            hhm_line = hhm_file.readline()
            idxx = 0
            while(hhm_line[0] != '#'):
                hhm_line = hhm_file.readline()
            for i in range(0,5):
                hhm_line = hhm_file.readline()
            while hhm_line:
                if(len(hhm_line.split()) == 23):
                    idxx += 1
                    if(idxx == fasta_len + 1):
                        break
                    each_item = hhm_line.split()[2:22]
                    for idx, s in enumerate(each_item):
                        if(s == '*'):
                            each_item[idx] = '99999'                            
                    for j in range(0, 20):
                        hhm_matrix[idxx - 1, j] = int(each_item[j])
                        #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))                                              
                elif(len(hhm_line.split()) == 10):
                    each_item = hhm_line.split()[0:10]
                    for idx, s in enumerate(each_item):
                        if(s == '*'):
                            each_item[idx] = '99999'                             
                    for j in range(20, 30):
                        hhm_matrix[idxx - 1, j] = int(each_item[j - 20]) 
                        #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))                                                                        
                hhm_line = hhm_file.readline()
            return hhm_matrix