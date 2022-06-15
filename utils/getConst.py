import numpy as np
import csv

def setConst(self):

    n = []
    paramFile = self.paramFile
    self.filenameBase = paramFile[:paramFile.find('params')]
    
    with open(paramFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 1:
                eps0 = float(row[0].strip())
                mu0 = float(row[1].strip())
                B0 = float(row[2].strip())
                self.dimsX = int(row[3].strip())
                self.dimsV = int(row[4].strip())
                self.po = int(row[5].strip())
                self.basis = row[6].strip()
                self.model = row[7].strip()
            elif line_count > 2:
                self.speciesFileIndex.append(row[0].strip())
                self.mu.append(float(row[1].strip()))
                self.q.append(float(row[2].strip()))
                self.temp.append(float(row[3].strip()))
                n.append(float(row[4].strip()))
            line_count += 1

    self.mu0 = mu0
    self.eps0 = eps0
    self.c = 1 / np.sqrt(mu0*eps0)
    self.vA =  B0 / np.sqrt(np.multiply(n, self.mu) * mu0)
    self.vt =  np.sqrt(2.*np.divide(self.temp, self.mu))
    self.omegaP = np.sqrt(np.divide( np.multiply( np.multiply(self.q,self.q), n), self.mu)/eps0)
    self.omegaC = np.divide(np.absolute(self.q)*B0, self.mu)
    self.d = np.divide(self.c, self.omegaP)
    self.debye = np.divide(self.vt, self.omegaP) / np.sqrt(2.)

    if B0 != 0.:
        self.beta = 2.*mu0*np.multiply(n,self.temp) / (B0*B0 )
        self.rho = np.divide(self.vt, self.omegaC)
    else:
        self.beta = np.zeros(len(self.mu))
        self.rho = np.zeros(len(self.mu))
    

