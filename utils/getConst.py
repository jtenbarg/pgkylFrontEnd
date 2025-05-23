import numpy as np
import csv

def setConst(self):

    paramFile = self.paramFile
    self.filenameBase = paramFile[:paramFile.find('params')]
    
    with open(paramFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 1:
                eps0 = float(row[0].strip())
                mu0 = float(row[1].strip())
                self.B0 = float(row[2].strip())
                self.dimsX = int(row[3].strip())
                self.dimsV = int(row[4].strip())
                self.po = int(row[5].strip())
                self.basis = row[6].strip()
                self.model = row[7].strip()
                self.suffix = row[8].strip()
            elif line_count > 2:
                self.speciesFileIndex.append(row[0].strip())
                self.mu.append(float(row[1].strip()))
                self.q.append(float(row[2].strip()))
                self.temp.append(float(row[3].strip()))
                self.n.append(float(row[4].strip()))
                if len(row) > 5:
                    self.model_spec.append(row[5].strip())
            line_count += 1
    self.mu0 = mu0
    self.eps0 = eps0
    self.c = 1 / np.sqrt(mu0*eps0)
    self.vA =  self.B0 / np.sqrt(np.multiply(self.n, self.mu) * mu0)
    self.vt =  np.sqrt(2.*np.divide(self.temp, self.mu))
    self.omegaP = np.sqrt(np.divide( np.multiply( np.multiply(self.q,self.q), self.n), self.mu)/eps0)
    self.omegaC = np.divide(np.absolute(self.q)*self.B0, self.mu)
    self.d = np.divide(self.c, self.omegaP)
    self.debye = np.divide(self.vt, self.omegaP) / np.sqrt(2.)
    self.nspec = len(self.mu)

    if self.B0 != 0.:
        self.beta = 2.*mu0*np.multiply(self.n,self.temp) / (self.B0*self.B0 )
        self.rho = np.divide(self.vt, self.omegaC)
    else:
        self.beta = np.zeros(len(self.mu))
        self.rho = np.zeros(len(self.mu))
    

