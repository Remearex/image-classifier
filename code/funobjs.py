import numpy as np

class ZtoAFun:
    def scalarRelu(self, x):
        pass

    def scalarReluDeriv(self, x):
        pass

    def eval(self, next_z):
        vectorRelu = np.vectorize(self.scalarRelu)
        return vectorRelu(next_z)
    
    def deriv(self, next_z):
        vectorReluDeriv = np.vectorize(self.scalarReluDeriv)
        return vectorReluDeriv(next_z)


class Relu(ZtoAFun):
    def scalarRelu(self, x):
        return max(0, 0.01 * x)
    
    def scalarReluDeriv(self, x):
        if x >= 0:
            return 0.01
        else:
            return 0