import numpy as np

import numpy as np

class FBA():
    '''Forward-backward algorithm'''
    def __init__(self, distribution, transition, output, observation):
        self.distribution = distribution
        self.transition = transition
        self.output = output 
        self.observation = observation
        self.states = range(len(self.distribution))
        self.t = len(self.observation) 
        self.s = len(self.distribution)
        self.o = self.output.shape[1]
        self.alpha = np.matrix(np.zeros((self.t, self.s)))
        self.beta = np.matrix(np.zeros((self.t, self.s)))
        self.alpha_scales = []
        self.beta_scales = []

    def __scale(self, mtrx, time):
        scale = mtrx.sum(axis=1).item(time)
        for state in range(self.s):
            mtrx[time, state] /= scale
            
        return 1 / scale
                
    def __forward_step(self, time):
        y = self.observation[time]
        for state in range(self.s):
            sum_prev = sum([self.alpha[time-1, s] * self.transition[s, state] for s in range(self.s)])
            self.alpha[time, state] = sum_prev * self.output[state, y]

        self.alpha_scales.append(self.__scale(self.alpha, time))
        
    def forward(self):
        y = self.observation[0]
        for state in range(self.s):
            self.alpha[0, state] = self.distribution[state] * self.output[state, y]

        self.alpha_scales.append(self.__scale(self.alpha, 0))

        for time in range(1, self.t):
            self.__forward_step(time)

        s = 1
        for scale in self.alpha_scales: s *= scale
        result = 1 / s

        return result
            
    def __backward_step(self, time):
        y = self.observation[time+1]
        for state in range(self.s):
            sum_prev = sum([self.transition[state, s] * self.output[s, y] * self.beta[time+1, s] for s in range(self.s)])
            self.beta[time, state] = sum_prev

        self.beta_scales.append(self.__scale(self.beta, time))

    def backward(self):
        for state in range(self.s):
            self.beta[-1, state] = 1

        self.beta_scales.append(self.__scale(self.beta, self.t-1))
        
        for time in range(self.t-2, -1, -1):
            self.__backward_step(time)

        result = 0
        y = self.observation[0]
        for state in range(self.s):
            result += self.beta[0, state] * self.output[state, y] * self.distribution[state]   

        return result