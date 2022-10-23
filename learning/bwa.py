import numpy as np
from estimation.fba import FBA

class BWA(FBA):
    '''Baum-Welch algorithm'''
    def __init__(self, distribution, transition, output, observation):
        FBA.__init__(self, distribution, transition, output, observation)

    def __transit(self, prob):
        self.i_transit = np.matrix(np.zeros((self.t, self.s)))
        self.ij_transit = np.zeros((self.t - 1, self.s, self.s))

        for time in range(self.t):
            if time != self.t - 1: y = self.observation[time+1]
            for i in range(self.s):
                self.i_transit[time, i] = self.alpha[time, i] * self.beta[time, i] / prob
                for j in range(self.s):
                    if time != self.t - 1:
                        self.ij_transit[time, i, j] = self.alpha[time, i] * self.transition[i, j] \
                                                * self.output[j, y] * self.beta[time+1, j] / prob

    def __step(self):
        prob = self.forward()
        self.backward()
        self.__transit(prob)

        self.distribution = np.array(self.i_transit[0])[0]
        for i in range(self.s):
            for j in range(self.s):
                ij_sum = sum([self.ij_transit[t][i, j] for t in range(self.t - 1)])
                i_sum = sum([self.i_transit[t, i] for t in range(self.t - 1)])
                self.transition[i, j] = ij_sum / i_sum
            for k in range(self.o):
                k_sum = sum([self.i_transit[t, i] for t in range(self.t) if self.observation[t] == k])
                i_sum = sum([self.i_transit[t, i] for t in range(self.t)])
                self.output[i, k] = k_sum / i_sum
    
    def learn(self, epsilon=0.001):
        prev_prob = self.forward()
        self.__step()
        cur_prob = self.forward()
        while cur_prob - prev_prob > epsilon:
            prev_prob = cur_prob
            self.__step()
            cur_prob = self.forward()
