import numpy as np
from evaluation.fba import FBA

class BWA(FBA):
    '''Baum-Welch algorithm'''
    def __init__(self, distribution, transition, output, observation):
        FBA.__init__(self, distribution, transition, output, observation)

    def __transit(self):
        self.i_transit = np.matrix(np.zeros((self.t, self.s)))
        self.ij_transit = np.zeros((self.t - 1, self.s, self.s))

        for time in range(self.t):
            if time != self.t - 1: 
                y = self.observation[time+1]

                prob_i, prob_ij = 0, 0
                for i in range(self.s):
                    prob_i += self.alpha[time, i] * self.beta[time, i]
                    for j in range(self.s):
                        prob_ij += self.alpha[time, i] * self.transition[i, j] \
                            * self.output[j, y] * self.beta[time+1, j]

            for i in range(self.s):
                self.i_transit[time, i] = self.alpha[time, i] * self.beta[time, i] / prob_i
                for j in range(self.s):
                    if time != self.t - 1:
                        self.ij_transit[time, i, j] = self.alpha[time, i] * self.transition[i, j] \
                                                * self.output[j, y] * self.beta[time+1, j] / prob_ij

    def __step(self):
        self.forward()
        prob = self.backward()
        self.__transit()

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
        
        return prob
    
    def learn(self, iters=100, epsilon=0.0000001):
        prev_prob = 0 

        for i in range(iters):
            cur_prob = self.__step() 
            if abs(cur_prob - prev_prob) < epsilon: break
            prev_prob = cur_prob

            if i % 10 == 0: print('{} - {}'.format(i, cur_prob))
            