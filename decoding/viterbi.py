import numpy as np 

delta_min, delta_max = 2 ** (-1022), -(2 ** 1022)

class Viterbi():
    def __init__(self, distribution, transition, output, observation):
        self.distribution = distribution
        self.transition = transition
        self.output = output 
        self.observation = observation
        self.states = range(len(self.distribution))
        self.t = len(self.observation) 
        self.s = len(self.distribution)
        self.o = self.output.shape[1]

    def __step(self, time):
        for state in range(self.s):
            prev = [self.delta[time-1, i] * self.transition[i, state] for i in range(self.s)]
            prev_max = max(prev)
            y = self.observation[time]
            self.delta[time, state] = self.output[state, y] * prev_max
            self.psi[time, state] = prev.index(prev_max)

    def _delta_psi_getter(self):
        self.delta = np.zeros((self.t, self.s))
        self.psi = np.zeros((self.t, self.s))

        for state in range(self.s):
            y = self.observation[0]
            self.delta[0, state] = self.distribution[state] * self.output[state, y] 

        for time in range(1, self.t):
            self.__step(time)

        final_delta = self.delta[-1].max()
        final_psi, = np.where(self.delta[-1] == final_delta)[0]

        return final_psi
    
    def decode(self):
        result = []
        result.append(int(self._delta_psi_getter()))

        for time in range(self.t-1, 0, -1):
            result.append(int(self.psi[time, result[-1]]))
        
        return result[::-1]

class altViterbi(Viterbi):
    def __init__(self, distribution, transition, output, observation):
        Viterbi.__init__(self, distribution, transition, output, observation)
        self.new_output = np.zeros((self.s, self.o)) 

    def __preprocess(self):
        prep = lambda el: np.log(el) if el > delta_min else delta_max
        
        for i in range(self.s):
            self.distribution[i] = prep(self.distribution[i])

            for j in range(self.s):
                self.transition[i, j] = prep(self.transition[i, j])

    def __initialization(self):
        self.delta = np.zeros((self.t, self.s))
        self.psi = np.zeros((self.t, self.s))

        y = self.observation[0]
        
        for state in range(self.s):
            self.new_output[state, y] =  np.log(delta_min) if self.output[state, y] == 0 else np.log(self.output[state, y]) 
            self.delta[0, state] = self.distribution[state] + self.new_output[state, y]

    
    def __alt_step(self, time):
        y = self.observation[time]

        for state in range(self.s):
            self.new_output[state, y] = np.log(delta_min) if self.output[state, y] == 0 else np.log(self.output[state, y]) 

            prev = [self.delta[time-1, i] + self.transition[i, state] for i in range(self.s)]
            prev_max = max(prev)
            self.delta[time, state] = prev_max + self.new_output[state, y]
            self.psi[time, state] = prev.index(prev_max)
    
    def alt_decode(self):     
        self.__preprocess()
        self.__initialization()
        for time in range(1, self.t):
            self.__alt_step(time)

        final_delta = self.delta[-1].max()
        final_psi, = np.where(self.delta[-1] == final_delta)[0]
        result = []
        result.append(int(final_psi))

        for time in range(self.t-1, 0, -1):
            result.append(int(self.psi[time, result[-1]]))
        
        return result[::-1]
        