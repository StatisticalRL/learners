import numpy as np
import src.MDPs_discrete.Equivalence.equivalence as sa




#TODO: Make it compatible with novel gymnasium interface and discrete MDP definition.


# ---------------------------------------------------------------------------
class UCRL_SA_oracle:
    def __init__(self,S, A, delta):
        """
        S: list of integers (index of each state)
        A: list of integers (index of each action)
        delta: double
            precision delta
        """
        self.S=S
        self.A=A
        self.nbS  = len(S)
        self.nbA = len(A)
        nbS=  self.nbS
        nbA = self.nbA
        # P and R will be dictionaries that will store the experienced transitions
        # and rewards. Counts will store the number of times a tuple (t,s,a ) is
        # visited.
        self.confR = np.ones((nbS,nbA))
        self.confP = np.ones((nbS,nbA))
        self.Restimate = np.zeros((nbS,nbA))
        self.Pestimate = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.CountsBeforeEpisode = np.zeros((nbS, nbA))
        self.LocalCounts = np.zeros((nbS,nbA))
        self.time =0
        self.CumulativeReward=0.
        self.c = 2 * nbS * nbA/float(delta)
        self.policy = np.ones((nbS,nbA)) / float(nbA) #random uniform policy
        self.NumberOfEpisodes = 0


        self.sortedPestimate = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.indexmap = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.reverseindexmap = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.eqClasses = []
        self.indexEqClass = np.zeros((nbS,nbA))


    def InitEquivalenceClasses(self,g):
        self.eqClasses,self.indexEqClass =  sa.equivalenceClasses(g,0.)
        for s in self.S:
            for a in self.A:
                sP,im,rim = sa.mapping(g,s, a)
                self.sortedPestimate[s][a] = sP
                self.indexmap[s][a] =im
                self.reverseindexmap[s][a] = rim
                #print("eqclasse:", s,a, sP,im,rim)

    def AggEstimate(self, s, a):
        nagg = 0.
        pagg = np.zeros((self.nbS))
        ragg = 0.

        indexeq = (int) (self.indexEqClass[s][a])
        equiv = self.eqClasses[indexeq]
        for e in equiv:
            s2,a2 = e
            # Use sigma o sigma - 1 !!
            #self.sortedPestimate[s][a][     self.indexmap[s][a][.] ] = self.Pestimate[s][a][.] ??
            #self.Pestimate[s2][a2][ self.reverseindexmap[s2][a2][.] ] = self.sortedPestimate[s2][a2][.] ??
            nagg+=( self.CountsBeforeEpisode[s2][a2] + self.LocalCounts[s2][a2])
            pagg+=      self.sortedPestimate[s2][a2]*( self.CountsBeforeEpisode[s2][a2] + self.LocalCounts[s2][a2])
            ragg+=      self.Restimate[s2][a2]*( self.CountsBeforeEpisode[s2][a2] + self.LocalCounts[s2][a2])
        if(nagg==0):
            return np.ones((self.nbS))/float(self.nbS), 0., 1., 1.
        else:
            pagg2 = np.zeros((self.nbS))
            for sn in self.S:
                pagg2[ sn] = pagg[ self.indexmap[s][a][sn]] # ??
            return pagg/float(nagg),ragg/float(nagg), np.sqrt(14*np.log(self.c*self.time) / max(nagg,1)), np.sqrt(2*np.log(self.c*self.time) / max(nagg,1))


    def MaxBellmanIteration(self,V):
        Q = np.zeros((self.nbS,self.nbA))
        V = np.zeros((self.nbS))
        pimax = np.zeros((self.nbS,self.nbA))

        worse_states = np.argsort(V +0.0)
        for s in self.S:
            for a in self.A:
                # print("Bellman s:",s,"a:",a)
                new_prob,ragg,confpagg,confragg = self.AggEstimate(s, a)
                # print("new_prob:", new_prob)
                best_state = worse_states[-1]
                new_prob[best_state] = min(1, new_prob[best_state] + confpagg / 2.0)
                i = 0
                while sum(new_prob) > 1 and i < len(worse_states):
                    state = worse_states[i]
                    prob_update = max(0, 1 - sum(new_prob) + new_prob[state])
                    new_prob[state] = prob_update
                    i += 1
                Q[s][a] = np.sum(new_prob * V)+confragg + ragg
                # print("s:",s,"a:",a,"Q:",Q[s][a])
            V[s] = np.max(Q[s])
            nb = 0.
            for a in self.A:
                if (Q[s,a] ==V[s]):
                    pimax[s][a] = 1.
                    nb+=1.
            pimax[s] = pimax[s]/float(nb)
        # print("Q:",Q)
        return V,pimax

    def ExtendedValueIteration(self,epsilon):
        pimax = np.zeros((self.nbS,self.nbA))
        newV = np.zeros((self.nbS))
        V = np.zeros((self.nbS))
        newV[0]=1.
        Z = newV-V
        print("Extended Value iteration with precision: ",epsilon)
        # print("Z: ", Z)
        while( np.max(Z)- np.min(Z) >= epsilon):
            newV,pimax = self.MaxBellmanIteration(V)
            # print("pimax:",pimax,"Vmax:",newV)
            Z = newV-V
            V=newV
        #print("pimax:", pimax, "Vmax:", newV)
        print("pimax:", pimax)
        return newV,pimax

    def UpdateOptimisticPolicy(self):
        # for s in self.S:
        #     self.policy[s] = np.zeros((self.nbA))
        #     self.policy[s][np.random.randint(self.nbA)]=1.
        # self.eqClasses = self.equivalenceClasses(0.)
        _,pimax = self.ExtendedValueIteration(1./float(np.sqrt(self.time)))
        self.policy = pimax



    def getPolicy(self):
        return self.policy

    def play(self,state):
        """
        :param state: index of a state
        :return: index of the action to be played
        """
        a = (np.random.choice(self.nbA, 1, p=self.policy[state]))[0]
        return a

    def update(self,state,action,reward,nextstate):

        nb = self.CountsBeforeEpisode[state, action]+self.LocalCounts[state,action]
        t = self.time

        self.Restimate[state][action] = (self.Restimate[state][action] *nb + reward) / float(nb + 1)
        self.Pestimate[state][action][nextstate] = (self.Pestimate[state][action][nextstate] * nb + 1) / float(nb + 1)
        self.confR[state][action] =self.confR[state][action]*np.sqrt(max(nb,1)*np.log(self.c*(t+1))/ ((nb+1)*np.log(self.c*t)) )
        self.confP[state][action] =self.confP[state][action]*np.sqrt(max(nb,1)*np.log(self.c*(t+1))/ ((nb+1)*np.log(self.c*t)) )

        self.CumulativeReward+=reward
        self.LocalCounts[state][action]+=1
        self.time+=1
        if (self.LocalCounts[state][action] >= self.CountsBeforeEpisode[state][action]): # Start a novel learning episode
            self.NumberOfEpisodes+=1
            print('Starting a novel episode (',self.NumberOfEpisodes,') at time ',self.time)
            # print("Restimate: ",self.Restimate, "Pestimate :",self.Pestimate)
            for s in self.S:
                for a in self.A:
                    self.CountsBeforeEpisode[state][action]+=self.LocalCounts[state][action]
                    self.LocalCounts[state][action]=0
            # self.UpdateEquivalenceClasses()
            self.UpdateOptimisticPolicy()

# ---------------------------------------------------------------------------
class UCRL_SA:
    def __init__(self,S, A, delta):
        """
        S: list of integers (index of each state)
        A: list of integers (index of each action)
        delta: double
            precision delta
        """
        self.S=S
        self.A=A
        self.nbS  = len(S)
        self.nbA = len(A)
        nbS=  self.nbS
        nbA = self.nbA
        # P and R will be dictionaries that will store the experienced transitions
        # and rewards. Counts will store the number of times a tuple (t,s,a ) is
        # visited.
        self.confR = np.ones((nbS,nbA))
        self.confP = np.ones((nbS,nbA))
        self.Restimate = np.zeros((nbS,nbA))
        self.Pestimate = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.CountsBeforeEpisode = np.zeros((nbS, nbA))
        self.LocalCounts = np.zeros((nbS,nbA))
        self.time =0
        self.CumulativeReward=0.
        self.c = 2 * nbS * nbA/float(delta)
        self.policy = np.ones((nbS,nbA)) / float(nbA) #random uniform policy
        self.NumberOfEpisodes = 0


        self.sortedPestimate = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.indexmap = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.reverseindexmap = np.zeros((nbS,nbA,nbS))#np.ones((nbS,nbA,nbS))/float(nbS)
        self.eqClasses = []
        self.indexEqClass = np.zeros((nbS,nbA))




    def UpdateEquivalenceClasses(self):
        for s in self.S:
            for a in self.A:
                sP,im,rim = self.mapping(s, a)
                self.sortedPestimate[s][a] = sP
                self.indexmap[s][a] =im
                self.reverseindexmap[s][a] = rim
                #print("eqclasse:", s,a, sP,im,rim)

    def mapping(self,s, a):
        sortedP = sorted(self.Pestimate[s][a])
        r = list(range(0, self.nbS))
        indexmap = np.zeros((self.nbS))
        reverseindexmap = np.zeros((self.nbS))
        for i in self.S:
            # print(st[i],",",i, ", " , t[i])
            k = 0
            j = r[k]
            while (sortedP[j] != self.Pestimate[s][a][i] and k < len(r)):
                k += 1
                j = r[k]
            indexmap[i] = j
            reverseindexmap[j] = i
            r.pop(k)
        return sortedP, indexmap, reverseindexmap  # st[indexmap[j]]=t[j]  st[j]=t[reverseindexmap[j]]

    def norm1(self,p1,p2):
        err=0
        for s in self.S:
            err += abs(p1[s] - p2[s])
        return err

    def equivalenceClass(self, s, a, eps):
        equiva = []
        for s2 in self.S:
                for a2 in self.A:
                    #print("1:", self.sortedPestimate[s][a])
                    #print("2:", self.sortedPestimate[s2][a2])
                    #print("3:", eps, ", ",self.confP[s][a],", ", self.confP[s2][a2])
                    if (self.norm1(self.sortedPestimate[s][a],self.sortedPestimate[s2][a2]) <= eps + self.confP[s][a]+ self.confP[s2][a2]):
                        equiva.append([s2, a2])
        return equiva

    #Only used for plotting.
    def equivalenceClasses(self,eps):
        eqclasses = []
        nbeqclasses =0
        stateactionpairs = []
        sasize = 0
        for s in self.S:
                for a in self.A:
                    stateactionpairs.append([s, a])
                    sasize += 1
        # print(stateactionpairs)
        while (sasize > 0):
            s, a = stateactionpairs.pop()
            sasize -= 1
            eqC = self.equivalenceClass(s, a, eps)
            eqclasses.append(eqC)
            nbeqclasses+=1
            self.indexEqClass[s][a] = nbeqclasses-1
            for e in eqC:
                # print(e)
                if (stateactionpairs.count(e) > 0):
                    s,a = e
                    self.indexEqClass[s][a] = nbeqclasses - 1
                    stateactionpairs.remove(e)
                    sasize -= 1
        return eqclasses


    def AggEstimate(self, s, a):
        nagg = 0.
        pagg = np.zeros((self.nbS))
        ragg = 0.

        indexeq = (int) (self.indexEqClass[s][a])
        equiv = self.eqClasses[indexeq]
        for e in equiv:
            s2,a2 = e
            # Use sigma o sigma - 1 !!
            #self.sortedPestimate[s][a][     self.indexmap[s][a][.] ] = self.Pestimate[s][a][.] ??
            #self.Pestimate[s2][a2][ self.reverseindexmap[s2][a2][.] ] = self.sortedPestimate[s2][a2][.] ??
            nagg+=( self.CountsBeforeEpisode[s2][a2] + self.LocalCounts[s2][a2])
            pagg+=      self.sortedPestimate[s2][a2]*( self.CountsBeforeEpisode[s2][a2] + self.LocalCounts[s2][a2])
            ragg+=      self.Restimate[s2][a2]*( self.CountsBeforeEpisode[s2][a2] + self.LocalCounts[s2][a2])
        if(nagg==0):
            return np.ones((self.nbS))/float(self.nbS), 0., 1., 1.
        else:
            pagg2 = np.zeros((self.nbS))
            for sn in self.S:
                pagg2[ sn] = pagg[ self.indexmap[s][a][sn]] # ??
            return pagg/float(nagg),self.Restimate[s][a], np.sqrt(14*np.log(self.c*self.time) / max(nagg,1)), np.sqrt(2*np.log(self.c*self.time) / max(nagg,1))


    def MaxBellmanIteration(self,V):
        Q = np.zeros((self.nbS,self.nbA))
        V = np.zeros((self.nbS))
        pimax = np.zeros((self.nbS,self.nbA))

        worse_states = np.argsort(V +0.0)
        for s in self.S:
            for a in self.A:
                # print("Bellman s:",s,"a:",a)
                new_prob,ragg,confpagg,confragg = self.AggEstimate(s, a)
                # print("new_prob:", new_prob)
                best_state = worse_states[-1]
                new_prob[best_state] = min(1, new_prob[best_state] + confpagg / 2.0)
                i = 0
                while sum(new_prob) > 1 and i < len(worse_states):
                    state = worse_states[i]
                    prob_update = max(0, 1 - sum(new_prob) + new_prob[state])
                    new_prob[state] = prob_update
                    i += 1
                Q[s][a] = np.sum(new_prob * V)+confragg + ragg
                # print("s:",s,"a:",a,"Q:",Q[s][a])
            V[s] = np.max(Q[s])
            nb = 0.
            for a in self.A:
                if (Q[s,a] ==V[s]):
                    pimax[s][a] = 1.
                    nb+=1.
            pimax[s] = pimax[s]/float(nb)
        # print("Q:",Q)
        return V,pimax

    def ExtendedValueIteration(self,epsilon):
        pimax = np.zeros((self.nbS,self.nbA))
        newV = np.zeros((self.nbS))
        V = np.zeros((self.nbS))
        newV[0]=1.
        Z = newV-V
        print("Extended Value iteration with precision: ",epsilon)
        # print("Z: ", Z)
        while( np.max(Z)- np.min(Z) >= epsilon):
            newV,pimax = self.MaxBellmanIteration(V)
            # print("pimax:",pimax,"Vmax:",newV)
            Z = newV-V
            V=newV
        #print("pimax:", pimax, "Vmax:", newV)
        print("pimax:", pimax)
        return newV,pimax

    def UpdateOptimisticPolicy(self):
        # for s in self.S:
        #     self.policy[s] = np.zeros((self.nbA))
        #     self.policy[s][np.random.randint(self.nbA)]=1.
        self.eqClasses = self.equivalenceClasses(0.)
        _,pimax = self.ExtendedValueIteration(1./float(np.sqrt(self.time)))
        self.policy = pimax



    def getPolicy(self):
        return self.policy

    def play(self,state):
        """
        :param state: index of a state
        :return: index of the action to be played
        """
        a = (np.random.choice(self.nbA, 1, p=self.policy[state]))[0]
        return a

    def update(self,state,action,reward,nextstate):

        nb = self.CountsBeforeEpisode[state, action]+self.LocalCounts[state,action]
        t = self.time

        self.Restimate[state][action] = (self.Restimate[state][action] *nb + reward) / float(nb + 1)
        self.Pestimate[state][action][nextstate] = (self.Pestimate[state][action][nextstate] * nb + 1) / float(nb + 1)
        self.confR[state][action] =self.confR[state][action]*np.sqrt(max(nb,1)*np.log(self.c*(t+1))/ ((nb+1)*np.log(self.c*t)) )
        self.confP[state][action] =self.confP[state][action]*np.sqrt(max(nb,1)*np.log(self.c*(t+1))/ ((nb+1)*np.log(self.c*t)) )

        self.CumulativeReward+=reward
        self.LocalCounts[state][action]+=1
        self.time+=1
        if (self.LocalCounts[state][action] >= self.CountsBeforeEpisode[state][action]): # Start a novel learning episode
            self.NumberOfEpisodes+=1
            print('Starting a novel episode (',self.NumberOfEpisodes,') at time ',self.time)
            # print("Restimate: ",self.Restimate, "Pestimate :",self.Pestimate)
            for s in self.S:
                for a in self.A:
                    self.CountsBeforeEpisode[state][action]+=self.LocalCounts[state][action]
                    self.LocalCounts[state][action]=0
            self.UpdateEquivalenceClasses()
            self.UpdateOptimisticPolicy()


