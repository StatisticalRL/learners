import numpy as np
import pylab as pl


def mapping(g,s,a):
    #s: int, a:int
    #_,t = g.transition(g.stateOfIndex(s),g.actionOfIndex(a))
    t = g.getTransition(s,a)
    st=sorted(t)
    r = list(range(0,len(t)))
    indexmap = np.zeros((len(t)))
    reverseindexmap = np.zeros((len(t)))
    for i in range(0,len(t)):
        # print(st[i],",",i, ", " , t[i])
        k = 0
        j = r[k]
        while (st[j] != t[i] and k< len(r)):
            k+=1
            j= r[k]
        indexmap[i]=j
        reverseindexmap[j]=i
        r.pop(k)

    return st,indexmap,reverseindexmap  #st[indexmap[j]]=t[j]  st[j]=t[reverseindexmap[j]]


def compare(g,s1,a1,s2,a2):
    #_,t1 = g.transition(g.stateOfIndex(s1),g.actionOfIndex(a1))
    #_,t2 = g.transition(g.stateOfIndex(s2),g.actionOfIndex(a2))
    t1 = g.getTransition(s1,a1)
    t2 = g.getTransition(s2, a2)
    st1=sorted(t1)
    st2=sorted(t2)
    err = 0
    for i in range(g.nS):
        err+= abs(st1[i]-st2[i])
    return err

def equivalenceClass(g,s,a,eps):
    equiva = []
    for s2 in range(g.nS):
            for a2 in range(g.nA):
                if (compare(g,s,a,s2,a2)<=eps):
                    equiva.append([s2,a2])
    return equiva

def equivalenceClasses(g,eps):
    eqclasses = []
    stateactionpairs = []
    sasize =0
    nbeqclasses =0
    #indexEqClass = np.zeros((g.nS,g.nA))
    indexEqClass =  np.empty((g.nS,g.nA), dtype=int)
    for s in range(g.nS):
            for a in range(g.nA):
                indexEqClass[s,a]=0
                stateactionpairs.append([s,a])
                sasize+=1
    # print(stateactionpairs)
    while(sasize>0):
        s,a =  stateactionpairs.pop()
        sasize-=1
        eqC = equivalenceClass(g,s,a,eps)
        eqclasses.append(eqC)
        nbeqclasses+=1
        indexEqClass[s][a] = nbeqclasses-1
        for e in eqC:
            # print(e)
            if(stateactionpairs.count(e)>0):
                s,a=e
                indexEqClass[s][a] = nbeqclasses - 1
                stateactionpairs.remove(e)
                sasize-=1
    return eqclasses,indexEqClass



def plotGridWorldEquivClasses(g, eqclasses, folder=".", numFigure=1):
    nbFigure = pl.gcf().number + 1
    pl.figure(nbFigure)
    actions = g.nameActions
    equiv0 = np.zeros((g.sizeX,g.sizeY))
    equiv1 = np.zeros((g.sizeX,g.sizeY))
    equiv2 = np.zeros((g.sizeX,g.sizeY))
    equiv3 = np.zeros((g.sizeX,g.sizeY))
    numq=0
    eqClasses = sorted(eqclasses,key=lambda x: len(x))
    for eq in eqClasses:
        numq+=1
        for e in eq:
            x,y=g.from_s(e[0])
            if(g.maze[x][y]>0):
                if(e[1]==0):
                    equiv0[x][y]=numq
                if(e[1]==1):
                    equiv1[x][y] = numq
                if(e[1]==2):
                    equiv2[x][y] = numq
                if(e[1]==3):
                    equiv3[x][y] = numq
    f, axarr = pl.subplots(2, 2)
    axarr[0, 0].imshow(equiv0, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[0, 0].set_title(actions[0])
    axarr[0, 1].imshow(equiv1, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[0, 1].set_title(actions[1])
    axarr[1, 0].imshow(equiv2, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[1, 0].set_title(actions[2])
    axarr[1, 1].imshow(equiv3, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[1, 1].set_title(actions[3])
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    pl.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    pl.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    pl.savefig('Classes.png')



def displayGridworldEquivalenceClasses(g, eps):
    eqClasses,indexEqClass=equivalenceClasses(g,eps)
    plotGridWorldEquivClasses(g, eqClasses)