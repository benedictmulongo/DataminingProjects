import numpy as np
import math 
import itertools 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class TriestBase:
    """A simple attempt to model a dog.""" 
    def __init__(self,index, reservoirSize):

        self.M = reservoirSize
        self.data, self.NumberOfEdges, self.NumberOfNodes, self.triangleCount = self.readDataForStats(index)
        self.S = nx.Graph()
        self.KK_global = [0]*(self.NumberOfNodes + 2)
        self.t = 0
        self.index = index
        self.results = [] 
        
    def readData(self, index = 3) :
        data1 = 'lesMiserables.txt'
        data2 = 'mmorenot_train.txt'
        data3 = 'morino_beach.txt'
        data4 = 'euroroad.txt'
        data = [data1, data2, data3, data4] 
        dat = np.loadtxt(data[index])
        return dat[2:], int(dat[1][0]), int(dat[1][1])

    def readDataForStats(self,index = 3) :
        data1 = 'lesMiserables.txt'
        data2 = 'mmorenot_train.txt'
        data3 = 'morino_beach.txt'
        data4 = 'euroroad.txt'
        data = [data1, data2, data3, data4] 
        dat = np.loadtxt(data[index])
        triangleCount = int(dat[0][0])
        return dat[2:], int(dat[1][0]), int(dat[1][1]), triangleCount

    def graph_data(self) :
        g = nx.Graph()
        for x in self.data :
            from_node = x[0]
            to_node = x[1]
            g.add_edges_from([(int(from_node), int(to_node))])
            
        nx.draw_networkx(g, node_color='blue') 
        plt.show()
    
        return 0
    
    def triest_base(self) :
        
        for element in self.data :
            self.t = self.t + 1
            from_node, to_node = element[0], element[1]
            if self.sampleEdge() :
                self.S.add_edges_from([(int(from_node), int(to_node))])
                self.updateCounters('+', (int(from_node), int(to_node))) 
                self.results.append(self.KK_global[-1])
    
        return self.KK_global, self.t
        
    def sampleEdge(self) :
        
        if self.t <= self.M :
            return True 
        elif self.flipBiasedCoin(self.M / self.t) :
            NumberOfEdges = self.S.number_of_edges()
            edgeNumber = np.random.randint(NumberOfEdges, size=1)[0]
            all_edges = self.S.edges()
            randomEdge = all_edges[edgeNumber]
            (u,v) = randomEdge
            self.S.remove_edge(u,v)
            self.updateCounters('-', randomEdge)
            return True 
            
        return False 
        
    def flipBiasedCoin(self, prob_heads=.5) :
        turn = np.random.uniform(0,1)
        return turn < prob_heads 
        
    def updateCounters(self,  sign, randomEdge) :
        
        (u,v) = randomEdge
        N_uS = set(self.S.neighbors(u))
        N_vS = set(self.S.neighbors(v))
        N_uvS = N_uS.intersection(N_vS)
        N_uvS = list(N_uvS)
        valor = len(N_uvS)
        if sign == '+' :
            self.KK_global[-1] = self.KK_global[-1] + valor
            self.KK_global[u] = self.KK_global[u] + valor
            self.KK_global[v] = self.KK_global[v] + valor
            for c in N_uvS :
                self.KK_global[c] = self.KK_global[c] + 1
        else :
            self.KK_global[-1] = self.KK_global[-1] - valor
            self.KK_global[u] = self.KK_global[u] - valor
            self.KK_global[v] = self.KK_global[v] - valor
            for c in N_uvS :
                self.KK_global[c] = self.KK_global[c] - 1
                
        return self.KK_global
        
    def estimateTriangles(self) :
        r = (self.t * (self.t-1)*(self.t-2)) / (self.M *(self.M -1)*(self.M -2))
        r = max(1,r)
        
        return r * self.KK_global[-1]


    def reInit(self, reservoirSize):

        self.M = reservoirSize
        self.S = nx.Graph()
        self.KK_global = [0]*(self.NumberOfNodes + 2)
        self.t = 0
        self.results = [] 

def TriestBasemakeStatistics(index = 2) :

    
    Triest = TriestBase(index = index, reservoirSize = 300)
    triangleCount = Triest.triangleCount
    data_size = Triest.NumberOfEdges
    minimun = data_size - 100
    maximum = data_size + 10
    M = list(np.arange(minimun ,maximum))
    
    estimates = [] 

    for reservoirsize in M :
        Triest.reInit(reservoirsize)
        Triest.triest_base()
        estimates.append(Triest.estimateTriangles()) 
        
    length = len(estimates) 
    correct = [triangleCount]*length
    
    plt.figure()
    plt.plot(M, estimates, 'k',label='Estimation', linewidth=0.5)
    plt.plot(M, correct,'b' ,label='Correct')
    plt.plot(data_size,triangleCount, 'ro', label='t <= M')
    plt.legend( loc='best')
    plt.xlabel("M - reservoir size ")
    plt.ylabel("Triangle count estimation")
    plt.title('TriestBase for data ' + str(index))
    plt.savefig("TriestBaseForData" + str(index) +  ".png")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show() 
    
    
    return np.var(estimates)     


# makeStatistics(3) 
# Triest = TriestBase(index = 1, reservoirSize = 300)
# Triest.graph_data()
# Triest.triest_base()
# print("Estimates = ", Triest.estimateTriangles())
# TriestBasemakeStatistics(index = 0)


