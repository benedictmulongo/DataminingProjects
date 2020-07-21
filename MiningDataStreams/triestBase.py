import numpy as np
import math 
import itertools 
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def readData(index = 3) :
    data1 = 'lesMiserables.txt'
    data2 = 'mmorenot_train.txt'
    data3 = 'morino_beach.txt'
    data4 = 'euroroad.txt'
    data = [data1, data2, data3, data4] 
    dat = np.loadtxt(data[index])
    return dat[2:], int(dat[1][0]), int(dat[1][1])
    
def readDataForStats(index = 3) :
    data1 = 'lesMiserables.txt'
    data2 = 'mmorenot_train.txt'
    data3 = 'morino_beach.txt'
    data4 = 'euroroad.txt'
    data = [data1, data2, data3, data4] 
    dat = np.loadtxt(data[index])
    triangleCount = int(dat[0][0])
    return dat[2:], int(dat[1][0]), int(dat[1][1]), triangleCount
    
def graph_data(data) :
    g = nx.Graph()
    for x in data :
        from_node = x[0]
        to_node = x[1]
        # weight = x[2]
        # g.add_weighted_edges_from([(int(from_node), int(to_node), int(weight))])
        g.add_edges_from([(int(from_node), int(to_node))])
        
    nx.draw_networkx(g, node_color='blue') 
    plt.show()

    
    return 0
        
def triest_base( M = 6) :
    data, NumberOfEdges, NumberOfNodes = readData()
    counter = [0]*(NumberOfNodes + 1)
     
    S = nx.Graph()
    t = 0
    counter = [0]*(NumberOfNodes + 3)
    
    
    for element in data :
        t = t + 1
        from_node, to_node = element[0], element[1]
        if sampleEdge(S,(from_node,to_node), t, M, NumberOfEdges, counter) :
            S.add_edges_from([(int(from_node), int(to_node))])
            updateCounters(S,'+', (int(from_node), int(to_node)), counter)

    return counter, t
    
def sampleEdge(S, edge, t, M, NumberOfEdges, KK_global) :
    
    if t <= M :
        return True 
    elif flipBiasedCoin(M/t) :
        
        NumberOfEdges = S.number_of_edges()
        edgeNumber = np.random.randint(NumberOfEdges, size=1)[0]
        all_edges = S.edges()
        randomEdge = all_edges[edgeNumber]
        (u,v) = randomEdge
        S.remove_edge(u,v)
        updateCounters(S, '-', randomEdge, KK_global)
        return True 
    
    return False 
    
def flipBiasedCoin(prob_heads=.5) :
    turn = np.random.uniform(0,1)
    return turn < prob_heads 
    
def updateCounters(graf, sign, randomEdge, KK_global) :
    
    (u,v) = randomEdge
    N_uS = set(graf.neighbors(u))
    N_vS = set(graf.neighbors(v))
    N_uvS = N_uS.intersection(N_vS)
    N_uvS = list(N_uvS)
    valor = len(N_uvS)
    if sign == '+' :
        KK_global[-1] = KK_global[-1] + valor
        KK_global[u] = KK_global[u] + valor
        KK_global[v] = KK_global[v] + valor
        for c in N_uvS :
            KK_global[c] = KK_global[c] + 1
    else :
        KK_global[-1] = KK_global[-1] - valor
        KK_global[u] = KK_global[u] - valor
        KK_global[v] = KK_global[v] - valor
        for c in N_uvS :
            KK_global[c] = KK_global[c] - 1

    return KK_global


def networkGraph() :
    
    g = nx.Graph()
    g.add_node(1)
    print(g.nodes())
    
    g.add_nodes_from([2, 3, 4])
    print(g.nodes())

    g.add_edges_from([(1,2),(3, 4), (5, 6)])
    print(g.edges())
    g.add_weighted_edges_from([(7, 8, 1.5), (9, 10,3.5)])
    
    print(g.edges(data = True))

    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()
    
    return 0


def makeStatistics(index = 2) :
    
    data, NumberOfEdges, NumberOfNodes, triangleCount = readDataForStats(index)
    graph_data(data)
    
    data_size = NumberOfEdges
    minimun = data_size - 100
    maximum = data_size + 10
    M = list(np.arange(minimun ,maximum))
    
    estimates = [] 

    for reservoirsize in M :
        counter, t = triest_base(reservoirsize) 
        r = (t * (t-1)*(t-2)) / (reservoirsize*(reservoirsize-1)*(reservoirsize-2))
        r = max(1,r)
        estimates.append(r*counter[-1]) 
        
    length = len(estimates) 
    correct = [triangleCount]*length
    
    plt.figure()
    plt.plot(M, estimates, 'k',label='Estimation', linewidth=0.5)
    plt.plot(M, correct,'b' ,label='Correct')
    plt.plot(data_size,triangleCount, 'ro', label='t <= M')
    plt.legend( loc='best')
    plt.xlabel("M - reservoir size ")
    plt.ylabel("Triangle count estimation")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show() 
    
    
    return 0 
    
    
def test() :
    index = 3
    data, NumberOfEdges, NumberOfNodes = readData(index)
    graph_data(data)
    
    print("Edges = ", NumberOfEdges)
    print("NumberOfNodes = ", NumberOfNodes)
    print()
    print()
    print("Test Triest : ")
    M = 337
    
    counter, t = triest_base(M) 
    print()
    print()
    print("Counter : ")
    print(counter)
    print("t : ")
    print(t)
    
    r = (t * (t-1)*(t-2)) / (M*(M-1)*(M-2))
    r = max(1,r)
    print("r = ", r)
    print("Total = ", counter[-1])
    print("Traingle count = ", r * counter[-1])

makeStatistics(3) 