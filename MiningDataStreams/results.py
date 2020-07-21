
from triestImprovedClass import *
from triestBaseClass import *

def resultsForAllDatasets() :
    
    TriestBasemakeStatistics(index = 0)
    TriestImprMakeStatistics(index = 0)

    TriestBasemakeStatistics(index = 1)
    TriestImprMakeStatistics(index = 1)
    
    
    TriestBasemakeStatistics(index = 2)
    TriestImprMakeStatistics(index = 2)

    TriestBasemakeStatistics(index = 3)
    TriestImprMakeStatistics(index = 3)
    


def TestVariance() :
    
    index = 0
    M = 200
    TriestA = TriestBase(index = index, reservoirSize = M)
    counter1, t1 = TriestA.triest_base()
    TriestB = TriestImproved(index = index, reservoirSize = M)
    counter2, t2 = TriestB.triest_base()
    variance1 = np.var(TriestA.results) 
    variance2 = np.var(TriestB.results)  
    print("variance1 : ", variance1)
    print("variance2 : ", variance2)
    print("apprx 1 : ", counter1[-1])
    print("apprx 2 : ", counter2[-1])
    triangleCount = TriestA.triangleCount
    data_size = TriestA.NumberOfEdges
    # plt.figure()
    plt.plot(TriestA.results,'k',label='TriestBase', linewidth=0.8)
    # plt.xlabel("t")
    # plt.ylabel("Triangle count estimation")
    # plt.show()
    
    
    # plt.figure()
    plt.plot(TriestB.results,'r',label='TriestImproved')
    plt.plot(data_size,triangleCount, 'bo', label='Correct')
    plt.legend( loc='best')
    plt.title('Reservoir size ' + str(M))
    plt.xlabel("t")
    plt.ylabel("Triangle count estimation")
    plt.show()
    
    
# resultsForAllDatasets()
TestVariance()