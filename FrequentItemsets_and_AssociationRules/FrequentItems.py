import numpy as np
import math 
import itertools 

# def loadDataSet():
#     return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    
def loadData() :
    datContent = [i.strip().split() for i in open("T10I4D100K.dat").readlines()]
    data = [  [int(y) for y in x] for x in datContent ]
    
    return data
    
def loadTestDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    
def loadTestDataSetA():
    return [[1,2,3],[1,4],[4,5],[1,2,4],[1,2,6,4,3],[2,6,3],[2,3,6]]

def findsubsets(s, n): 
    return [set(i) for i in itertools.combinations(s, n)] 
    
def init_pass(data) :
    unique_items = []
    
    for transaction in data :
        for item in transaction :
            if {item} not in  unique_items :
                unique_items.append({item})

    return unique_items
    
def scanData(data, C1, minSupport = 0.4) :
    
    map = {}
    unique_items = set()
    
    for transaction in data :
        for candidate in C1 :
            if candidate.issubset(transaction) :
                
                if len(candidate) < 2 :
                    cand = tuple(candidate)[0]
                    # print("Cand => ", cand)
                    # print("Log => ", tuple(candidate))
                    if cand in map :
                        map[cand]  += 1
                    else :
                        map[cand] = 1      
                else :
                    
                    if tuple(candidate) in map :
                        # print("Cand => ", candidate)
                        # print("Log => ", tuple(candidate))
                        map[tuple(candidate)]  += 1
                    else :
                        map[tuple(candidate)] = 1
                       
          
    map_supp = {k: v / len(data)  for k, v in map.items() if v / len(data) >= minSupport}
    # print("List ---> ", list(map_supp.keys()))
    # keys = list(map_supp.keys())
    # unique_items = [ set(x) for x in keys ]
    unique_items = list(map_supp.keys())
    
    return unique_items, map, map_supp
    
    
def firstScan(data, minSupport = 0.4) :
    
    map = {}
    unique_items = set()
    
    for transaction in data :
        for item in transaction :
            if item in map :
                map[item]  += 1
            else :
                map[item] = 1
                
            unique_items.add(item)
            
    map_supp = {k: v / len(data)  for k, v in map.items() if v / len(data) >= minSupport}
    
    return map, unique_items, map_supp
    

    
def candidate_generation(Fk, k) :
    
    length_k = len(Fk)
    Ck = []
    
    element = Fk[0]
    if isinstance(element, int) :
        Fk = [ [x] for x in Fk]
    else :
        Fk = [ list(x) for x in Fk]

    
    for i in range(length_k) :
        for j in range(i+1, length_k) :
            
            L1 = list(Fk[i])
            L2 = list(Fk[j])
            L1.sort()
            L2.sort()
            
            if (L1[:-1] == L2[:-1]) and (L1[-1] != L2[-1]) :
                candidate = set(L1) | set(L2)
                #print("ok -> ", candidate)
                subsets =  findsubsets(candidate, k-1)
                ToAdd = True
                for subset in subsets :
                    if list(subset) not in Fk :
                        ToAdd = False
                
                if ToAdd :
                    Ck.append(candidate)
            
    return Ck


def apriori_algorithm(dataset, minSupp) :
    
    print("Here (1) -----> ")
    C1 = init_pass(dataset)
    unique_items, maps, map_supp = scanData(dataset, C1, minSupp)
    #print("map : ", map)
    A = []
    B = []
    C = [] 
    A.append(unique_items)
    #B.append(map)
    C.append(map_supp)
    k = 2
    print("Here -----> ")
    while(unique_items) :
        Ck = candidate_generation(unique_items, k)
        unique_items, maper, map_supp = scanData(dataset, Ck, minSupp)
        maps.update(maper)
        A.append(unique_items)
        #B.append(map)
        C.append(map_supp)
        k += 1
    
    #print("maper : ", map)
    return A, maps, C
    

def generate_rules(dataset,minSupp = 0.5, conf = 0.7 ) :
    
    dataTemp = dataset
    uniques, map, map_support = apriori_algorithm(dataset, minSupp) 
    rules = []
    print("Uniques => ")
    print(uniques)
    print()
    print("Map_supp => ")
    print(map_support)
    print()
    #print("Map : ===> ", map)
    for cnt, f in enumerate(uniques) :
        if cnt >= 1 : 
            for itemset in f :
                length_f = len(itemset)
                for i in range(1,length_f) :
                    subsets = findsubsets(itemset, i)
                    for beta in subsets :
                        f_b = set(itemset) - beta
                        
                        confidence = map[itemset] 
                        if len(f_b) <= 1 :
                            confidence = confidence * 1.0 / map[list(f_b)[0]] 
                        else :
                            confidence = confidence * 1.0 / map[tuple(f_b)] 
                            
                        if confidence >= conf :
                            rules.append((f_b, beta))
                        # print("F_i - beta ==> ", f_b)
                        # print("beta ==> ", beta)
                        # print("Conf : ", confidence)
                        # print("*********"*4)
    rulesOnes = []   
    if len(uniques) <= 2 :

        data_as_set = [ set(x) for x in dataset ]
        countA = 0
        countB = 0
        datr = uniques[0]
    
        length_k = len(datr)
        for i in range(length_k) :
            for j in range(i+1, length_k) :
                Aset = set((datr[i],datr[j] ))
                Bset = set([datr[j]])
    
                for dear in data_as_set :
                    if Aset.issubset(dear) :
                        countA += 1
                    if Bset.issubset(dear) :
                        countB += 1
                
                confidence = countA / countB
                if confidence >= conf :
                    rulesOnes.append((Bset, set([datr[i]])))
    print("RulesOnes => ",rulesOnes)
    
    return rules
    

#dat = loadTestDataSet()
#dat = loadTestDataSetA()
dat = loadData() 
minSupp = 0.01
# conf = 0.45
conf = 0.50
print("Begining --> ")
# A,map,  C = apriori_algorithm(dat, minSupp) 
# print("Uniques => ")
# print(A)
# print()
# print("Map_supp => ")
# print(C)
# print()
rules = generate_rules(dat,minSupp,conf)
print("The rules !!!! ")
print(rules)