import numpy as np 

def reservoirSampling(k, stream) :
    # Initialize an empty reservoir (any containerdata type).
    reservoir = list(np.zeros(k))
    n = 0
    
    for item in stream :
        if n < k :
            # print("n : ", n)
            reservoir[n] = item 
        elif flipBiasedCoin(n/k) :
            random_k = np.random.randint(k, size=1)[0]
            reservoir[random_k] = item
        n = n + 1
    
    return reservoir
            
def flipBiasedCoin(prob_heads=.5) :
    turn = np.random.uniform(0,1)
    return turn < prob_heads 
     
N = 100
k = 9
streams = [ flipBiasedCoin(0.6) for i in range(N)]
size_stream = 10
estimates = float(streams.count(True))
print("True estimates : ", estimates)
count = 0
for j in range(int(N/size_stream)) :
    j_start = j*size_stream;
    j_end = (j+1)*size_stream;
    stream = streams[j_start:j_end]
    samples = reservoirSampling(k, stream)
    count += float(samples.count(True))

print("Apprx estimates : ", count)
    
