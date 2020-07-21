import pandas as pd
import csv
import json

#Initialization of the dictionary
#dataset = {"textA":[] ,"textB":[] , "duplicate":[]  }
dataset = {"texts":[]  }

with open('questions.csv', encoding="utf8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter=',')
    for i, row in enumerate(reader):
        if i > 0 :
            # dataset['textA'].append(row[3])
            # dataset['textB'].append(row[4])
            # dataset['duplicate'].append(int(row[5]))
            dataset['texts'].append(row[3])
            dataset['texts'].append(row[4])
            
        if i > 1000 :
            break  
            # print()
            # print(row)
            # print(row[3])
            # print(row[4])
            
# 
#Save to file
f = open('dataDuplicates.json', 'w')
json.dump(dataset, f, indent=2)
f.close()



        
