# Dataset tools: raw datasets (ratings.csv/tags.csv/airlines arff/kasandr.zip)
# are located at experiment/Distributed JoinSize/datasets/; this script is used to
# generate/compute ground truth.
# Paper correspondence: the ground-truth column of Table 2 / Table 6.

import sys

from collections import Counter

# if len(sys.argv)<2:
#     print("lack of the p of lp")
#     sys.exit(1)
# else:
#     p = int(sys.argv[1])

ftrue=open("Join3True.csv",'w')

lis = ["uniform.dat","zipf1.1.dat","zipf1.5.dat"]

for i in range(3):
    for j in range(i,3):
        for t in range(j,3):
            result=0
            f1 = open("../Lp/"+lis[i],'r')
            f2 = open("../Lp/"+lis[j]+"2",'r')
            f3 = open("../Lp/"+lis[j]+"3",'r')
            data1=f1.read().split("\n")
            data2=f2.read().split("\n")
            data3=f3.read().split("\n")
            data1=dict(Counter(data1))
            data2=dict(Counter(data2))
            data3=dict(Counter(data3))
            for key, value in data1.items():
                if key in data2.keys() and key in data3.keys():
                    result+=value*data2[key]*data3[key]

            ftrue.write(str(result)+"\n")
            f1.close()
            f2.close()
            f3.close()

    

ftrue.close()

