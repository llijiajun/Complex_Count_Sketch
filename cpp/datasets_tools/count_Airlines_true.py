# Dataset tools: raw datasets (ratings.csv/tags.csv/airlines arff/kasandr.zip)
# are located at experiment/Distributed JoinSize/datasets/; this script is used to
# generate/compute ground truth.
# Paper correspondence: the ground-truth column of Table 2 / Table 6.

import sys

from collections import Counter

if len(sys.argv)<2:
    print("lack of the p of lp")
    sys.exit(1)
else:
    p = int(sys.argv[1])

ftrue=open("Airlines"+str(p)+".csv",'w')

ftrue.write("True\n")

for name in ["DepDelay","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier","Origin","Dest","Distance"]:
    f=open("Airlines/"+name,'r')
    data=f.read().split("\n")
    data=Counter(data)
    result=0
    for key, value in dict(data).items():
    	result+=value**p
    ftrue.write(str(result)+"\n")
    f.close()


ftrue.close()

