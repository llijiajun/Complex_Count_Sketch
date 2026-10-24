# Dataset tools: raw datasets (ratings.csv/tags.csv/airlines arff/kasandr.zip)
# are located at experiment/Distributed JoinSize/datasets/; this script is used to
# generate/compute ground truth.
# Paper correspondence: the ground-truth column of Table 2 / Table 6.

ftrain = open("ratings.csv",'r')



head = ftrain.readline().replace("\n","").split(",")

n=len(head)

data=[[] for i in range(n)]
while True:
    line = ftrain.readline()
    if not line:
        break
    line=line.replace("\n","").split(",")
    if(len(line)!=n):
        print(line)
        break
    for i in range(n):
        data[i].append(line[i])
for i in range(n):
    f=open("ML/ratings/"+head[i],'w')
    f.write("\n".join(data[i]))
    f.close()


ftrain.close()
