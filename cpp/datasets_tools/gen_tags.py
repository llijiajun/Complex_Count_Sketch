# Dataset tools: raw datasets (ratings.csv/tags.csv/airlines arff/kasandr.zip)
# are located at experiment/Distributed JoinSize/datasets/; this script is used to
# generate/compute ground truth.
# Paper correspondence: the ground-truth column of Table 2 / Table 6.

ftrain = open("tags.csv",'r')



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
        #break
        data[0].append(line[0])
        data[1].append(line[1])
        #print("".join(line[2:(len(line)-1)]))
        data[2].append("".join(line[2:(len(line)-1)]))
        #print(line[-1])
        data[3].append(line[-1])
        #break
    else:    
        for i in range(n):
            data[i].append(line[i])
for i in range(n):
    f=open("ML/tags/"+head[i],'w')
    f.write("\n".join(data[i]))
    f.close()


ftrain.close()
