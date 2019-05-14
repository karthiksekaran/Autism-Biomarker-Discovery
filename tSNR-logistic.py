import pandas as pd
import numpy as np
dataset = pd.read_csv("GSE26415-Full.csv")
transpose = dataset.T
transpose.to_csv("GSE26415-Updated.csv", header=False)
dataset = pd.read_csv("GSE26415-Updated.csv")
del dataset['ID_REF']
dataset.to_csv("26415-54613.csv", index=None)
dataset = pd.read_csv("26415-54613.csv").values
#54613
l = []
temp = 0
sum = 0
avg = 0
count = 0
size = len(dataset)
average = []
for j in range(0, 54613):
    for i in dataset:
        temp = i[j]
        sum = sum + temp
    l.append(sum)
    temp = 0
    sum = 0
print("Stage-1")
std_calc = []
stdev = 0
stdev_res = []
std_calc_var = 0
for i in l:
    avg = i/size
    average.append(avg)
for j in range(0,54613):
    for i in dataset:
        std_calc_var = i[j]
        std_calc.append(std_calc_var)
    stdev = np.std(std_calc, ddof=1)
    stdev_res.append(stdev)
    std_calc = []
print("Stage-2")
SNR = []
snr_val = 0
for i in range(0,len(average)):
    snr_val = average[i]/stdev_res[i]
    SNR.append(snr_val)
print(SNR)
print("Stage-3")
aver = 0
for i in SNR:
    aver = aver + i
aver = aver/len(SNR)
print("Threshold:",aver)
sigmoid = []
sig = 0
for i in SNR:
    sig = 1/(1+np.exp(-i))
    sigmoid.append(sig)
print(sigmoid)
print("Stage-4")
aveg = 0
for i in sigmoid:
    aveg = aveg+i
aveg = aveg/len(sigmoid)
print(aveg)
flag = 0
final = []
pos = []
for i in sigmoid:
    flag = flag+1
    if(i>aveg):
       final.append(i)
       pos.append(flag-1)
print(final)
print(pos)
print(len(final))
print("Stage-5")
import pandas as pd
dataset = pd.read_csv("26415-54613.csv")
l = []
for col in dataset.columns:
    l.append(col)
ind = []
for i in pos:
    ind.append(l[i])
print(ind)
csv_update = pd.DataFrame(ind)
csv_update.to_csv("Gene-List-GSE26415.csv", index=False)