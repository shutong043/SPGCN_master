import numpy as np
from scipy.io import loadmat
import random
import sys
iter=sys.argv[1]
# iter=str(0)
data = loadmat('data/Houston.mat')
label = loadmat('data/Houston_gt.mat')
data=data['Houston']
label=label['Houston_gt']
seeds=[22,134,32,14,2,7,56]
se=int(iter)%len(seeds)
random.seed(seeds[se])
print(data.shape)
print(label.shape)
H,W,D=data.shape
long_data=[]
long_label=[]
location=[]
for i in range(H):
    for j in range(W):
        if label[i,j]!=0:
            long_data.append(data[i,j,:])
            long_label.append(label[i,j])
            location.append([i,j])
location=np.array(location)
L=len(long_label)
classes=np.max(long_label)
onehot_label=np.zeros([L,classes])
for i in range(L):
    onehot_label[i,long_label[i]-1]=1

long_data=np.array(long_data)
QTY,bands=long_data.shape
data1=[]
for i in range(bands):
    data1.append((long_data[:,i] - np.mean(long_data[:,i]))/np.std(long_data[:,i]))
long_data=np.transpose(np.array(data1))

num_all=np.zeros(classes)
for i in range(classes):
    num_all[i]=float(len(np.array(long_label)[np.array(long_label)==i+1]))
print(num_all)

permutation = np.random.permutation(long_data.shape[0])
location = location[permutation,:]
long_data = long_data[permutation,:]
long_label = onehot_label[permutation,:]
train_data=[]
train_label=[]
train_location=[]
test_data=long_data
test_label=long_label
test_location=location
index=[]

numbers_H2=np.int32(num_all*0.02)
numbers_H5=np.int32(num_all*0.05)
numbers_H8=np.int32(num_all*0.08)
numbers_H10=np.int32(num_all*0.10)
numbers_H15=np.int32(num_all*0.15)
numbers_H20=np.int32(num_all*0.20)
# numbers_H=[198,190,192,188,186,182,196,191,193,191,181,192,184,181,187]
# numbers_H=[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
# numbers_H=[200,200,200,200,200,200,200,200,200,200,200,200,200,200,200]
numbers_H=[50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
# numbers_H=[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
# numbers_I_5=[int(numbers/2) for numbers in numbers_I]
# numbers_I_5[6]=numbers_I_5[6]+1
# numbers_I_5[8]=numbers_I_5[8]+1
# numbers_I_5[0]=numbers_I_5[0]+1
numbers_P=[133,373,42,62,27,101,27,74,19]
numbers_S=[41,75,40,28,54,80,72,226,125,66,22,39,19,22,146,37]
for i in range(classes):
    k=0
    for j in range(QTY):
        if np.argmax(long_label[j])==i:
            k=k+1
            train_data.append(long_data[j])
            train_label.append(long_label[j])
            train_location.append(location[j])
            index.append(j)

        if k==numbers_H[i]:
            break



train_data=np.array(train_data)
train_label=np.array(train_label)
train_location=np.array(train_location)
permutation = np.random.permutation(train_data.shape[0])
train_data = train_data[permutation,:]
train_label = train_label[permutation,:]
train_location = train_location[permutation,:]

all_data=test_data
all_label=test_label
all_location=test_location
permutation = np.random.permutation(all_data.shape[0])
all_data = all_data[permutation,:]
all_label = all_label[permutation,:]
all_location = all_location[permutation,:]

test_data=np.delete(test_data,np.array(index),axis=0)
test_label=np.delete(test_label,np.array(index),axis=0)
test_location=np.delete(test_location,np.array(index),axis=0)
val_data=[]
val_label=[]
val_location=[]
index=[]
# val_numbers_I=[5,143,83,24, 49, 73, 3, 48, 2, 98, 246, 60, 21, 127, 39,10]
# val_numbers_H=[int(numbers/5) for numbers in numbers_H]
# val_numbers_H=[200,200,200,200,200,200,200,200,200,200,200,200,200,200,200]
val_numbers_H=[50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
# val_numbers_H=np.int32(num_all-numbers_H)
# val_numbers_H=numbers_H
val_numbers_P=numbers_P
val_numbers_S=numbers_S
for i in range(classes):
    k=0
    for j in range(QTY):
        if np.argmax(test_label[j])==i:
            k=k+1
            val_data.append(test_data[j])
            val_label.append(test_label[j])
            val_location.append(test_location[j])
            index.append(j)

        if k==val_numbers_H[i]:
            break
val_data=np.array(val_data)
val_label=np.array(val_label)
val_location=np.array(val_location)
perval=np.random.permutation(len(val_data))
val_data=val_data[perval]
val_label=val_label[perval]
val_location=val_location[perval]
test_data=np.delete(test_data,np.array(index),axis=0)
test_label=np.delete(test_label,np.array(index),axis=0)
test_location=np.delete(test_location,np.array(index),axis=0)
permutation = np.random.permutation(test_data.shape[0])
test_data = test_data[permutation,:]
test_label = test_label[permutation,:]
test_location = test_location[permutation,:]
# val_data=test_data
# val_label=test_label
# val_location=test_location


np.save('pretreatment_val/Ht_train_data.npy',train_data)
np.save('pretreatment_val/Ht_train_label.npy',train_label)
np.save('pretreatment_val/Ht_train_location.npy',train_location)

np.save('pretreatment_val/Ht_val_data.npy',val_data)
np.save('pretreatment_val/Ht_val_label.npy',val_label)
np.save('pretreatment_val/Ht_val_location('+iter+').npy',val_location)


np.save('pretreatment_val/Ht_test_data.npy',test_data)
np.save('pretreatment_val/Ht_test_label.npy',test_label)
np.save('pretreatment_val/Ht_test_location('+iter+').npy',test_location)

np.save('pretreatment_val/Ht_all_data.npy',all_data)
np.save('pretreatment_val/Ht_all_label.npy',all_label)
np.save('pretreatment_val/Ht_all_location.npy',all_location)

