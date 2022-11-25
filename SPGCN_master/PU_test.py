import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import math
from tqdm import tqdm
import datetime
import sys
iter=sys.argv[1]
cps=sys.argv[2]
# iter=str(21)
test_data=np.load('pretreatment_val/PaviaU_test_data.npy')
test_label=np.load('pretreatment_val/PaviaU_test_label.npy')
test_location=np.load('pretreatment_val/PaviaU_test_location('+iter+').npy')

data = loadmat('data/PaviaU.mat')
label = loadmat('data/PaviaU_gt.mat')
data=data['paviaU']
label=label['paviaU_gt']
tf.set_random_seed(2)
test_labels=np.argmax(test_label,1)
classes=np.max(test_labels)+1
num_test=np.zeros(classes)
for i in range(classes):
    num_test[i] = float(len(test_labels[test_labels == i]))

epoches=4000
cal_batch=64
gen_batch=32
# batch=16
# test_batch=16
# val_batch=16
if cps==str(3):
    crop_size=np.array([21,9,3]).astype(int)
    cs=np.array([21,9,3]).astype(int)
elif cps==str(1):
    crop_size=np.array([21,15,9,3]).astype(int)
    cs=np.array([21,15,9,3]).astype(int)
elif cps == str(4):
    crop_size=np.array([21,17,13,9,3]).astype(int)
    cs=np.array([21,17,13,9,3]).astype(int)
elif cps == str(0):
    crop_size=np.array([9,7,5,3]).astype(int)
    cs=np.array([9,7,5,3]).astype(int)
elif cps == str(2):
    crop_size=np.array([15,11,7,3]).astype(int)
    cs=np.array([15,11,7,3]).astype(int)
# elif cps == str(5):
#     crop_size=np.array([27,19,11,3,1]).astype(int)
#     cs=np.array([27,19,11,3,1]).astype(int)
# crop_size=np.array([21,15,9,3]).astype(int)
# cs=np.array([21,15,9,3]).astype(int)
# crop_size=np.array([15,11,7,3]).astype(int)
# cs=np.array([15,11,7,3]).astype(int)
# crop_size=np.array([9,7,5,3]).astype(int)
# cs=np.array([9,7,5,3]).astype(int)
idx=[]
for i in range(len(cs)-1):
    a=np.zeros((cs[i],cs[i]),dtype=bool)
    up=left=int((cs[i]-cs[i+1])/2)
    down=right=int((cs[i]+cs[i+1])/2)
    a[up:down,left:right]=True
    mask_cur=np.reshape(a,-1)
    idx_cur=np.argwhere(mask_cur==True).squeeze()
    idx.append(idx_cur)

local_mask=[]
for i in range(1,len(cs)):
    a=np.zeros((cs[0],cs[0]),dtype=bool)
    up=left=int((cs[0]-cs[i])/2)
    down=right=int((cs[0]+cs[i])/2)
    a[up:down,left:right]=True
    mask_cur=np.reshape(a,-1)
    local_mask.append(mask_cur)

data_size=np.int32(cs**2)
gen_batch_size=math.ceil(data_size[0]/gen_batch)
half_crop_size=np.int32(crop_size/2)
num_labels=np.max(label)
features=np.int32(np.append(64*np.ones(len(cs)-1),num_labels))
H,W,S=data.shape
data=np.array(np.float32(data))

for i in tqdm(range(S)):
    data[:,:,i]=(data[:,:,i] - np.mean(data[:,:,i]))/np.std(data[:,:,i])
#取出初始点的邻域下标


def select_local(part_data,location):
    crop_list=[]
    for j in range(1):
        data_crop=[]
        for i in tqdm(range(part_data.shape[0])):
            mask = np.float32(np.zeros([crop_size[j], crop_size[j], S]))
            try :
                up=location[i,0]-half_crop_size[j]
                down=location[i,0]+half_crop_size[j]
            except IOError:
                print("Error")
            left = location[i, 1] - half_crop_size[j]
            right = location[i, 1] + half_crop_size[j]
            up= 0 if up<0 else up
            left = 0 if left < 0 else left
            down= H-1 if down>H-1 else down
            right= W-1 if right>W-1 else right
            mask[half_crop_size[j]-(location[i,0]-up):half_crop_size[j]+down-location[i,0]+1,half_crop_size[j]-(location[i,1]-left):half_crop_size[j]+(right-location[i,1])+1,:]=data[up:down+1,left:right+1,:]
            assert (mask[half_crop_size[j], half_crop_size[j], :] == data[location[i, 0], location[i, 1], :]).all()
            mask=np.reshape(mask,[-1,S])
            data_crop.append(mask)
        crop_list.append(data_crop)
    data_crop=[]
    input_data=np.float32(np.array(crop_list[0]))
    return input_data


data_placeholder=tf.placeholder(shape=[None,data_size[0],S],dtype=tf.float32)
label_placeholder=tf.placeholder(shape=[None,num_labels],dtype=tf.float32)
N_placeholder=tf.placeholder(dtype=tf.float32)
tf_is_training=tf.placeholder(dtype=tf.bool)
#开始构建tensor流图
kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)
bias_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)
l2_regularizer = tf.contrib.layers.l2_regularizer(0.1)
global_step=tf.placeholder(tf.int32)
learning_rate = tf.train.exponential_decay(0.001, global_step,500, 0.5, staircase=True)

next_data = tf.expand_dims(data_placeholder, axis=1)
for i in tqdm(range(gen_batch_size)):
    if i != gen_batch_size-1:
        tile_data = tf.expand_dims(data_placeholder[:,i*gen_batch:(i+1)*gen_batch,], axis=-2)
        minus = tf.subtract(tile_data, next_data)
        a = -tf.reduce_sum(tf.square(minus), -1)
        if i==0:
            dist=a
        else:
            dist=tf.concat([dist, a], 1)
    else:
        tile_data = tf.expand_dims(data_placeholder[:,i*gen_batch:,], axis=-2)
        minus = tf.subtract(tile_data, next_data)
        a = -tf.reduce_sum(tf.square(minus), -1)
        dist = tf.concat([dist, a], 1)
dist = tf.exp(dist / S)
dist = dist + tf.eye(crop_size[0] * crop_size[0])
# topk, _ = tf.nn.top_k(dist, data_size[0] / 2)
# min = tf.reduce_min(topk, axis=-1, keepdims=True)
# zeros = tf.zeros(tf.shape(dist))
# dist = tf.where(dist < min, zeros, dist)

for i in range(len(features)):
    if i==0:
        a=tf.layers.dense(data_placeholder,features[i],name='dense0',kernel_initializer=kernel_initializer,kernel_regularizer=l2_regularizer,bias_initializer=bias_initializer,reuse=tf.AUTO_REUSE)
        next_a=tf.gather(a,idx[i],axis=1)
        res_a=next_a
        cur_adj = tf.boolean_mask(dist, local_mask[i], axis=1)
        cur_adj = cur_adj / tf.reduce_sum(cur_adj, 2, keepdims=True)
        cur_adj = cur_adj
        b=tf.matmul(cur_adj,a)
        c=tf.layers.batch_normalization(b+res_a,training=tf_is_training)
        layer_out=tf.nn.relu(c)
        layer_out=tf.layers.dropout(layer_out,rate=0.5,training=tf_is_training)
    elif i==len(features)-1:
        out_shape=layer_out.get_shape().as_list()
        # layer_out=tf.reduce_mean(layer_out,axis=1)
        layer_out=tf.reshape(layer_out,[-1,out_shape[1]*out_shape[2]])
        a = tf.layers.dense(layer_out, features[i],name='dense'+str(i), kernel_initializer=kernel_initializer,kernel_regularizer=l2_regularizer,bias_initializer=bias_initializer,reuse=tf.AUTO_REUSE)
        output = tf.squeeze(tf.nn.softmax(a))
    else:
        a = tf.layers.dense(layer_out, features[i], name='dense'+str(i),kernel_initializer=kernel_initializer,kernel_regularizer=l2_regularizer,bias_initializer=bias_initializer,reuse=tf.AUTO_REUSE)
        next_a = tf.gather(a, idx[i], axis=1)
        res_a=next_a
        cur_adj = tf.boolean_mask(dist, local_mask[i], axis=1)
        cur_adj = tf.boolean_mask(cur_adj, local_mask[i - 1], axis=2)
        cur_adj = cur_adj / tf.reduce_sum(cur_adj, 2, keepdims=True)
        cur_adj=cur_adj
        b = tf.matmul(cur_adj, a)
        c = tf.layers.batch_normalization(b + res_a,training=tf_is_training)
        layer_out = tf.nn.relu(c)
        layer_out = tf.layers.dropout(layer_out, rate=0.5, training=tf_is_training)
cross_entropy = -tf.reduce_sum(label_placeholder * tf.log(output + 1e-10))
reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
l2_loss = tf.add_n(reg_set) /500
losss=cross_entropy+l2_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losss)
saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
config.gpu_options.allow_growth = True




big_batch=10000
big_batch_num=math.ceil(test_data.shape[0]/big_batch)

import gc
OA = 0
los = 0
test_time=0
predict_all = []
predict_class = np.zeros(classes)
all_acc = np.zeros(classes)
for i in range(big_batch_num):
    if i != big_batch_num-1:
        test_input_data=select_local(test_data[i*big_batch:(i+1)*big_batch],test_location[i*big_batch:(i+1)*big_batch])
        test_input_label=test_label[i*big_batch:(i+1)*big_batch]
    else:
        test_input_data=select_local(test_data[i*big_batch:],test_location[i*big_batch:])
        test_input_label=test_label[i*big_batch:]

    with tf.Session(config=config) as sess:
        # print('final_epoch:', iters)
        # print('final_val_OA:', val_OA)
        # print('final_val_AA:', val_AA)
        # print('final_val_celoss:', val_ce_loss)
        saver.restore(sess, "wjy_data/PaviaU_model.ckpt")
        # if (n==10):
        #     n=0
        #     m=m+1
        # # if epoch > 0:
        tic = datetime.datetime.now()
        start = 0
        for i in tqdm(range(math.ceil(test_input_data.shape[0] / cal_batch))):
            if start + cal_batch <= test_input_data.shape[0]:
                feed_dict = {data_placeholder: test_input_data[start:start + cal_batch],
                             label_placeholder: test_input_label[start:start + cal_batch], tf_is_training: False}
                # for j in range(len(adj_placeholder)):
                #     feed_dict.update({adj_placeholder[j]: test_adj_list_whole[i][j]})
                # feed_dict.update(
                #     {dist_placeholder: np.eye(crop_size[0]**2)[np.newaxis,:]})
                out, loss = sess.run([output, cross_entropy], feed_dict=feed_dict)
                predict = np.float32(np.argmax(out, axis=1))
                predict_all.append(predict)
                lab = np.float32(np.argmax(test_input_label[start:start + cal_batch], 1))
                for k in range(classes):
                    index = (lab == k)
                    f = np.sum((predict[index] == lab[index]) != 0)
                    all_acc[k] = all_acc[k] + float(f)
                for k in range(classes):
                    predict_class[k] = predict_class[k] + len(predict[predict == k])
                acc = np.sum(predict == lab)
                OA = OA + acc
            else:
                feed_dict = {data_placeholder: test_input_data[start:test_input_data.shape[0]],
                             label_placeholder: test_input_label[start:test_input_data.shape[0]], tf_is_training: False}
                # for j in range(len(adj_placeholder)):
                #     feed_dict.update({adj_placeholder[j]: test_adj_list_whole[i][j]})
                # feed_dict.update(
                #     {dist_placeholder: np.eye(crop_size[0] ** 2)[np.newaxis,:]})
                out, loss = sess.run([output, cross_entropy], feed_dict=feed_dict)
                if len(out.shape)==1:
                    predict = np.float32(np.argmax(out))
                else:
                    predict = np.float32(np.argmax(out, axis=1))
                predict_all.append(predict)
                predict = np.append(predict, -1)
                lab = np.float32(np.argmax(test_input_label[start:test_input_data.shape[0]], 1))
                lab=np.append(lab,-2)
                for k in range(classes):
                    index = (lab == k)
                    f = np.sum((predict[index] == lab[index]) != 0)
                    all_acc[k] = all_acc[k] + float(f)
                for k in range(classes):
                    predict_class[k] = predict_class[k] + len(predict[predict == k])
                acc = np.sum(predict == lab)
                OA = OA + acc
            start = start + cal_batch
            los = loss + los
        toc = datetime.datetime.now()
        test_time=test_time+(toc - tic).total_seconds()
    del test_input_data
    gc.collect()

all_acc = all_acc / num_test
AA = np.mean(all_acc)
OA = OA / test_data.shape[0]
Kappa = (OA - np.sum(predict_class * num_test) / (np.sum(num_test) * np.sum(num_test))) / (
        1 - np.sum(predict_class * num_test) / (np.sum(num_test) * np.sum(num_test)))
print('all_accuracy:', all_acc)
print('test_AA:', AA)
print('test_OA:', OA)
# print('power:',pow)
print('Kappa:', Kappa)
print('test_loss:', los)
global_AA = AA
global_OA = OA
global_Kappa = Kappa
global_all_acc = all_acc
global_predict = np.array(predict_all)
print('test_time:', test_time)
with open("./PaviaU_time_result.txt", "a") as f:
    # f.write('train_time:' + str(train_time) + '\n')
    f.write('test_time:' + str(test_time) + '\n')
    f.write('OA:' + str(global_OA) + '\n')
    f.write('AA:' + str(global_AA) + '\n')
    f.write('Kappa:' + str(global_Kappa) + '\n')
    f.write('all_acc:' + str(global_all_acc) + '\n')
    all_class = global_predict
    np.save('pretreatment_val/PaviaU_test_result(' + iter + ').npy', all_class)
    f.close()