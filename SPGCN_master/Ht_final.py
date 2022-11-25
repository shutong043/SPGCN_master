import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import math
from tqdm import tqdm
import datetime
import sys
iter=sys.argv[1]
cps=sys.argv[2]
# iter=str(0)
# cps=str(0)
train_data=np.load('pretreatment_val/Ht_train_data.npy')
train_label=np.load('pretreatment_val/Ht_train_label.npy')
train_location=np.load('pretreatment_val/Ht_train_location.npy')

val_data=np.load('pretreatment_val/Ht_val_data.npy')
val_label=np.load('pretreatment_val/Ht_val_label.npy')
val_location=np.load('pretreatment_val/Ht_val_location('+iter+').npy')


test_data=np.load('pretreatment_val/Ht_test_data.npy')
test_label=np.load('pretreatment_val/Ht_test_label.npy')
test_location=np.load('pretreatment_val/Ht_test_location('+iter+').npy')

data = loadmat('data/Houston.mat')
label = loadmat('data/Houston_gt.mat')
data=data['Houston']
label=label['Houston_gt']
tf.set_random_seed(2)
# epoches=10
epoches=10000
cal_batch=8
gen_batch=64
# batch=16
# test_batch=16
# val_batch=16
# crop_size=np.array([21,15,9,3]).astype(int)
# cs=np.array([21,15,9,3]).astype(int)
# crop_size=np.array([27,19,11,3]).astype(int)
# cs=np.array([27,19,11,3]).astype(int)
# crop_size=np.array([9,7,5,3]).astype(int)
# cs=np.array([9,7,5,3]).astype(int)

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
half_crop_size=np.int32(crop_size/2)
num_labels=np.max(label)
features=np.int32(np.append(64*np.ones(len(cs)-1),num_labels))
# features=[64,64,64,num_labels]
H,W,S=data.shape
data=np.array(np.float32(data))

#统计各类数目
test_labels=np.argmax(test_label,1)
val_labels=np.argmax(val_label,1)
train_labels=np.argmax(train_label,1)
classes=np.max(test_labels)+1
num_train=np.zeros(classes)
num_val=np.zeros(classes)
num_test=np.zeros(classes)
for i in range(classes):
    num_train[i]=float(len(train_labels[train_labels==i]))
    num_val[i] = float(len(val_labels[val_labels == i]))
    num_test[i] = float(len(test_labels[test_labels == i]))

print('num_train:',num_train)
print('num_val:',num_val)
print('num_test:',num_test)
print('num_total:',np.array(num_val+num_test+num_train))
print('total_train_num:', (train_labels).shape[0])




#数据归一化
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

input_data=select_local(train_data,train_location)
val_input_data=select_local(val_data,val_location)
print()
#
# def compute_adj(innodes,outnodes):
#     inshape=innodes.shape[0]
#     outshape=outnodes.shape[0]
#     innodes=np.tile(np.expand_dims(innodes,0),[outshape,1,1])
#     outnodes=np.tile(np.expand_dims(outnodes,1),[1,inshape,1])
#     minus=innodes-outnodes
#     a = -np.sum(minus ** 2, -1)
#     dist = np.exp(a / S)
#     dist=dist+np.eye(dist.shape[0])
#     # k=int(dist.shape[1]*3/4)
#     # indices = np.argsort(dist, axis=1)
#     # row_indices = indices[:, :k].reshape(-1)
#     # col_indices = np.tile(np.arange(dist.shape[0]), [k, 1]).transpose().reshape(-1)
#     # dist[col_indices, row_indices] = 0
#
#     adj=np.float32(dist/np.expand_dims(np.sum(dist,1),1))
#
#     return adj
#
# # def compute_adj(innodes,outnodes):
# #     innodes=np.expand_dims(innodes,1)
# #     outnodes=np.expand_dims(outnodes,2)
# #     minus=innodes-outnodes
# #     a = -np.sum(minus ** 2, -1)
# #     dist = np.exp(a / S)
# #     adj=np.float32(dist/np.expand_dims(np.sum(dist,1),1))
# #     return adj
#
# # def generate_adj(lists):
# #     adj_list=[]
# #     for j in range(len(lists)-1):
# #         adj=[]
# #         for i in tqdm(range(len(lists[0]))):
# #             adj.append(compute_adj(lists[j][i],lists[j+1][i]))
# #         np.float32(np.array(adj))
# #         adj_list.append(adj)
# #     return adj_list
#
# def generate_adj(data,mask):
#     adj=[]
#     for i in tqdm(range(len(data))):
#         adj.append(compute_adj(data[i],data[i]))
#     adj=np.float32(np.array(adj))
#     adj_list=[]
#     for i in range(len(mask)):
#         if i ==0:
#             cur_adj=adj[:,mask[i],:]
#             cur_adj = np.float32(cur_adj / np.sum(cur_adj, 2,keepdims=True))
#             adj_list.append(cur_adj)
#         else:
#             cur_adj = adj[:, mask[i],:][:,:,mask[i-1]]
#             cur_adj = np.float32(cur_adj / np.sum(cur_adj, 2, keepdims=True))
#             adj_list.append(cur_adj)
#     return adj_list
#
# adj_list=generate_adj(input_data,local_mask)
# val_adj_list=generate_adj(val_input_data,local_mask)

# tile_data=tf.tile(tf.expand_dims(data_placeholder,axis=1),[1,data_placeholder.shape[1],1,1])
# next_data=tf.tile(tf.expand_dims(data_placeholder,axis=-2),[1,1,data_placeholder.shape[1],1])

g1=tf.Graph()

with g1.as_default():
    data_placeholder = tf.placeholder(shape=[None, data_size[0], S], dtype=tf.float32)
    tile_data = tf.expand_dims(data_placeholder, axis=1)
    next_data = tf.expand_dims(data_placeholder, axis=-2)
    minus = tf.subtract(tile_data, next_data)
    a = -tf.reduce_sum(tf.square(minus), -1)
    dist = tf.exp(a / S)
    dist = dist + tf.eye(crop_size[0] * crop_size[0])
    topk, _ = tf.nn.top_k(dist, data_size[0] / 2)
    min = tf.reduce_min(topk,axis=-1,keepdims=True)
    zeros = tf.zeros(tf.shape(dist))
    dist = tf.where(dist < min, zeros, dist)

batch_num=math.ceil(input_data.shape[0]/cal_batch)
val_batch_num=math.ceil(val_input_data.shape[0]/cal_batch)
dist_train=np.zeros([input_data.shape[0],data_size[0],data_size[0]])
dist_val=np.zeros([val_input_data.shape[0],data_size[0],data_size[0]])
with tf.Session(graph=g1) as sess:
    for i in tqdm(range(batch_num)):
        if i!=batch_num-1:
            feed_dict = {data_placeholder: input_data[i*cal_batch:(i+1)*cal_batch]}
            cur_dist = sess.run(dist, feed_dict=feed_dict)
            dist_train[i*cal_batch:(i+1)*cal_batch]=cur_dist
            # if i==0:
            #     dist_train=cur_dist
            # else:
            #     dist_train=np.append(dist_train,cur_dist,axis=0)
        else:
            feed_dict = {data_placeholder: input_data[i * cal_batch:]}
            cur_dist = sess.run(dist, feed_dict=feed_dict)
            dist_train[i * cal_batch:] = cur_dist
            # dist_train = np.append(dist_train, cur_dist, axis=0)
    for i in tqdm(range(val_batch_num)):
        if i!=val_batch_num-1:
            feed_dict = {data_placeholder: val_input_data[i*cal_batch:(i+1)*cal_batch]}
            cur_dist = sess.run(dist, feed_dict=feed_dict)
            dist_val[i * cal_batch:(i + 1) * cal_batch] = cur_dist
            # if i==0:
            #     dist_val=cur_dist
            # else:
            #     dist_val=np.append(dist_val,cur_dist,axis=0)
        else:
            feed_dict = {data_placeholder: val_input_data[i * cal_batch:]}
            cur_dist = sess.run(dist, feed_dict=feed_dict)
            dist_val[i * cal_batch:] = cur_dist

            # dist_val = np.append(dist_val, cur_dist, axis=0)

#邻接矩阵和输入特征的placeholder

data_placeholder=tf.placeholder(shape=[None,data_size[0],S],dtype=tf.float32)
label_placeholder=tf.placeholder(shape=[None,num_labels],dtype=tf.float32)
dist_placeholder=tf.placeholder(shape=[None,data_size[0],data_size[0]],dtype=tf.float32)
N_placeholder=tf.placeholder(dtype=tf.float32)
tf_is_training=tf.placeholder(dtype=tf.bool)
# tf_is_testing=tf.placeholder(dtype=tf.bool)
# tf_is_testing = tf.placeholder(tf.int16)
#开始构建tensor流图
kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)
var_kernel_initializer=tf.random_normal_initializer(mean=1e-8,stddev=0.01)
bias_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)
l2_regularizer = tf.contrib.layers.l2_regularizer(0.1)
global_step=tf.placeholder(tf.int32)
learning_rate = tf.train.exponential_decay(0.001, global_step,500, 0.5, staircase=True)

for i in range(len(features)):
    if i==0:
        a=tf.layers.dense(data_placeholder,features[i],name='dense0',kernel_initializer=kernel_initializer,kernel_regularizer=l2_regularizer,bias_initializer=bias_initializer,reuse=tf.AUTO_REUSE)
        next_a=tf.gather(a,idx[i],axis=1)
        res_a=next_a
        cur_adj = tf.boolean_mask(dist_placeholder, local_mask[i], axis=1)
        cur_adj = cur_adj / tf.reduce_sum(cur_adj, 2, keepdims=True)
        cur_adj = cur_adj
        b=tf.matmul(cur_adj,a)
        c=tf.layers.batch_normalization(b+res_a,training=tf_is_training)
        # c=batch_norm(b+res_a,is_training=tf_is_training,updates_collections=None)
        layer_out=tf.nn.relu(c)
        layer_out=tf.layers.dropout(layer_out,rate=0.5,training=tf_is_training)
    elif i==len(features)-1:
        out_shape=layer_out.get_shape().as_list()
        layer_out=tf.reduce_mean(layer_out,axis=1)
        # layer_out=tf.reshape(layer_out,[-1,out_shape[1]*out_shape[2]])
        a = tf.layers.dense(layer_out, features[i],name='dense'+str(i), kernel_initializer=kernel_initializer,kernel_regularizer=l2_regularizer,bias_initializer=bias_initializer,reuse=tf.AUTO_REUSE)
        output = tf.squeeze(tf.nn.softmax(a))
    else:
        a = tf.layers.dense(layer_out, features[i], name='dense'+str(i),kernel_initializer=kernel_initializer,kernel_regularizer=l2_regularizer,bias_initializer=bias_initializer,reuse=tf.AUTO_REUSE)
        next_a = tf.gather(a, idx[i], axis=1)
        res_a=next_a
        cur_adj = tf.boolean_mask(dist_placeholder, local_mask[i], axis=1)
        cur_adj = tf.boolean_mask(cur_adj, local_mask[i - 1], axis=2)
        cur_adj = cur_adj / tf.reduce_sum(cur_adj, 2, keepdims=True)
        cur_adj=cur_adj
        b = tf.matmul(cur_adj, a)
        c = tf.layers.batch_normalization(b + res_a,training=tf_is_training)
        # c = batch_norm(b + res_a, is_training=tf_is_training, updates_collections=None)
        layer_out = tf.nn.relu(c)
        layer_out = tf.layers.dropout(layer_out, rate=0.5, training=tf_is_training)
cross_entropy = -tf.reduce_sum(label_placeholder * tf.log(output + 1e-10))
reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
l2_loss = tf.add_n(reg_set) /500
losss=cross_entropy+l2_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losss)
global_OA=0
global_AA=0
n=0
m=0
cum=0
block=0
iters=0
val_ce_loss=999999999999
val_accuracy=0
val_AA=0
val_OA=0
saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
config.gpu_options.allow_growth = True

def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('parameters: {}'.format(total_parameters))
    return total_parameters

def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))

count_flops(tf.get_default_graph())
parameters = count()


with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    tic_all = datetime.datetime.now()

    for epoch in range(epoches):
        start=0
        accur = 0
        ce_los_train=0
        l2_losss=0
        perval = np.random.permutation(len(input_data))
        input_data = input_data[perval]
        dist_train = dist_train[perval]
        train_label = train_label[perval]
        for i in range(math.ceil(train_data.shape[0] / gen_batch)):
            if start + gen_batch <= train_data.shape[0]:
                feed_dict ={data_placeholder:input_data[start:start + gen_batch],label_placeholder: train_label[start:start + gen_batch],N_placeholder:np.float32(epoch+1)}
                feed_dict.update({dist_placeholder:dist_train[start:start + gen_batch],global_step:epoch+1,tf_is_training: True})
                _, out, loss,l2_los= sess.run([train_step, output, cross_entropy,l2_loss], feed_dict=feed_dict)
                acc = np.sum(
                    np.float32(np.argmax(out, axis=1) == np.argmax(np.array(train_label[start:start + gen_batch]), axis=1)))
                accur = accur + acc
                ce_los_train = loss + ce_los_train
                l2_losss=l2_losss+l2_los
            else:
                feed_dict = {data_placeholder: input_data[start:train_data.shape[0]],
                             label_placeholder: train_label[start:train_data.shape[0]],N_placeholder:np.float32(epoch+1)}
                feed_dict.update({dist_placeholder:dist_train[start:train_data.shape[0]],global_step: epoch+1,tf_is_training: True})
                _, out, loss,l2_los = sess.run([train_step, output, cross_entropy,l2_loss], feed_dict=feed_dict)
                acc = np.sum(
                    np.float32(np.argmax(out, axis=1) == np.argmax(np.array(train_label[start:start + train_data.shape[0]]), axis=1)))
                accur = accur + acc
                l2_losss = l2_losss + l2_los
                ce_los_train = loss + ce_los_train
            start = start + gen_batch
        accur = accur / train_data.shape[0]
        print('epoch:',epoch)
        print('l2_loss:',l2_losss)
        print('ce_loss:',ce_los_train)
        # print('accuracy:',accur)
        # toc = datetime.datetime.now()
        # print('train_time:',(toc - tic).total_seconds())
        if math.isnan(ce_los_train) or math.isnan(l2_losss):
            break
        # if accur==1:
        #     n=n+1
        # if m==4:
        #     break
# ########################
#         if (epoch>=800 and epoch%20==0) or (accur==1):
        if ce_los_train<15:
        # if epoch >0:
        #     break
            val_acc = np.zeros(classes)
            start = 0
            accur = 0
            ce_los = 0
            l2_losss = 0
            tic = datetime.datetime.now()
            for i in range(math.ceil(val_data.shape[0] / gen_batch)):
                if start + gen_batch <= val_data.shape[0]:
                    feed_dict = {data_placeholder: val_input_data[start:start + gen_batch],
                                 label_placeholder: val_label[start:start + gen_batch],
                                 N_placeholder: np.float32(epoch + 1)}
                    feed_dict.update({dist_placeholder:dist_val[start:start + gen_batch],tf_is_training: False})
                    out, loss, l2_los = sess.run([output, cross_entropy, l2_loss], feed_dict=feed_dict)
                    acc = np.sum(
                        np.float32(
                            np.argmax(out, axis=1) == np.argmax(np.array(val_label[start:start + gen_batch]), axis=1)))
                    accur = accur + acc
                    ce_los = loss + ce_los
                    l2_losss = l2_losss + l2_los
                    predict = np.float32(np.argmax(out, axis=1))
                    val_lab = np.float32(np.argmax(val_label[start:start + gen_batch], 1))
                    for k in range(classes):
                        index = (val_lab == k)
                        f = np.sum((predict[index] == val_lab[index]) != 0)
                        val_acc[k] = val_acc[k] + float(f)
                else:
                    feed_dict = {data_placeholder: val_input_data[start:val_data.shape[0]],
                                 label_placeholder: val_label[start:val_data.shape[0]],
                                 N_placeholder: np.float32(epoch + 1)}
                    feed_dict.update({dist_placeholder:dist_val[start:val_data.shape[0]],tf_is_training: False})
                    out, loss, l2_los = sess.run([output, cross_entropy, l2_loss], feed_dict=feed_dict)
                    acc = np.sum(
                        np.float32(np.argmax(out, axis=1) == np.argmax(
                            np.array(val_label[start:start + val_data.shape[0]]), axis=1)))
                    accur = accur + acc
                    l2_losss = l2_losss + l2_los
                    ce_los = loss + ce_los

                    predict = np.float32(np.argmax(out, axis=1))
                    val_lab = np.float32(np.argmax(val_label[start:val_data.shape[0]], 1))
                    for k in range(classes):
                        index = (val_lab == k)
                        f = np.sum((predict[index] == val_lab[index]) != 0)
                        val_acc[k] = val_acc[k] + float(f)
                start = start + gen_batch
            accur = accur / val_data.shape[0]
            val_acc = val_acc / num_val
            AA = np.mean(val_acc)
            print('val_epoch:', epoch)
            print('val_l2_loss:', l2_losss)
            print('val_ce_loss:', ce_los)
            print('val_OA:', accur)
            print('val_AA:', AA)
            print('val_accuracy:', val_acc)
            toc = datetime.datetime.now()
            print('val_time:', (toc - tic).total_seconds())
            # if (val_ce_loss>ce_los or val_accuracy<accur) or accur==1:
            if AA > (val_AA-0.005) and accur > (val_OA-0.005):
                block=0
                print('------------------------------------------------------------------save-------------------------------------------------------------------')
                iters = epoch
                saver_path = saver.save(sess,"wjy_data/Ht_model.ckpt")
                cum = cum + 1
                if cum == 10:
                    break
                if AA > val_AA:
                    cum=0
                    val_AA = AA
                if  accur > val_OA:
                    cum = 0
                    val_OA = accur
            else:
                block=block+1
                if block>10:
                    break
         # val_ce_loss = ce_los
    toc_all = datetime.datetime.now()
    train_time=(toc_all - tic_all).total_seconds()
    print('train_time:', train_time)
    print('final_val_OA:', val_OA)
    print('final_val_AA:', val_AA)
    print('final_val_celoss:', val_ce_loss)
with open("./Ht_time_result.txt", "a") as f:
    f.write('train_time:' + str(train_time) + '\n')
    f.write('cs:' + str(cs) + '\n')







