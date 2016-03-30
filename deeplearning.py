import numpy as np
import caffe 
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics

caffe.set_mode_cpu() #Use the CPU.
deploy = "/home/raghuram/caffe/models/bvlc_reference_caffenet/deploy.prototxt" # Network to use
model = "/home/raghuram/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"

## Other parameters ##
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

net = caffe.Net(deploy,model,caffe.TEST)

# Do preprocessing on the images
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0) #Scale data to 0-255 range

#Set batch size
net.blobs['data'].reshape(1,3,227,227)

# Load the textfile containing info about the training and test files
train_labels = []
test_labels = []

train_file = open("training.txt","r")
test_file = open("testing.txt","r")

for info in train_file:
    info = info[:-1]
    train_labels.append(int(info.split()[1]))
   
train_labels = np.array(train_labels)

for test_info in test_file:
    test_info = test_info[:-1] 
    test_labels.append(int(test_info.split()[1]))

test_labels = np.array(test_labels)


#total_training_images = len(train_labels)

train_feats_fc7= []
test_feats_fc7= []

train_feats_fc8= []
test_feats_fc8= []

image_ctr = 1

file1 = open("training.txt","r")
file2 = open("testing.txt","r")

for line in file1:
    inpt = line.split()
    image_ctr+=1
    net.blobs["data"].data[...] = transformer.preprocess("data",caffe.io.load_image(inpt[0]))
    out = net.forward()
    feat_fc7= np.array(net.blobs["fc7"].data[0])
    feat_fc8= np.array(net.blobs["fc8"].data[0]) 
    train_feats_fc7.append(feat_fc7)
    train_feats_fc8.append(feat_fc8)
      
train_feats_fc7 = np.array(train_feats_fc7)
train_feats_fc8 = np.array(train_feats_fc8)


for line in file2:
    inpt = line.split()
    image_ctr+=1
    net.blobs["data"].data[...] = transformer.preprocess("data",caffe.io.load_image(inpt[0]))
    out = net.forward()
    feat_fc7 = np.array(net.blobs["fc7"].data[0])
    feat_fc8 = np.array(net.blobs["fc8"].data[0])
    test_feats_fc7.append(feat_fc7)
    test_feats_fc8.append(feat_fc8)

test_feats_fc7 = np.array(test_feats_fc7)
test_feats_fc8 = np.array(test_feats_fc8)

C = 1.0
svc = svm.SVC(kernel = 'linear', C=C).fit(train_feats_fc7,train_labels)
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7,C=0.7).fit(train_feats_fc7,train_labels)
poly_svc = svm.SVC(kernel = 'poly',degree = 3,C=1).fit(train_feats_fc7,train_labels)


for __,clf in enumerate((svc,rbf_svc,poly_svc)):
    print("\n Classification using:")
    print(clf)
    Z = clf.predict(test_feats_fc7)
    print("\n Accuracy is:")
    print(metrics.classification_report(test_labels,Z))


