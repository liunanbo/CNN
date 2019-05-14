from PIL import Image
import data_utils as du
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from collections import namedtuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import fileinput

coords, chips, classes = du.get_labels('/Users/nanboliu/Desktop/xview/data/xView_train.geojson')
img_dict={}
for i in range(chips.shape[0]):
    if chips[i] not in ['24.tif']:
        if chips[i] not in img_dict:
            img_dict[chips[i]]=set()    
        img_dict[chips[i]].add(i)

#create a label dictionary
labels = {}
for line in fileinput.input('/Users/nanboliu/Desktop/xview/data/class_labels.txt'):
    labels[int(line.split(":")[0])] = line.split(":")[1].rstrip('\n')
    

class_vector=[keys for keys in labels]
n_class=len(class_vector)
resized_image=(32,32)
dataset = namedtuple('Dataset', ['X', 'y'])
path='/Volumes/Nanbo/xview_data/train_images/'

      
def read_img(file_path, n_class, resized_image):
    objects = []
    labels = []
    for img_name in [x for x in img_dict.keys()][0:400]:
        print(img_name)
        file_path=path+img_name
        img=np.array(Image.open(file_path))
        for ind in img_dict[img_name]:
            coord=coords[ind]
            class_ind=classes[ind]
            if class_ind in class_vector:
                tem=[max(0,int(coord[1])),
                         min(int(coord[3]),img.shape[0]),
                         max(0,int(coord[0])),
                         min(int(coord[2]),img.shape[1])]
                if (tem[1]-tem[0])*(tem[3]-tem[2])!=0:
                    obj=img[tem[0]:tem[1],tem[2]:tem[3]]
                    obj=resize(obj,resized_image,mode='constant')
                    objects.append(obj)
                    label = np.zeros((n_class, ), dtype=np.float32)
                    label[class_vector.index(class_ind)] = 1.0
                    labels.append(label)
    return dataset( X=np.array(objects).astype(np.float32),y=np.array(labels).astype(np.float32))

dataset= read_img(path, n_class,resized_image)









idx_train, idx_test = train_test_split(range(dataset.X.shape[0]),
   test_size=0.25, random_state=101)
X_train = dataset.X[idx_train, :, :, :]
X_test = dataset.X[idx_test, :, :, :]
y_train = dataset.y[idx_train, :]
y_test = dataset.y[idx_test, :]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
   
   
 #train model and make prediction
def minibatcher(X, y, batch_size, shuffle):
   assert X.shape[0] == y.shape[0]
   n_samples = X.shape[0]
   if shuffle:
      idx = np.random.permutation(n_samples)
   else:
      idx = list(range(n_samples))
   for k in range(int(np.ceil(n_samples/batch_size))):
      from_idx = k*batch_size
      to_idx = (k+1)*batch_size
      yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]
  
for mb in minibatcher(X_train, y_train, 100000, True):
    print(mb[0].shape, mb[1].shape)   
   
#model building
def fc_no_activation_layer(in_tensors, n_units):
   w = tf.get_variable('fc_W',
      [in_tensors.get_shape()[1], n_units],
      tf.float32,
      tf.contrib.layers.xavier_initializer())
   b = tf.get_variable('fc_B',
      [n_units, ],
      tf.float32,
      tf.constant_initializer(0.0))
   return tf.matmul(in_tensors, w) + b

def fc_layer(in_tensors, n_units):
   return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))


def conv_layer(in_tensors, kernel_size, n_units):
   w = tf.get_variable('conv_W',
      [kernel_size, kernel_size, in_tensors.get_shape()[3], n_units],
      tf.float32,
      tf.contrib.layers.xavier_initializer())
   b = tf.get_variable('conv_B',
      [n_units, ],
      tf.float32,
      tf.constant_initializer(0.0))
   return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1, 1, 1, 1], 'SAME') +
   b)

def maxpool_layer(in_tensors, sampling):
   return tf.nn.max_pool(in_tensors, [1, sampling, sampling, 1], [1, sampling,
   sampling, 1], 'SAME')

def dropout(in_tensors, keep_proba, is_training):
   return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep_proba),
   lambda: in_tensors)

def model(in_tensors, is_training):
   # First layer: 5x5 2d-conv, 32 filters, 2x maxpool, 20% drouput
   with tf.variable_scope('l1'):
      l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
      l1_out = dropout(l1, 0.8, is_training)
   # Second layer: 5x5 2d-conv, 64 filters, 2x maxpool, 20% drouput
   with tf.variable_scope('l2'):
      l2 = maxpool_layer(conv_layer(l1_out, 5, 64), 2)
      l2_out = dropout(l2, 0.8, is_training)
   with tf.variable_scope('flatten'):
      l2_out_flat = tf.layers.flatten(l2_out)
   # Fully collected layer, 1024 neurons, 40% dropout
   with tf.variable_scope('l3'):
      l3 = fc_layer(l2_out_flat, 1024)
      l3_out = dropout(l3, 0.6, is_training)

   with tf.variable_scope('out'):
      out_tensors = fc_no_activation_layer(l3_out, n_class)
   return out_tensors


tf.reset_default_graph()
max_epochs=10
batch_size=10000
learning_rate=0.001
in_X_tensors_batch = tf.placeholder(tf.float32, shape = (None,
resized_image[0], resized_image[1], 3))
in_y_tensors_batch = tf.placeholder(tf.float32, shape = (None, n_class))
is_training = tf.placeholder(tf.bool)


logits = model(in_X_tensors_batch, is_training)
out_y_pred = tf.nn.softmax(logits)
loss_score = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
labels=in_y_tensors_batch)
loss = tf.reduce_mean(loss_score)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
   
session=tf.Session()
       
session.run(tf.global_variables_initializer())
for epoch in range(max_epochs):
   print("Epoch=", epoch)
   tf_score = []
   for mb in minibatcher(X_train, y_train, batch_size, shuffle = True):           
      tf_output = session.run([optimizer, loss],
                              feed_dict = {in_X_tensors_batch : mb[0],
                                           in_y_tensors_batch : mb[1],
                                               is_training : True})
      tf_score.append(tf_output[1])
   #Print train loss score
   print(" train_loss_score=", np.mean(tf_score))
y_test_pred, test_loss = session.run([out_y_pred, loss],
                                        feed_dict = {in_X_tensors_batch :X_test,
                                                     in_y_tensors_batch : y_test,
                                                     is_training : False})
#print test loss score
print(" test_loss_score=", test_loss)
y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)
y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)
#Report
print(classification_report(y_test_true_classified, y_test_pred_classified))


#save all the variables
saver = tf.train.Saver()
saver.save(session, "/Users/nanboliu/Desktop/xview/CNN/model/model.ckpt")
 
# Restore weights variables from disk.
saver = tf.train.Saver()
sess=tf.Session()
saver.restore(sess, "/Users/nanboliu/Desktop/xview/CNN/model/model.ckpt") 
 
#Prediction
bbox=np.load('/Users/nanboliu/Desktop/xview/CNN/30_rgb+xy/cnn_input_rgb.npy') 
pred= sess.run(out_y_pred,feed_dict = {in_X_tensors_batch :bbox,
                                                     is_training : False})

np.save('bbox_pred_rgb.npy',pred)






