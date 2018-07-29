
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


sess = tf.Session()


# In[4]:


hello = tf.constant("hello tensorflow")


# In[5]:


print(sess.run(hello))


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import numpy as np


# In[8]:


import math


# In[9]:


num_house = 100
np.random.seed(45)
house_size = np.random.randint(low = 1000, high = 3500, size = num_house)


# In[13]:


np.random.seed(42)
house_cost = house_size *100.0 + np.random.randint(low = 25000, high = 100000, size = num_house)


# In[14]:


plt.plot(house_size,house_cost,"bx")
plt.ylabel("Price")
plt.xlabel("size")
plt.show()


# In[15]:


def normalize(array):
    return (array-array.mean())/array.std()
    


# In[16]:


num_train_samples = math.floor(num_house*0.7)


# In[30]:


train_price = np.asanyarray(house_cost[:num_train_samples:])
train_house  = np.asarray(house_size[:num_train_samples])


# In[31]:


train_size_norm  = normalize(train_house)


# In[32]:


train_price_norm = normalize(train_price)


# In[33]:


test_size = np.array(house_size[num_train_samples:])
test_price = np.array(house_cost[num_train_samples:])


# In[34]:


test_size_norm = normalize(test_size)
test_price_norm = normalize(test_price)


# In[60]:


for (x,y) in zip(train_size_norm,train_price_norm):
    


# In[37]:


tf_house_size = tf.placeholder("float",name="house_size")
tf_price = tf.placeholder("float",name = "price")


# In[68]:


tf_size_factor = tf.Variable(np.random.randn(), name = "size_factor")
tf_price_offcet = tf.Variable(np.random.randn(), name = "price_offcet")


# In[42]:


tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offcet)


# In[43]:


tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price,2))/(2*num_train_samples)


# In[45]:


learning_rate = 0.1


# In[46]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


# In[70]:


init = tf.global_variables_initializer()



# In[64]:


with tf.Session() as sess:
    sess.run(init)
    display_every = 2
    num_training_iter = 50
    
    for iteration in range(num_training_iter):
        for (x,y) in zip(train_size_norm,train_price_norm):
            sess.run(optimizer,feed_dict={tf_house_size:x,tf_price:y})

        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost,feed_dict={tf_house_size:train_size_norm, tf_price: train_price_norm})
            print("iteration #:",'%04d' %(iteration+1),"cost= ","{:.9f}".format(c),            "size_factor = ",sess.run(tf_size_factor),"price_offcet = ",sess.run(tf_price_offcet))

    print("optimization Finished")    
    
    training_cost = sess.run(tf_cost,feed_dict={tf_house_size:train_size_norm,tf_price:train_price_norm})
    
    print("Trained_cost=",training_cost,"size_factor = ",sess.run(tf_size_factor),"price_offset = ",sess.run(tf_price_offcet))
    
    train_house_size_mean = train_house.mean()
    train_house_size_std = train_house.std()
    train_price_mean = train_price.mean()
    train_price_std = train_price.std()
    
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size")
    
    plt.plot(train_house,train_price,'go',label="training set")
    plt.plot(test_size, test_price,'mo',label="Testing data")
    
    plt.plot(train_size_norm*train_house_size_std + train_house_size_mean,
            (sess.run(tf_size_factor)*train_size_norm+sess.run(tf_price_offcet))*train_price_std+train_price_mean,
            label='Learned Regression')
    
    plt.legend(loc='upper left')
    plt.show()


# In[51]:





# In[40]:





# In[29]:





# In[28]:





# In[12]:




