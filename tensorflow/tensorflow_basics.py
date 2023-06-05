#!/usr/bin/env python
# coding: utf-8

# # Fundamentals of tensors using tensorflow

# In[1]:


import tensorflow as tf
print(tf.__version__)


# # creating tensors with tf.constant()

# In[2]:


scalar = tf.constant(7)
scalar


# ### check(ndim- number of dimentions)

# In[3]:


scalar.ndim


# In[4]:


# Create a vector
vector = tf.constant([10,10])
vector


# In[5]:


# check ndim
vector.ndim


# In[6]:


# Create a matrix(has more than 1 dimension)
matrix = tf.constant([[10,7],
                     [7,10]])
matrix


# In[7]:


matrix.ndim


# In[8]:


float_matrix = tf.constant([[10.,7.],
                           [5.,3.],
                           [2.3,4.8]], dtype = tf.float16)
float_matrix



# In[9]:


# create a tensor (scalar -> vector -> matrix -> tensors)
tensor = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
tensor


# # # create tensors with variables

# In[10]:


changable_tensor = tf.Variable([10,7])
unchangable_tensor = tf.constant([10,7])

changable_tensor, unchangable_tensor


# In[11]:


#changable_tensor[0]=7 results to an error so...
changable_tensor[0].assign(7)
changable_tensor


# # create random tensors

# In[12]:


random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))

# is random_1 == random_2
random_1, random_2, random_1==random_2


# # shuffle the order of elements in a tensor

# In[13]:


not_shuffled = tf.constant([[10,7],[3,4],[2,5]])

tf.random.shuffle(not_shuffled)


#  ## Other ways to make tensors

# In[14]:


# tf.ones much like numpy.ones()
tf.ones([10,7])


# In[15]:


#tf.zeros
tf.zeros([6,7])


# # NB:You can turn numpy arrays to tensors
# #### *The main difference between numpy arrays and tensors is that tensors run much faster on GPU

# In[16]:


import numpy as np

# X = tf.constant(matrix or tensor) -> we use capital while naming a matrix or tensor
# y = tf.constant(vector) -> we use a small letter while naming a vector
numpy_A = np.arange(1,25, dtype=np.int32)
numpy_A


# In[17]:


# converting numpy array to tensor
A = tf.constant(numpy_A,shape=(2,3,4))
B = tf.constant(numpy_A,shape=(3,8))
c =  tf.constant(numpy_A) # note tha small letter name on vector # not a must but a convention
A,B,c


# # Getting Information from tensors
The following are attributes when dealing with tensors:
    shape
    rank
    axis or dimension 
    size
# # Manipulating tensors (tensor operations)

# In[18]:


# Additional operations
tensor = tf.constant([[10,7],[3,4]])
tensor + 10


# In[19]:


# Multiplication
tensor * 10


# In[20]:


# Subtraction
tensor - 10


# # Matrix multiplication
# ### In Machine Learning, matrix multiplication is one of the most common tensor operation

# In[21]:


#tf.linalg.matmul || tf.matmu   .linalg -> linear algebra --> this gives us the dot product
# tensor * tensor will give us an element-wise result hence differs from dot product, this csn be solved by using '@' sign
tf.matmul(tensor,tensor)


# # Reshaping vs Transopsing Tensors

# In[22]:


transform_X = tf.constant([[1,2],[3,4],[5,6]])
print('Here is the normal tensor: \n',transform_X)
print('Here is a reshaped tensor: \n', tf.reshape(transform_X,shape=(2,3)))
print('Here is a transposed tensor: \n', tf.transpose(transform_X)) # here the axis is flipped


# # Changing the datatype of a tensor

# In[23]:


# Create tensor -> default data types are (int32 || float32)

b = tf.constant([1.7,7.4])
c = tf.constant([7,10])

b.dtype,c.dtype


# In[24]:


# from float32 to float16 -> reduced precision
d = tf.cast(b,dtype=tf.float16)
d


# # Aggregating tensors
# #### Aggregating tensors = condensing them from multiple values down to a smaller amount of values

# In[25]:


# Get absolute values
D = tf.constant([-7,-10])
D


# In[26]:


# The absolute -> tf.abs
tf.abs(D)

Lets go through the following forms of aggregation:
    * Get the minimum
    * Get the maximum
    * Get the mean of a tensor
    * Get the sum of a tensor
# In[27]:


E = tf.constant(np.random.randint(0,100, size = 50))
E


# In[28]:


tf.size(E), E.shape, E.ndim


# In[29]:


# find the min -> .reduce_min()
tf.reduce_min(E)


# In[30]:


# find max -> .reduce_max()
tf.reduce_max(E)


# In[31]:


# find mean -> .reduce_mean()
tf.reduce_mean(E)


# In[32]:


# find the sum -> .reduce_sum()
tf.reduce_sum(E)


# ### variance and standard deviation by imported probability function

# In[33]:


import tensorflow_probability as tfp


# In[34]:


# finding the variance -> tpf.stats.variance() 

tfp.stats.variance(E)


# In[35]:


# finding the standard deviation -> tf.math.reduce_std() NB: use dtype as float32 or float64
tf.math.reduce_std(tf.cast(E, dtype= tf.float32))


# ### Find the positional maximum and minimum

# In[36]:


tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
F


# In[37]:


# Find the positional maximum -> .argmax()
tf.argmax(F)


# In[38]:


# the largest value using its position

F[tf.argmax(F)]


# In[39]:


#find the positional minimum

tf.argmin(F)


# In[40]:


# The value using its position
F[tf.argmin(F)]


# ### Squeezing a tensor ( removing all single dimension )

# In[41]:


tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]),shape = (1,1,1,1,50))
G


# In[42]:


# tf.squeeze() removes any dimension of * 1 * in the tensor
G_squeezed = tf.squeeze(G)
G_squeezed,G_squeezed.shape


# ### One hot encoding tensors 

# In[43]:


# Create a list of indices
some_list = [0,1,2,3] # could be red,green,blue,purple

#One hot encode our list of indices
tf.one_hot(some_list, depth=4)


# ### Squaring, Log, Square root

# In[44]:


H = tf.range(1,10)
H


# In[45]:


# Square
tf.square(H)


# In[46]:


# Square root -> tf.sqrt()
# NB: for finding square root int32 is not allowed,use floats instead

tf.sqrt(tf.cast(H, dtype=tf.float32))


# In[47]:


# find logs -> tf.math.log
tf.math.log(tf.cast(H,dtype=tf.float32))


# ## Tensors and Numpy 
# ### Tensors are built on Numpy, Numpy are built on Arrays

# In[48]:


# create a tensor directly fom numpy
J = tf.constant(np.array([3.,7.,10.]))
J


# In[52]:


# Converting tensors to numpy arrays
np.array(J),type(np.array(J))



# In[51]:


#or
J.numpy()


# In[ ]:




