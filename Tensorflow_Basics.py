#use this file as a useful tensorflow reference

import tensorflow as tf

#constants are numbers or tensors that are set and cannot change
const1 = tf.constant(3.0, dtype = tf.float32)

#placeholders are parameters that need to be given input when running a function
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#these are functions based on said constants(can be tensors or just regular numbers)
adder_node = a+b
add_and_triple = adder_node*3

#session needs to be created to run aforementioned functions
sess = tf.Session()

#use functions below to run session, first comes the function, then comes the placeholder numbers.  Functions can also be constants and variables themselves to obtain their values.  
print(sess.run(a+b, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a:[50,32,3], b: [2,4,7]}))
print("value of const1: ",sess.run([const1]))#example of treating constant as function to output value

#variables are numbers or tensors that are set and can change on runtime
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b 


#in order to initialize variables, use this:
init = tf.global_variables_initializer()
sess.run(init)

#running the linear model for various values of x
print(sess.run(linear_model, {x: [1,2,3,4,5,6,7,8,9,10]}))

#finding amount of error using sums of squares linear regression
#y will represent the expected output of the function
y = tf.placeholder(tf.float32)
#tf.square squares the difference in this case
squared_deltas = tf.square(linear_model - y)
#tf.reduce_sum adds all the elements of a tensor together- basically a cumulate function
loss = tf.reduce_sum(squared_deltas)
resultingLoss = sess.run(loss, {x: [1,2,3,4,5,6,7,8,9,10], y:[0,-1,-2,-3,-4,-5,-6,-7,-8,-9]})
print(resultingLoss)
#goal is for W=-1 and b=1

#the loss is massive, so how to reduce this using machine learning?
#tensorflow has a GradientDescentOptimizer that automatically optimizes the variables according to the magnitude of their derivatives with respect to the loss function
optimizer = tf.train.GradientDescentOptimizer(0.01)#0.01 is the learning rate in this case
train = optimizer.minimize(loss)#set objective for optimizer
sess.run(init)#reset original variable values
x_train = [1,2,3,4,5]#training data
y_train = [0,-1,-2,-3,-4]
for i in range(1000): #running optimizer a certain number of times (training loop)
	sess.run(train, {x:x_train, y:y_train})


print("loss value: ",sess.run(loss, {x:x_train, y:y_train}))#gets really close to 0
print("W value: ", sess.run([W]))#gets really close to -1
print("b value: ", sess.run([b]))#gets really close to 1
