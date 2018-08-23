# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
#hello = tf.constant('Hello, TensorFlow!')
a_place = tf.placeholder(tf.float32, name='Jhony-place')
#print(a_place)
a_var = tf.Variable(initial_value=0.0, name='Scaler-var')
#print(a_var)
b_var = tf.Variable(initial_value=[0.0, 0.0, 0.0], name='Vector-var')
    

# Declare only the type of the placeholder
with tf.name_scope("Equation_1"):
    #add_op = tf.assign(a_var, a_place + 1, name='Scaler-assign-increment-Jhony')
    add_op = tf.assign(a_var, tf.add(a_place, 1, name='Add-place'), name='Scaler-assign-increment-Jhony')

with tf.name_scope("Equation_2"):
    add_vct = tf.assign(b_var, tf.add(a_place, 1, name='Add-place'), name='Vector-assign-increment-Jhony')


init_op = tf.global_variables_initializer()

# you can get the operations from the d
graph = tf.get_default_graph()
#print(graph.get_operations())


with tf.Session() as sess:
    
    #Initialize the variables -  shoud be the first operation
    sess.run(init_op)
    
    # Place holders must be initialized either from file or manually
    #print(sess.run(a_place, feed_dict={a_place:99.99}))
    #print(sess.run(a_var))
    
    print(sess.run(add_op, feed_dict={a_place:99.99}))    
    print(sess.run(add_vct , feed_dict={a_place:[4.5, 3.5, 9.9]}))

    # Write the graph to log folder
    writer = tf.summary.FileWriter('./tp-practice', sess.graph)
    writer.close()
    
    #Command for Tensorboard>> tensorboard --logdir="tp-practice"
    


    
    
    
    
    