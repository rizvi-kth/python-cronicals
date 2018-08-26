# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

print(tf.__version__)

# Reset the default graph to avoide grap duplication in Tensorboard.
tf.reset_default_graph()

# hello = tf.constant('Hello, TensorFlow!')
a_place = tf.placeholder(tf.float32, name='Place-A')
# print(a_place)
a_var = tf.Variable(initial_value=0.0, name='Scaler-var')
# print(a_var)
b_var = tf.Variable(initial_value=[1.0, 1.0, 1.0], name='Vector-var')
    

with tf.name_scope("Node_0"):
    assign_vr = tf.assign(a_var, 10.90, name = 'assign_vr')

with tf.name_scope("Node_1"):
    # add_op = tf.assign(a_var, a_place + 1, name='Scaler-assign-increment-Jhony')
    add_sc = tf.add(a_place, assign_vr, name='add_sc')
    
print_node = tf.Print(add_sc, [add_sc], "Intermidiate add_sc:", name='print_node')

with tf.name_scope("Node_2"):
    mul_vct = tf.multiply(b_var, print_node, name='mul_vct')


init_op = tf.global_variables_initializer()

# you can get the operations from the graph
graph = tf.get_default_graph()
# print(graph.get_operations())


with tf.Session() as sess:
    
    # Initialize the variables -  shoud be the first operation
    sess.run(init_op)
    print(sess.run(mul_vct, feed_dict={a_place:0.10}))

    # Place holders must be initialized either from file or manually
    # print(sess.run(a_place, feed_dict={a_place:99.99}))
    # print(sess.run(a_var))
    
    # print(sess.run(add_op, feed_dict={a_place:99.99}))
    # print(sess.run(add_vct , feed_dict={a_place:[4.5, 3.5, 9.9]}))

    # Write the graph to log folder
    writer = tf.summary.FileWriter('./tp-practice', sess.graph)
    
    # You can save different run in different directory and then select indevidual 
    # run in Tensorboard and evaluate.
    # writer = tf.summary.FileWriter('./tp-practice/test1', sess.graph)
        
    writer.close()

    # Command for Tensorboard>> tensorboard --logdir="tp-practice"
    


    
    
    
    
    