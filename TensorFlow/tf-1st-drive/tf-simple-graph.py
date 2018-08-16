# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:06:58 2018

@author: Rizvi
"""

import tensorflow as tf

a = tf.constant(6.5, name='const_a')
b = tf.constant(6.5, name='const_a')
c = tf.constant(6.5, name='const_a')
d = tf.constant(6.5, name='const_a')

square = tf.square(a, name='square_a')
power = tf.pow(b,c,name='pow_b_c')
sqrt = tf.sqrt(d, name='sqrt_d')

total_sum = tf.add_n([square, power, sqrt], name='Total_Sum')

sess = tf.Session()

print("Square of a ", sess.run(square))
print("Power of a ", sess.run(power))
print("Squire root of a ", sess.run(sqrt))

print("Total ", sess.run(total_sum))


writer = tf.summary.FileWriter('./tf-simple-graph', sess.graph)
writer.close()
sess.close()