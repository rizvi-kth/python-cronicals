# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:03:47 2018

@author: Rizvi
"""

import tensorflow as tf
sess = tf.Session()

zeroD = tf.constant(12,dtype=tf.int32)
print("Rank of zeroD : ", sess.run(tf.rank(zeroD)))


oneD = tf.constant([1,4,5],dtype=tf.int16)
print("Rank of oneD %d : %s " %(sess.run(tf.rank(oneD)),sess.run(oneD)))


twoD = tf.constant([[1.2,3.2,7.9],[1.2,3.2,7.9]],dtype=tf.float32)
print("Rank of twoD %d : %s " %(sess.run(tf.rank(twoD)),sess.run(twoD)))


threD = tf.constant([[[1,6,9],[1,3,7]],[[2,6,9],[2,3,7]]],dtype=tf.int32)
print("Rank of threD %d : %s " %(sess.run(tf.rank(threD)),sess.run(threD)))

print("Notice the starting bracket number is same as the rank nr.")

sum_oneD = tf.reduce_sum(oneD)
print("sum_oneD : ", sess.run(sum_oneD ))

# axis=0 > Column-wise sum
# axis=1 > Row-wise sum
sum_twoD = tf.reduce_sum(twoD,axis=1,keepdims=True)
print("sum_twoD : ", sess.run(sum_twoD ))

writer = tf.summary.FileWriter('./tf-tensor', sess.graph)
writer.close()
sess.close()
