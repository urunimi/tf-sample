import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

ss = tf.Session()

print(ss.run(adder_node, {a: 3, b: 4.5}))
print(ss.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(ss.run(add_and_triple, {a: 3, b: 4.5}))