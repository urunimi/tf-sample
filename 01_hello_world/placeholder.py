import tensorflow as tf

#Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

ss = tf.Session()

print(ss.run(adder_node, {a: 3, b: 4.5}))
print(ss.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(ss.run(add_and_triple, {a: 3, b: 4.5}))

# Variable
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
ss.run(init)

print(ss.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
print('linear_model - y: {}'.format(ss.run(linear_model - y, {x:[1, 2, 3, 4], y: [0, -1, -2, -3]})))
squared_deltas = tf.square(linear_model - y)
print('squared_deltas: {}'.format(ss.run(squared_deltas, {x:[1, 2, 3, 4], y: [0, -1, -2, -3]})))
loss = tf.reduce_sum(squared_deltas)
print('loss: {}'.format(ss.run(loss, {x:[1, 2, 3, 4], y: [0, -1, -2, -3]})))