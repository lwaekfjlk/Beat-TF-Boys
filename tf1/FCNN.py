import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=False)

def neural_net(x_dict):
    num_classes = 10
    print(x_dict)
    x = x_dict['images']
    # from (a,b,c) ---> (a,b,256)
    layers_1 = tf.layers.dense(x, 256)
    # from (a,b,256) ---> (a,b,256) 
    layers_2 = tf.layers.dense(layers_1, 256)
    out_layer = tf.layers.dense(layers_2, num_classes)
    return out_layer

def model_fn(features, labels, mode):
    # model
    # feature input
    # pred classes and pred probs output
    logits = neural_net(features)

    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # PREDICtION fixed part
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=out)

    # LOSS & OPTIMIZER fixed part
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    acc = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op, eval_metric_ops={'accuracy':acc})

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images},y=mnist.train.labels,batch_size=128, shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images},y=mnist.test.labels,batch_size=128, shuffle=False)

model = tf.estimator.Estimator(model_fn)
model.train(train_input_fn, steps=1000)
res_dict = model.evaluate(test_input_fn)

print("testing accuracy : ", res_dict['accuracy'])
