#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:03:02 2018

@author: dufumier
"""
# Creates a CNN with the same weights as the ones used during testing phase
# and returns accuracy
#def test():
#    correct_prediction = tf.equal(y_true, y_pred)
#    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y_true, 1), \
#                                   predictions=tf.argmax(y_pred, 1))
#    return acc, acc_op, correct_prediction
#
## TESTING PHASE
#y_pred, _ = create_network(training=False)
#accuracy, accuracy_op, correct_prediction_test = test()
#
#
#def testing(data, labels, batch_size):
#    with tf.Session() as sess:
#        restore_checkpoint(sess)
#        k = 0
#        hot_predictions = np.zeros((len(labels), nb_classes), dtype=np.float32)
#        hot_label_values = pd.get_dummies(labels).values
#        while(k < len(labels)):
#            feed_dict_test = {x : data[k:k+batch_size,:,:,:], y_true: hot_label_values[k:k+batch_size,:]}
#            hot_predictions[k:k+batch_size] = sess.run(y_pred, feed_dict = feed_dict_test)
#            k += batch_size
#        predictions_tf = tf.argmax(hot_predictions, 1)
#        predictions = sess.run(predictions_tf)
#    acc = sum(predictions == labels)/len(labels)
#    return acc, predictions, hot_predictions

