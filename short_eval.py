eval_dir = 'here'
eval_interval_secs = 30
import math
import model
import task
import tensorflow as tf
import numpy as np
import datetime
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


num_examples = 100 #FIXME: WHAT DOES THIS DO ?????

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto(device_count = {'GPU': 0})
  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state('refine')
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      predictions = sess.run([top_k_op])

      # Compute precision @ 1.
      precision = predictions[0]
      print('%s: precision @ 1 = %.3f' % (100.0, precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

from dataset import csv_inputs
import time
def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, depths, invalid_depths = csv_inputs('test.csv', 8)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    keep_conv = tf.placeholder(tf.float32)
    keep_hidden = tf.placeholder(tf.float32)
    coarse = model.inference(images, trainable=False)
    logits = model.inference_refine(images, coarse, .5, keep_hidden)
    tf.summary.image('images2', logits*255.0, max_outputs=3)
    # Calculate predictions.
    top_k_op = model.loss(logits, depths, invalid_depths)
    
    # Session
    with tf.Session(config=tf.ConfigProto()) as sess:

        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(eval_dir, g)
        while True:
          eval_once(saver, summary_writer, top_k_op, summary_op)
          time.sleep(eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  with tf.device('/cpu:0'):
    evaluate()


if __name__ == '__main__':
  tf.app.run()

