import tensorflow as tf
from cohnKanade import cohnKanadeDataset
from deepnn import deepnn
import argparse
from imageData import ImageData



WIDTH   = 32
HEIGHT  = 32
NUM_CLASSES = 7
STEP_SIZE = 0.0001
BATCH_SIZE = 50
EPOCHS = 20

class TF_Graph:
    pass

expressions = {
    1:'anger',
    2:'contempt',
    3:'disgust',
    4:'fear',
    5:'happy',
    6:'sad',
    7:'surprise'
}

def setup_tf_computational_graph():
    g = TF_Graph()

    g.x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH])
    g.y_= tf.placeholder(tf.float32, [None, NUM_CLASSES])
    y_conv, keep_prob = deepnn(g.x)
    g.y_conv = y_conv
    g.keep_prob = keep_prob
    g.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=g.y_, logits=g.y_conv))
    g.train_step = tf.train.AdamOptimizer(STEP_SIZE).minimize(g.cross_entropy)
    g.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(g.y_conv, 1), tf.argmax(g.y_, 1)), tf.float32))

    return g

def train_model(train, g):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        loops = int((EPOCHS * train.total_points)//BATCH_SIZE)

        for i in range(loops):
          images, labels = train.next_batch(BATCH_SIZE)
          if i % 100 == 0:
            train_accuracy = g.accuracy.eval(feed_dict={ g.x: images, g.y_: labels, g.keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
          g.train_step.run(feed_dict={g.x: images, g.y_: labels, g.keep_prob: 0.7})
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model file saved in %s" % save_path)



def test_model(test ,g):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,"/tmp/model.ckpt")
        print('test accuracy %g' % g.accuracy.eval(feed_dict={g.x: test.images, g.y_: test.labels, g.keep_prob: 1.0}))


def predict(imagefile ,g):
    top_k_expressions = tf.nn.top_k(tf.nn.softmax(g.y_conv),k=3)
    I = ImageData(imagefile)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,"/tmp/model.ckpt")
        faces = I.preprocessed_faces
        num_faces = faces.shape[0]
        percentages, indices = sess.run(top_k_expressions,feed_dict={g.x: faces, g.keep_prob:1.0})
        percentages = percentages.tolist()
        indices = indices.tolist()
        predictions = []
        for i in range(num_faces):
            prediction = {}
            for j in range(3):
                prediction[expressions[int(indices[i][j])+1]] = percentages[i][j]
            predictions.append(prediction)
        print(predictions)
    I.showResults(predictions)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', action="store_true", default=False)
  parser.add_argument('--test', action="store_true", default=False)
  parser.add_argument('--predict', action="store_true", default=False)
  parser.add_argument('--image-path', type=str, help='New image to predict the facial expression')
  parser.add_argument('--images-packfile', type=str, default='./im.pck', help='binary file containing packaged images')
  parser.add_argument('--labels-packfile', type=str, default='./lbl.pck', help='binary file containing packaged labels')
  args = parser.parse_args()

  g = setup_tf_computational_graph()

  if args.train or args.test:
      ck = cohnKanadeDataset(args.images_packfile, args.labels_packfile)
  if args.train:
      train_model(ck.train,g)
  if args.test:
      test_model(ck.test,g)
  if args.predict:
      predict(args.image_path,g)
