# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: model.py
   create time: 2017年06月25日 星期日 10时47分48秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
from poem_generator_web_project.config import *

class MODEL:
    """model class"""
    def __init__(self, trainData):
        self.trainData = trainData

    def buildModel(self, wordNum, gtX, hidden_units = 128, layers = 2):
        """build rnn"""
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE): #embedding
            embedding = tf.get_variable("embedding", [wordNum, hidden_units], dtype = tf.float32)
            inputbatch = tf.nn.embedding_lookup(embedding, gtX)

        # basicCell = []
        # for i in range(layers):
        #     basicCell.append(tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple=True))
        # stackCell = tf.contrib.rnn.MultiRNNCell(basicCell, state_is_tuple = True)

        basicCell = tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple = True)
        stackCell = tf.contrib.rnn.MultiRNNCell([basicCell] * layers)

        initState = stackCell.zero_state(np.shape(gtX)[0], tf.float32)
        outputs, finalState = tf.nn.dynamic_rnn(stackCell, inputbatch, initial_state = initState)
        outputs = tf.reshape(outputs, [-1, hidden_units])

        with tf.variable_scope("softmax"):
            w = tf.get_variable("w", [hidden_units, wordNum])
            b = tf.get_variable("b", [wordNum])
            logits = tf.matmul(outputs, w) + b

        probs = tf.nn.softmax(logits)
        return logits, probs, stackCell, initState, finalState

    def probsToWord(self, weights, words):
        """probs to word"""
        prefixSum = np.cumsum(weights) #prefix sum
        ratio = np.random.rand(1)
        index = np.searchsorted(prefixSum, ratio * prefixSum[-1]) # large margin has high possibility to be sampled
        return words[index[0]]

    def test(self, checkpointsPath):
        """write regular poem"""
        logging.info("genrating...")
        gtX = tf.placeholder(tf.int32, shape=[1, None])  # input
        logits, probs, stackCell, initState, finalState = self.buildModel(self.trainData.wordNum, gtX)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            # if have checkPoint, restore checkPoint
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                logging.info("restored %s" % checkPoint.model_checkpoint_path)
            else:
                logging.info("no checkpoint found!")
                exit(1)

            poems = []
            for i in range(generateNum):
                state = sess.run(stackCell.zero_state(1, tf.float32))
                x = np.array([[self.trainData.wordToID['[']]]) # init start sign
                probs1, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
                word = self.probsToWord(probs1, self.trainData.words)
                poem = ''
                sentenceNum = 0
                while word not in [' ', ']']:
                    poem += word
                    if word in ['。', '？', '！', '，']:
                        sentenceNum += 1
                        if sentenceNum % 2 == 0:
                            poem += '\n'
                    x = np.array([[self.trainData.wordToID[word]]])
                    #logging.info(word)
                    probs2, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
                    word = self.probsToWord(probs2, self.trainData.words)
                logging.info(poem)
                poems.append(poem)
            return poems

    def testHead(self, checkpointsPath, characters):
        """write head poem"""
        logging.info("genrating...")
        gtX = tf.placeholder(tf.int32, shape=[1, None])  # input
        logits, probs, stackCell, initState, finalState = self.buildModel(self.trainData.wordNum, gtX)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            # if have checkPoint, restore checkPoint
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                logging.info("restored %s" % checkPoint.model_checkpoint_path)
            else:
                logging.info("no checkpoint found!")
                exit(1)
            flag = 1
            endSign = {-1: "，", 1: "。"}
            poem = ''
            state = sess.run(stackCell.zero_state(1, tf.float32))
            x = np.array([[self.trainData.wordToID['[']]])
            probs1, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
            for word in characters:
                if self.trainData.wordToID.get(word) == None:
                    logging.info("胖虎不认识这个字，你真是文化人！关键字：" + word)
                    exit(0)
                flag = -flag
                while word not in [']', '，', '。', ' ', '？', '！']:
                    poem += word
                    x = np.array([[self.trainData.wordToID[word]]])
                    probs2, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
                    word = self.probsToWord(probs2, self.trainData.words)

                poem += endSign[flag]
                # keep the context, state must be updated
                if endSign[flag] == '。':
                    probs2, state = sess.run([probs, finalState],
                                             feed_dict={gtX: np.array([[self.trainData.wordToID["。"]]]), initState: state})
                    poem += '\n'
                else:
                    probs2, state = sess.run([probs, finalState],
                                             feed_dict={gtX: np.array([[self.trainData.wordToID["，"]]]), initState: state})

            logging.info(characters)
            logging.info(poem)
            return poem

    def testTail(self, checkpointsPath, characters):
        """write head poem"""
        logging.info("genrating...")
        gtX = tf.placeholder(tf.int32, shape=[1, None])  # input
        logits, probs, stackCell, initState, finalState = self.buildModel(self.trainData.wordNum, gtX)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            # if have checkPoint, restore checkPoint
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                logging.info("restored %s" % checkPoint.model_checkpoint_path)
            else:
                logging.info("no checkpoint found!")
                exit(1)
            flag = 1
            endSign = {-1: "，", 1: "。"}
            poem = ''
            state = sess.run(stackCell.zero_state(1, tf.float32))
            x = np.array([[self.trainData.wordToID['[']]])
            probs1, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
            for word in characters:
                sentence = ''
                if self.trainData.wordToID.get(word) == None:
                    logging.info("胖虎不认识这个字，你真是文化人！")
                    exit(0)
                    tf.reset_default_graph()
                flag = -flag
                while word not in [']', '，', '。', ' ', '？', '！']:
                    sentence = word + sentence
                    x = np.array([[self.trainData.wordToID[word]]])
                    probs2, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
                    word = self.probsToWord(probs2, self.trainData.words)

                sentence += endSign[flag]
                poem += sentence
                # keep the context, state must be updated
                if endSign[flag] == '。':
                    probs2, state = sess.run([probs, finalState],
                                             feed_dict={gtX: np.array([[self.trainData.wordToID["。"]]]), initState: state})
                    poem += '\n'
                else:
                    probs2, state = sess.run([probs, finalState],
                                             feed_dict={gtX: np.array([[self.trainData.wordToID["，"]]]), initState: state})

            logging.info(characters)
            logging.info(poem)
            return poem