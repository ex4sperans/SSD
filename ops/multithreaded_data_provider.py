import tensorflow as tf

class MultithreadedTensorProvider():

    """ A class designed to provide tensors input in a
    separate threads. """

    def __init__(
                 self,
                 capacity,
                 sess,
                 dtypes,
                 number_of_threads=1):

        """Initialize a class to provide a tensors with input data.

        Args:
            capacity: maximum queue size. 
            sess: a tensorflow session.
            dtypes: list of data types.
        """

        self.dtypes = dtypes
        self.sess = sess
        self.number_of_threads = number_of_threads

        self.queue = tf.FIFOQueue(capacity=capacity, dtypes=dtypes)

        self.q_size = self.queue.size()

    def get_input(self):
        """ Return input tensors """
        self.batch = self.queue.dequeue()
        return self.batch

    def set_data_provider(self, data_provider):
        """ Set data provider to generate inputs.

        Args:
            data_provider: a callable to produce a tuple of inputs. All inputs
            in tuple are assumed to be numpy arrays.
        Raises:
            TypeError: if data provider is not a callable
        """

        if not callable(data_provider):
            raise TypeError('Data provider should be a callable.')
        
        data = tf.py_func(data_provider, [], self.dtypes)
        enqueue_op = self.queue.enqueue(data)
        qr = tf.train.QueueRunner(self.queue, [enqueue_op]*self.number_of_threads)
        tf.train.add_queue_runner(qr)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        


