import tensorflow as tf
import horovod.tensorflow as hvd

def test_horovod_allgather():
    """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(config=config) as session:

        tensor = tf.ones([1])* rank
        tensor = tf.cast(tensor, dtype=tf.float32)
        gathered = hvd.allgather(tensor)
        gathered_tensor = session.run(gathered)

        while True:
            print('gathered_tensor = ', gathered_tensor)


test_horovod_allgather()
