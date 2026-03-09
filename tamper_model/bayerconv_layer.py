import tensorflow as tf


class BayarConv(tf.keras.layers.Layer): # type: ignore
    def __init__(self, filters=3, kernel_size=5):
        super(BayarConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        k = self.kernel_size
        
        # Trainable weights except center
        self.w = self.add_weight(
            shape=(k, k, input_shape[-1], self.filters),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x):
        k = self.kernel_size
        center = k // 2
        in_channels = tf.shape(self.w)[2]
        out_channels = tf.shape(self.w)[3]
        num_updates = in_channels * out_channels
        in_idx, out_idx = tf.meshgrid(
            tf.range(in_channels),
            tf.range(out_channels),
            indexing='ij'
        )
        center_indices = tf.stack(
            [
                tf.fill([num_updates], center),
                tf.fill([num_updates], center),
                tf.reshape(in_idx, [-1]),
                tf.reshape(out_idx, [-1])
            ],
            axis=1
        )
        
        # Create mask to zero-out center during normalization
        mask = tf.ones_like(self.w)
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=center_indices,
            updates=tf.zeros([num_updates], dtype=self.w.dtype)
        )
        
        # Apply mask (remove center weight temporarily)
        w_no_center = self.w * mask
        
        # Normalize non-center weights to sum = 1
        sum_others = tf.reduce_sum(w_no_center, axis=(0,1,2), keepdims=True)
        w_normalized = w_no_center / (sum_others + 1e-8)
        
        # Set center weight to -1
        center_kernel = tf.zeros_like(w_normalized)
        center_kernel = tf.tensor_scatter_nd_update(
            center_kernel,
            indices=center_indices,
            updates=tf.fill([num_updates], tf.cast(-1.0, w_normalized.dtype))
        )
        
        final_kernel = w_normalized + center_kernel
        
        return tf.nn.conv2d(x, final_kernel, strides=1, padding='SAME') 
    
    
    




