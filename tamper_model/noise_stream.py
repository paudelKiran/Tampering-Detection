class BayarConv(tf.keras.layers.Layer):
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
        
        # Create mask to zero-out center during normalization
        mask = tf.ones_like(self.w)
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=[[center, center, i, j]
                     for i in range(self.w.shape[2])
                     for j in range(self.w.shape[3])],
            updates=tf.zeros([self.w.shape[2]*self.w.shape[3]])
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
            indices=[[center, center, i, j]
                     for i in range(w_normalized.shape[2])
                     for j in range(w_normalized.shape[3])],
            updates=tf.fill([w_normalized.shape[2]*w_normalized.shape[3]], -1.0)
        )
        
        final_kernel = w_normalized + center_kernel
        
        return tf.nn.conv2d(x, final_kernel, strides=1, padding='SAME') # type: ignore