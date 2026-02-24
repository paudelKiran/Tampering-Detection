import tensorflow as tf
#RGB to YCbCr
def rgb_to_ycbcr(image):
    image = tf.cast(image,tf.float32)/255.0
    matrix = tf.constant([
         [0.299, -0.168736, 0.5],
        [0.587, -0.331264, -0.418688],
        [0.114, 0.5, -0.081312]
    ])

    shift = tf.constant([0., 0.5, 0.5])

    ycbcr = tf.tensordot(image,matrix,axes = 1) + shift     #axes = 1 means that we are performing the dot product along the last axis of the image tensor and the first axis of the matrix. This allows us to convert each pixel's RGB values to YCbCr values correctly.


    return ycbcr


#8*8 DCT

def block_dct(channel):
    #channel: [B,H,W]
    B = tf.shape(channel)[0] #B -> batch size; how many image being processed at a time
    H = tf.shape(channel)[1]
    W = tf.shape(channel)[2]
    
    channel = channel[:, :H - H  % 8, : W - W % 8] #cropping the image so that we can turn them to  8*8 bolcks and the  leftover block wont affect the model 
    
    patches = tf.image.extract_patches(
    image = tf.expand_dims(channel , -1),
    sizes = [1,8,8,1],
    strides = [1,8,8,1],
    rates = [1,1,1,1],
    padding = 'VALID'
    )
    
    patches = tf.reshape(patches,[-1,8,8])     #-1 automatically calculates total number of blocks
    dct = tf.signal.dct(patches, type = 2, norm = 'ortho') #row dct
    dct = tf.signal.dct(tf.transpose(dct, perm = [0,2,1]), type = 2, norm = 'ortho') #horizontal dct
    dct = tf.transpose(dct, perm = [0,2,1]) # transpose again to get back into the original shape
    
    return dct



def reconstruct_dct_volume(dct_blocks, H , W):  #input shape: [N,8,8]  OUTPUT SHAPE : [B,64,64,64]
    blocks_per_image = (H//8)  * (W//8)
    dct_blocks = tf.reshape(dct_blocks,[-1,blocks_per_image,8,8])
    dct_volume = tf.reshape(dct_blocks,[-1,H//8, W//8,64])
    return dct_volume

def simulate_quantization(dct_volume, q = 10.0):
    return tf.round(dct_volume / q)


def binary_volume_encoding(dct_volume, thresholds = (-20,-10,0,10,20)):
    binary_maps = []
    for t in thresholds:
        binary_maps.append(tf.cast(dct_volume > t, tf.float32))
    return tf.concat(binary_maps, axis = -1)     #output =  (B, H/8, W/8, 320)
    
    
    
def frequency_backbone(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x2 = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x1)
    x3 = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x2)

    return tf.keras.Model(inputs, [x1, x2, x3])
    
    
def frequency_branch_pipeline(image):

    B = tf.shape(image)[0]
    H = tf.shape(image)[1]
    W = tf.shape(image)[2]

    ycbcr = rgb_to_ycbcr(image)
    Y = ycbcr[..., 0]

    dct_blocks = block_dct(Y)
    dct_volume = reconstruct_dct_volume(dct_blocks, H, W)

    dct_volume = simulate_quantization(dct_volume)
    binary_volume = binary_volume_encoding(dct_volume)

    model = frequency_backbone(binary_volume.shape[1:])
    features = model(binary_volume)

    return features