from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    
    out = x.reshape(len(x), -1) @ w + b
    cache = (x, w, b)
    
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    
    x, w, _ = cache
    dx = (dout @ w.T).reshape(x.shape)
    dw = x.reshape(len(x), -1).T @ dout
    db = dout.sum(axis=0)

    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """

    out = np.maximum(0, x)
    cache = x
    
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    
    dx, x = None, cache
    dx = dout * (x > 0)

    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    
    loss, dx = None, None
    N = len(y) # number of samples

    P = np.exp(x - x.max(axis=1, keepdims=True)) # numerically stable exponents
    P /= P.sum(axis=1, keepdims=True)            # row-wise probabilities (softmax)

    loss = -np.log(P[range(N), y]).sum() / N     # sum cross entropies as loss

    P[range(N), y] -= 1
    dx = P / N

    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    D = x.shape[1]
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    if mode == "train":

        shape = bn_param.get('shape', x.shape)              
        axis = bn_param.get('axis', 0)                     
                
        mu = x.mean(axis=axis)        
        var = x.var(axis=axis)        
        std = np.sqrt(var + eps)   
        x_hat = (x - mu) / std     
        out = gamma * x_hat + beta 

        cache = x, mu, var, std, gamma, x_hat, shape, axis 

        if axis == 0:                                                    
            running_mean = momentum * running_mean + (1 - momentum) * mu 
            running_var = momentum * running_var + (1 - momentum) * var  

    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    
    dx, dgamma, dbeta = None, None, None
    x, mu, var, std, gamma, x_hat, shape, axis = cache

    dgamma = (x_hat * dout).sum(axis=0, keepdims=True)
    dbeta = (np.ones_like(x_hat) * dout).sum(axis=0, keepdims=True)
    dx_hat = gamma * dout
    dx1 = (std**-1) * dx_hat
    dmu = (-(std**-1) * dx_hat).sum(axis=axis, keepdims=True)
    dstd = ((-(x-mu)/(std**2)) * dx_hat).sum(axis=axis, keepdims=True)
    dvar = 0.5 * (dstd/std)
    dx2 = ((2*(x-mu))/shape[axis]) * dvar
    dx3 = (np.ones_like(x)/shape[axis]) * dmu
    dx = dx1 + dx2 + dx3

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    
    dx, dgamma, dbeta = None, None, None
    x, mu, var, std, gamma, x_hat, shape, axis = cache
    
    _, _, _, std, gamma, x_hat, shape, axis = cache
    S = lambda x: x.sum(axis=axis)
    
    dgamma = (x_hat * dout).sum(axis=0, keepdims=True)
    dbeta = (np.ones_like(x_hat) * dout).sum(axis=0, keepdims=True)
    dx = dout * gamma / (len(dout) * std)        
    dx = len(dout)*dx  - S(dx*x_hat)*x_hat - S(dx) 

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    
    out, cache = None, None
    
    N, D = x.shape
    eps = ln_param.get("eps", 1e-5)
    shape = ln_param.get('shape', (N, D))              
    axis = ln_param.get('axis', 1)

    mu = x.mean(axis=axis, keepdims=True)    
    var = x.var(axis=axis, keepdims=True)        
    std = np.sqrt(var + eps)   
    x_hat = (x - mu) / std     
    out = gamma * x_hat + beta 
                     
    cache = x, mu, var, std, gamma, x_hat, shape, axis 
    
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    
    # dgamma = (x_hat * dout).sum(axis=0, keepdims=True)
    # dbeta = (np.ones_like(x_hat) * dout).sum(axis=0, keepdims=True)
    # dx_hat = gamma * dout
    # dx1 = (std**-1) * dx_hat
    # dmu = (-(std**-1) * dx_hat).sum(axis=axis, keepdims=True)
    # dstd = ((-(x-mu)/(std**2)) * dx_hat).sum(axis=axis, keepdims=True)
    # dvar = 0.5 * (dstd/std)
    # dx2 = ((2*(x-mu))/shape[axis]) * dvar
    # dx3 = (np.ones_like(x)/shape[axis]) * dmu
    # dx = dx1 + dx2 + dx3
    
    dx, dgamma, dbeta =  batchnorm_backward(dout, cache)
        
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    
    mask = None
    out = None
    
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    
    dx = None    
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
        
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    out = None

    stride, pad = conv_param.values()
    pad_width = ((0,0), (0,0), (pad, pad), (pad, pad))
    N_W, C_W, H_W, W_W = w.shape
    w_size = C_W * H_W *  W_W
    
    #padding
    x_pad = np.pad(x, pad_width)
    
    #shaping the out
    N_X, _, H_X, W_X = x_pad.shape                                                             
    H = (H_X - H_W) // stride + 1
    W = (W_X - W_W) // stride + 1 
    out = np.zeros((N_X, N_W, H, W))
    
    for idx, img in enumerate(x_pad):
      col = 0
      X_col = np.zeros((w_size, H*W))
      for i in range(0, H_X-H_W+1, stride):
        for j in range(0, W_X-W_W+1, stride):
          filter_col = img[:, i:i+H_W, j:j+W_W].reshape(w_size, )          
          X_col[:, col] = filter_col
          col += 1
      
      W_row = w.reshape(N_W, w_size)
      conv = W_row.dot(X_col) + b.reshape(-1, 1)
      out[idx] = conv.reshape(N_W, H, W)

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    stride, pad = conv_param.values()
    pad_width = ((0,0), (0,0), (pad, pad), (pad, pad))
    w_height = w.shape[2]
    w_width = w.shape[3]
    
    x_pad = np.pad(x, pad_width)
    
    dw = np.zeros_like(w)
    dx = np.zeros_like(x_pad)
    db = np.zeros_like(b)
    
    for i in range(0, x_pad.shape[2]-w_height+1, stride):
      for j in range(0, x_pad.shape[2]-w_width+1, stride):
        img_patch = x_pad[:, :, i:i+w_height, j:j+w_width]
        do = dout[:, :, i//stride, j//stride]
        
        w_flatten = w.reshape(w.shape[0], -1)
        dx[:, :, i:i+w_height, j:j+w_width] += (do @ w_flatten).reshape(img_patch.shape)
        
        img_patches = np.tile(img_patch, (dw.shape[0], 1, 1, 1, 1))
        do_T_extended = do.T.reshape(do.shape[1], do.shape[0], 1, 1, 1)
        dw += (do_T_extended * img_patches).sum(1)
        
        db += do.sum(0).T
    
    #removing padding
    dx = dx[:, :, 1:-pad, 1:-pad]
        
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    pool_height, pool_width, stride = pool_param.values()
    N, C, H, W = x.shape
    H_A = (H - pool_height) // stride + 1
    W_A = (W - pool_width) // stride + 1
    out = np.zeros((N, C, H_A, W_A))
        
    for i in range(0, H-pool_height+1, stride):
      for j in range(0, W-pool_width+1, stride):
        pooling_block = x[:, :, i:i+pool_height, j:j+pool_width]
        out[:, :, i//stride, j//stride] = np.max(pooling_block, axis=(2, 3))
        
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    
    dx = None

    x, pool_param = cache
    N, C, H, W = dout.shape
    dx = np.zeros_like(x)
    HP, WP, S = pool_param.values()

    for i in range(H):
      for j in range(W):
        [ns, cs], h, w = np.indices((N, C)), i * S, j * S
        pooling_block = x[:, :, h:(h+HP), w:(w+WP)].reshape(N, C, -1)
        d1, d2 = np.unravel_index(np.argmax(pooling_block, 2), (HP, WP))
        dx[ns, cs, d1+h, d2+w] = dout[ns, cs, i, j]
        
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    
    out, cache = None, None
    
    N, C, H, W = x.shape
    x_reshaped = np.moveaxis(x, 1, -1).reshape(-1, C)
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = np.moveaxis(out.reshape(N, H, W, C), -1, 1)
    
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape                             
    dout = np.moveaxis(dout, 1, -1).reshape(-1, C)      
    dx, dgamma, dbeta = batchnorm_backward(dout, cache) 
    dx = np.moveaxis(dx.reshape(N, H, W, C), -1, 1)     

    return dx, dgamma, dbeta