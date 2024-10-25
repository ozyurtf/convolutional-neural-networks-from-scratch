import torch
import numpy as np
from utils import Linear, ReLU, softmax_loss

class Conv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases of shape (F)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """

        stride = conv_param['stride']
        pad = conv_param['pad']

        N, C, H, W = x.shape
        F, C, HH, WW = w.shape

        H_padded = H + 2 * pad
        W_padded = W + 2 * pad

        x_padded = torch.zeros((N, C, H_padded, W_padded), dtype=x.dtype, device=x.device)
        x_padded[:, :, pad:pad + H, pad:pad + W] = x
        
        H_out = 1 + (H_padded - HH) // stride
        W_out = 1 + (W_padded - WW) // stride        
         
        out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device=x.device)

        for f in range(F): 
            filter = w[f, :, :, :]
            bias = b[f]
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
              
                    w_start = j * stride
                    w_end = w_start + WW

                    x_padded_slide = x_padded[:, :, h_start:h_end, w_start:w_end] 
                    out[:, f, i, j] = torch.sum(x_padded_slide * filter, dim = [1,2,3]) + bias

        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """

        x, w, b, conv_param = cache
        stride = conv_param['stride']
        pad = conv_param['pad']

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        H_padded = H + 2 * pad
        W_padded = W + 2 * pad

        x_padded = torch.zeros((N, C, H_padded, W_padded), dtype=x.dtype, device=x.device)
        x_padded[:, :, pad:pad + H, pad:pad + W] = x

        H_out = 1 + (H_padded - HH) // stride
        W_out = 1 + (W_padded - WW) // stride

        dx_padded = torch.zeros_like(x_padded, device=x.device)
        dw = torch.zeros_like(w, device=x.device)
        db = dout.sum(dim=(0, 2, 3)) 

        for n in range(N):
            for i in range(H_out):
               for j in range(W_out):
                  h_start = i * stride
                  h_end = h_start + HH
                  w_start = j * stride
                  w_end = w_start + WW
                  x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]

                  dw += dout[n, :, i, j].view(F, 1, 1, 1) * x_slice
                  dx_padded[n, :, h_start:h_end, w_start:w_end] += (w * dout[n, :, i, j].view(F, 1, 1, 1)).sum(dim=0)

        dx = dx_padded[:, :, pad:pad + H, pad:pad + W]

        return dx, dw, db


class MaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """

        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        
        H_out = 1 + (H - pool_height) // stride
        W_out = 1 + (W - pool_width) // stride
        
        out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width
                out[:, :, i, j] = x[:, :, h_start:h_end, w_start:w_end].max(dim=2)[0].max(dim=2)[0]
        
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """

        x, pool_param = cache
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        
        H_out = 1 + (H - pool_height) // stride
        W_out = 1 + (W - pool_width) // stride
        
        dx = torch.zeros_like(x)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width
                
                x_pool = x[:, :, h_start:h_end, w_start:w_end]
                max_vals = x_pool.max(dim=3)[0].max(dim=2)[0]
                mask = (x_pool == max_vals[:, :, None, None])
                dx[:, :, h_start:h_end, w_start:w_end] += mask * dout[:, :, i:i+1, j:j+1]
                
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C, H, W  = input_dims
        HH = filter_size 
        WW = filter_size
        F = num_filters

        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        pad = conv_param['pad']
        conv_stride = conv_param['stride']

        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        pool_stride = pool_param['stride']

        H_padded = H + 2 * pad
        W_padded = W + 2 * pad        

        H_conv_out = 1 + (H_padded - HH) // conv_stride
        W_conv_out = 1 + (W_padded - WW) // conv_stride      

        H_maxpool_out = 1 + (H_conv_out - pool_height) // pool_stride
        W_maxpool_out = 1 + (W_conv_out - pool_width) // pool_stride     
        
        # Conv - ReLU - 2x2 Max Pool - Linear - ReLu - Linear - Softmax Loss
        
            # Conv (Stride: 1, Pad: Filter Size - 1 //2)
                # Input  (x)  : N, C, H, W
                # Filter (w1) : F, C, HH, WW 
                # Bias   (b1) : F, 1
                # Output      : N, F, H_conv_out, W_conv_out    
                
            # ReLU 
                # Input  (x) : N, F, H_conv_out, W_conv_out
                # Output     : N, F, H_conv_out, W_conv_out

            # MaxPool (Pool Height: 2, Pool Width: 2)
                # Input  (x) : N, F, H_conv_out, W_conv_out
                # Output     : N, F, H_maxpool_out, W_maxpool_out

            # Flatten
                # Input  (x) : N, F, H_maxpool_out, W_maxpool_out
                # Output     : N, F * H_maxpool_out * W_maxpool_out

            # Linear - 1 (Hidden Dimension = hidden_dim): 
                # Input (x)  : N, F * H_maxpool_out * W_maxpool_out
                # Weight(w2) : F * H_maxpool_out * W_maxpool_out, hidden_dim
                # Bias  (b2) : F * H_maxpool_out * W_maxpool_out, 1
                # Output     : N, hidden_dim

            # ReLU: 
                # Input  (x) : N, hidden_dim
                # Output     : N, hidden_dim
            
            # Linear - 2 (Hidden Dimension = num_classes): 
                # Input  (x) : N, hidden_dim
                # Weight(w3):  hidden_dim, num_classes
                # Bias  (b3):  hidden_dim, 1
                # Output     : N, num_classes

            # Softmax Loss: 
                # Input  (x) : N, num_classes
                # Output     : loss     
                        
        # Conv
        self.params["W1"] = torch.normal(mean = 0.0, 
                                         std = weight_scale, 
                                         size = (F, C, HH, WW), 
                                         dtype = dtype, 
                                         device = device)
          
        self.params["b1"] = torch.zeros(F, dtype = dtype, 
                                        device = device)

        # Linear - 1
        self.params["W2"] = torch.normal(mean = 0.0, 
                                         std = weight_scale, 
                                         size = (F * H_maxpool_out * W_maxpool_out, hidden_dim), 
                                         dtype = dtype, 
                                         device = device)
        
        self.params["b2"] = torch.zeros(hidden_dim, 
                                        dtype = dtype, 
                                        device = device)

        # Linear - 2
        self.params["W3"] = torch.normal(mean = 0.0, 
                                         std  = weight_scale, 
                                         size = (hidden_dim, num_classes), 
                                         dtype = dtype, 
                                         device = device)
        
        self.params["b3"] = torch.zeros(size = (1, num_classes), 
                                        dtype = dtype, 
                                        device = device)

        self.conv = Conv()
        self.relu1 = ReLU()
        self.max_pool = MaxPool()
        self.linear1 = Linear()
        self.relu2 = ReLU()
        self.linear2 = Linear()


    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        N = X.shape[0]
        
        X, conv_cache = self.conv.forward(X, W1, b1, conv_param)
        X, relu1_cache = self.relu1.forward(X)
        X, maxpool_cache = self.max_pool.forward(X, pool_param)
        X, linear1_cache = self.linear1.forward(X, W2, b2)
        X, relu2_cache = self.relu2.forward(X)
        scores, linear2_cache = self.linear2.forward(X, W3, b3)
        
        if y is None:
            return scores

        grads = {}

        loss, dout = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1**2) + torch.sum(W2**2) + torch.sum(W3**2))
        
        # dw3 and dw2 represent how the loss changes 
        # with respect to the weights of their respective layers.
        # These gradients are specific to their layers. 
        # They don't represent how the loss changes with respect to the layer's input.
        # The previous layer needs to know how its output (which is the next layer's input)
        # affects the loss, which is what dx represents.
        # That's why dx3 is ised in self.relu2.backward(.) instead of dw3 and
        # dx2 is used in self.max_pool.backward(.) instead of dw2.

        dx3, dw3, db3 = self.linear2.backward(dout, linear2_cache)
        dx_relu2 = self.relu2.backward(dx3, relu2_cache)

        dx2, dw2, db2 = self.linear1.backward(dx_relu2, linear1_cache)
        dx_maxpool = self.max_pool.backward(dx2, maxpool_cache)

        dx_relu1 = self.relu1.backward(dx_maxpool, relu1_cache)
        dx1, dw1, db1 = self.conv.backward(dx_relu1, conv_cache)

        grads["W1"] = dw1
        grads["b1"] = db1

        grads["W2"] = dw2
        grads["b2"] = db2

        grads["W3"] = dw3
        grads["b3"] = db3  

        return loss, grads

class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dims=(3, 32, 32),
               num_filters=[8, 8, 8, 8, 8],
               max_pools=[0, 1, 2, 3, 4],
               batchnorm=False,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               weight_initializer=None,
               dtype=torch.float, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
  
    if device == 'cuda':
      device = 'cuda:0'
    
    self.layers = {}
    C, H, W = input_dims
    
    HH = 3 
    WW = 3

    conv_param = {'stride': 1, 'pad': (HH - 1) // 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    conv_pad = conv_param["pad"]
    conv_stride = conv_param["stride"]
    
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    pool_stride = pool_param["stride"]

    for l in range(self.num_layers-1): 
        F = num_filters[l]

        H_padded = H + 2 * conv_pad
        W_padded = W + 2 * conv_pad        

        H_conv_out = 1 + (H_padded - HH) // conv_stride
        W_conv_out = 1 + (W_padded - WW) // conv_stride     

        if l in self.max_pools:
            H_final = 1 + (H_conv_out - pool_height) // pool_stride
            W_final = 1 + (W_conv_out - pool_width) // pool_stride  
        else: 
            H_final = H_conv_out
            W_final = W_conv_out 
          
        # Conv (Stride: 1, Pad: 1)
            # Input  (x)  : N, C, H, W
            # Filter (w1) : F, C, HH, WW 
            # Bias   (b1) : F, 1
            # Output      : N, F, H_conv_out, W_conv_out    
            
        # ReLU 
            # Input  (x) : N, F, H_conv_out, W_conv_out
            # Output     : N, F, H_conv_out, W_conv_out

        # MaxPool (Pool Height: 2, Pool Width: 2)
            # Input  (x) : N, F, H_conv_out, W_conv_out
            # Output     : N, F, H_maxpool_out, W_maxpool_out        

        self.layers[f"conv{l}"] = Conv()

        if self.batchnorm:
            self.layers[f"spatial_batchnorm{l}"] = SpatialBatchNorm()
            
            self.params[f"gamma{l}"] = torch.full((num_filters[l],), 1., 
                                                   dtype=dtype, 
                                                   device=device) 
                                                   
            self.params[f"beta{l}"] = torch.full((num_filters[l],), 0., 
                                                  dtype=dtype, 
                                                  device=device) 

        self.layers[f"relu{l}"] = ReLU()

        if l in self.max_pools:
            self.layers[f"maxpool{l}"] = MaxPool()

        if weight_scale == "kaiming": 
            self.params[f"W{l}"] = kaiming_initializer(Din = C, 
                                                      Dout = F, 
                                                      K = HH, 
                                                      dtype = dtype,
                                                      device = device)

        else:   
            self.params[f"W{l}"] = torch.normal(mean = 0.0, 
                                                std  = weight_scale, 
                                                size = (F, C, HH, WW), 
                                                dtype = dtype, 
                                                device = device)
        
        self.params[f"b{l}"] = torch.zeros(F, 
                                           dtype = dtype, 
                                           device = device)
        H = H_final
        W = W_final
        C = F

    self.layers["linear"] = Linear()

    if weight_scale == "kaiming": 

        self.params[f"W{self.num_layers-1}"] = kaiming_initializer(Din = num_classes, 
                                                                  Dout = F * H_final * W_final, 
                                                                  device = device, 
                                                                  dtype = dtype)

    else:
        self.params[f"W{self.num_layers-1}"] = torch.normal(mean = 0.0, 
                                                            std = weight_scale,
                                                            size = (F * H_final * W_final, num_classes), 
                                                            dtype = dtype, 
                                                            device = device)
    
    self.params[f"b{self.num_layers-1}"] = torch.zeros(size = (1, num_classes), 
                                                       dtype = dtype, 
                                                       device = device)
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
      
    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))


  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
    scores = None

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    self.cache = {}

    for l in range(self.num_layers-1): 
        conv = self.layers[f"conv{l}"]
        W = self.params[f'W{l}']
        b = self.params[f'b{l}']
        X, self.cache[f"conv_cache{l}"] = conv.forward(X, W, b, conv_param)

        if self.batchnorm: 
            spatial_batchnorm = self.layers[f"spatial_batchnorm{l}"]
            gamma = self.params[f"gamma{l}"]
            beta = self.params[f"beta{l}"]       

            X, self.cache[f"batchnorm_cache{l}"] = spatial_batchnorm.forward(X, gamma, beta, 
                                                                             self.bn_params[l])

        relu = self.layers[f"relu{l}"] 
        X, self.cache[f"relu_cache{l}"] = relu.forward(X)

        if l in self.max_pools:
            maxpool = self.layers[f"maxpool{l}"]
            X, self.cache[f"maxpool_cache{l}"] = maxpool.forward(X, pool_param)             

    W = self.params[f'W{self.num_layers-1}']
    b = self.params[f'b{self.num_layers-1}']

    linear = self.layers["linear"] 
    scores, self.cache["linear"] = linear.forward(X, W, b)

    if y is None:
      return scores

    grads = {}
    loss, dout = softmax_loss(scores, y)

    for l in range(self.num_layers): 
        loss += self.reg * torch.sum(self.params[f"W{l}"]**2)
    
    # dw3 and dw2 represent how the loss changes 
    # with respect to the weights of their respective layers.
    # These gradients are specific to their layers. 
    # They don't represent how the loss changes with respect to the layer's input.
    # The previous layer needs to know how its output (which is the next layer's input)
    # affects the loss, which is what dx represents.
    # That's why dx3 should be ised in self.relu2.backward(.) instead of dw3 and
    # dx2 should be used in self.max_pool.backward(.) instead of dw2.

    dx, dw, db = self.layers["linear"].backward(dout, self.cache["linear"])

    grads[f"W{self.num_layers-1}"] = dw + 2 * self.reg * self.params[f"W{self.num_layers-1}"]
    grads[f"b{self.num_layers-1}"] = db

    for l in range(self.num_layers-2, -1, -1): 
        if l in self.max_pools:
            dx = self.layers[f"maxpool{l}"].backward(dx, self.cache[f"maxpool_cache{l}"])

        dx = self.layers[f"relu{l}"].backward(dx, self.cache[f"relu_cache{l}"])

        if self.batchnorm: 
            _, _, _, x_normalized, _, _, _, _ = self.cache[f"batchnorm_cache{l}"]
            grads[f"gamma{l}"] = torch.sum(dx * x_normalized, dim = (0,2,3))
            grads[f"beta{l}"] = torch.sum(dx, dim = (0,2,3))
            dx, dgamma, dbeta = self.layers[f"spatial_batchnorm{l}"].backward(dx, self.cache[f"batchnorm_cache{l}"])

        dx, dw, db = self.layers[f"conv{l}"].backward(dx, self.cache[f"conv_cache{l}"])

        grads[f"W{l}"] = dw + 2 * self.reg * self.params[f'W{l}']
        grads[f"b{l}"] = db
  
    return loss, grads


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None

  if K is None:
    fan_in = Din
    std = (gain / fan_in) ** 0.5 
    weight = torch.randn(Dout, Din, device=device, dtype=dtype) * std
  else:
    fan_in = Din * K * K
    std = (gain / fan_in) ** 0.5
    weight = torch.randn(Dout, Din, K, K, device=device, dtype=dtype) * std

  return weight

class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

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
    they do not require an additional estimation step; the PyTorch
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
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    out, cache = None, None
    if mode == 'train':
 
      sample_mean = torch.mean(x, dim=0)  
      sample_var = torch.var(x, dim=0, unbiased=False)  
      sample_std = torch.sqrt(sample_var + eps)

      x_normalized = (x - sample_mean) / sample_std
      out = gamma*x_normalized + beta

      running_mean = momentum * running_mean + (1 - momentum) * sample_mean
      running_var = momentum * running_var + (1 - momentum) * sample_var          

      cache = (x, sample_mean, sample_var, x_normalized, gamma, beta, eps, mode)
      
    elif mode == 'test':
      
      running_std = torch.sqrt(running_var + eps)
      x_normalized = (x - running_mean) / running_std
      out = gamma*x_normalized + beta

      cache = (x, running_mean, running_var, x_normalized, gamma, beta, eps, mode)
      
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for batch normalization.

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
   
    x, mean, var, x_normalized, gamma, beta, eps, mode = cache
    N = x.shape[0]

    dx_norm = dout * gamma
    dgamma = (dout * x_normalized).sum(dim=0)
    dbeta = dout.sum(dim=0)

    inv_std = 1 / (var + eps).sqrt()

    if mode == 'train': 
        dx = (1 / N) * inv_std * (N * dx_norm - dx_norm.sum(dim=0) - x_normalized * (dx_norm * x_normalized).sum(dim=0))

    elif mode == 'test':
        dx = inv_std * dx_norm
    
    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """

    x, mean, var, x_normalized, gamma, beta, eps, mode = cache 
    N = x.shape[0]

    d_xnorm = dout * gamma
    dgamma = torch.sum(dout * x_normalized, dim = 0)
    dbeta = torch.sum(dout, dim = 0)    
    
    if mode == 'train':
        d_var = torch.sum(d_xnorm * (x - mean) * -0.5 * (var + eps)**(-3/2), dim=0)
        d_mean = torch.sum(d_xnorm * (-1 / torch.sqrt(var + eps)), dim=0) + d_var * (torch.sum(-2*x + 2*mean, dim = 0)/N)

    elif mode == 'test': 
        # The running mean and variance are updated during the training phase 
        # but they are fixed during the test phase. 
        # We use the running mean and variance for only inference.
        # That's why we don't include them 
        # as part of the backpropagation of the test phase.
        d_var = 0 
        d_mean = 0

    dx = d_xnorm * (1/torch.sqrt(var + eps)) + (d_var * (2*x - 2*mean)/N) + d_mean/N    
    return dx, dgamma, dbeta

class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

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
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, C, H, W = x.shape

    running_mean = bn_param.get('running_mean', torch.zeros(C, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(C, dtype=x.dtype, device=x.device))

    sample_mean = torch.zeros(size = (C,1), dtype = x.dtype, device = x.device)
    sample_var = torch.zeros(size = (C,1), dtype = x.dtype, device = x.device)

    x_normalized = torch.zeros_like(x)
    x_scaled = torch.zeros_like(x)

    out = torch.zeros_like(x)

    if mode == 'train': 
        for c in range(C): 
            x_channel = x[:,c,:,:]      

            sample_mean[c] = torch.mean(x_channel)
            sample_var[c] = torch.var(x_channel, unbiased=False)
            sample_std = torch.sqrt(sample_var[c] + eps)             
            
            x_normalized[:,c,:,:] = (x_channel - sample_mean[c]) / sample_std
            
            x_scaled[:,c,:,:] = gamma[c]*x_normalized[:,c,:,:] + beta[c]            
            
            running_mean[c] = momentum * running_mean[c] + (1 - momentum) * sample_mean[c]
            running_var[c] = momentum * running_var[c] + (1 - momentum) * sample_var[c]    

            out[:,c,:,:] = x_scaled[:,c,:,:]

        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
              
        cache = (x, sample_mean, sample_var, x_normalized, gamma, beta, eps, mode)   
      
    elif mode == 'test':
        for c in range(C): 
            running_std = torch.sqrt(running_var[c] + eps)          
            x_channel = x[:, c, :, :]
            x_normalized[:,c,:,:] = (x_channel - running_mean[c]) / running_std
            x_scaled[:,c,:,:] = gamma[c] * x_normalized[:,c,:,:] + beta[c]
            out[:, c, :, :] = x_scaled[:,c,:,:]

            cache = (x, running_mean, running_var, x_normalized, gamma, beta, eps, mode)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
  
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    
    x, sample_mean, sample_var, x_normalized, gamma, beta, eps, mode = cache
    N, C, H, W = x.shape

    dx = torch.zeros_like(x) 
    d_xnorm = torch.zeros_like(x_normalized) 
    dgamma = torch.zeros_like(gamma) 
    dbeta = torch.zeros_like(beta) 
    d_var = torch.zeros_like(sample_var)
    d_mean = torch.zeros_like(sample_mean)

    for c in range(C): 
      d_xnorm[:,c,:,:] += dout[:,c,:,:] * gamma[c] # N, H, W
      dgamma[c] = torch.sum(dout[:,c,:,:] * x_normalized[:,c,:,:])
      dbeta[c] = torch.sum(dout[:,c,:,:])
    
      if mode == 'train':
          d_var[c] = torch.sum(d_xnorm[:,c,:,:] * (x[:,c,:,:] - sample_mean[c]) * -0.5 * (sample_var[c] + eps)**(-3/2)) 
          d_mean[c] = torch.sum(d_xnorm[:,c,:,:] * (-1 / torch.sqrt(sample_var[c] + eps))) + d_var[c] * (torch.sum(-2*x[:,c,:,:] + 2*sample_mean[c])/(N*H*W))

      elif mode == 'test': 
          d_var[c] = 0 
          d_mean[c] = 0

      dx[:,c,:,:] = d_xnorm[:,c,:,:] * (1/torch.sqrt(sample_var[c] + eps)) + (d_var[c] * (2*x[:,c,:,:] - 2*sample_mean[c])/(N*H*W)) + d_mean[c]/(N*H*W)
    
    return dx, dgamma, dbeta