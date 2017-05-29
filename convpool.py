import theano
import theano.tensor as T
import numpy as np

class ConvPool(object):
    def __init(self,rng,n_input,n_output,
               filter_shape,
               image_shape,
               pool_size=(2,2)):

        self.input=input
        fan_in=np.prod(filter_shape[1:])
        fan_out=(filter_shape[0]*np.prod(filter_shape[2:])/
                 np.prod(pool_size))

        init_W=np.asarray(rng.uniform(low=-np.sqrt(6./(fan_in+fan_out)),
                                      high=np.sqrt(6./fan_in+fan_out),
                                      size=filter_shape),
                          dtype=theano.config.floatX)
        self.W=theano.shared(value=init_W,name='W',borrow=True)

        init_b=np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=init_b,borrow=True)

        conv_out=T.nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        pool_out=T.signal.max_pool_2d(
            input=conv_out,
            ds=pool_size,
            ignore_border=True
        )

        self.activation=T.tanh(pool_out+self.b.dimshuffle('x',0,'x','x'))
        self.params=[self.W,self.b]




