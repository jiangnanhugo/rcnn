from gru import GRU
from lstm import LSTM
from convpool import ConvPool
from updates import *
import theano
import numpy as np
import theano.tensor as T
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
log=logging.getLogger()

class RCNNModel(object):
    def __init__(self,n_input,n_vocab,n_hidden,cell='gru',optimizer='adam',dropout=0.1,sim='eucdian'):

        self.x=T.imatrix('batched input query one')
        self.y=T.imatrix('batched input query two')

        self.n_input=n_input
        self.n_hidden=n_hidden

        self.xmask = T.matrix('batched masked query one')
        self.ymask = T.matrix('batched masked query two')
        self.label=T.imatrix('batched similarity label')

        self.cell=cell
        self.optimizer=optimizer
        self.dropout=dropout
        self.sim=sim
        self.is_train=T.iscalar('is_train')

        init_Embd = np.asarray(
            np.random.uniform(low=-np.sqrt(6. / (n_input+n_vocab)), high=np.sqrt(1. / (n_input+n_vocab)), size=(n_vocab, n_input)),
            dtype=theano.config.floatX)

        self.E = theano.shared(value=init_Embd, name='word_embedding', borrow=True)

        self.rng=RandomStreams(1234)
        self.build()

    def build(self):
        log.info('building rnn cell....')
        if self.cell=='gru':
            recurent_x=GRU(self.rng,
                           self.n_input,
                           self.n_hidden,
                           self.x,self.E,self.xmask,
                           self.is_train,self.dropout)

            recurent_y = GRU(self.rng,
                             self.n_input,
                             self.n_hidden,
                             self.x,self.E, self.ymask,
                             self.is_train, self.dropout)
        elif self.cell=='lstm':
            recurent_x=LSTM(self.rng,
                            self.n_input,
                            self.n_hidden,
                            self.y,self.E,self.xmask,
                            self.is_train,self.dropout)

            recurent_y = LSTM(self.rng,
                              self.n_input,
                              self.n_hidden,
                              self.y,self.E, self.ymask,
                              self.is_train, self.dropout)
        log.info('build the sim matrix....')
        if self.sim=='eucdian':
            sim_layer=T.dot(recurent_x.activation,recurent_y.activation)

        log.info('building convolution pooling layer....')
        conv_pool_layer=ConvPool(sim_layer)
        cost=T.nnet.binary_crossentropy(conv_pool_layer.activation,self.label)

        self.params=[self.E,]
        self.params+=recurent_x.params
        self.params+=recurent_y.params
        self.params+=conv_pool_layer.params

        lr=T.scalar('lr')
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]

        if self.optimizer=='sgd':
            updates=sgd(self.params,gparams,lr)
        elif self.optimizer=='adam':
            updates=adam(self.params,gparams,lr)
        elif self.optimizer=='rmsprop':
            updates=rmsprop(self.params,gparams,lr)

        self.train=theano.function(inputs=[self.x,self.y,self.label,self.lr],
                                   outputs=cost,
                                   updates=updates,
                                   givens={self.is_train:np.cast['int32'](1)})






