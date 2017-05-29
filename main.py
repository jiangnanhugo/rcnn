import time
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
log=logging.getLogger()

from argparse import ArgumentParser
import numpy as np
from models import RCNNModel
from utils import TextIterator

arg=ArgumentParser(usage='argument tips',description='Not yet')
arg.add_argument('--cell',default='gru',type=str,help='recurrent modules')
arg.add_argument('--optimizer',default='adam',type=str,help='optimizer')
arg.add_argument('--train_file',default='data/train.txt',type=str,help='training dataset')
arg.add_argument('--valid_file',default='data/valid.txt',type=str,help='validation dataset')
arg.add_argument('--test_file',default='data/test.txt',type=str,help='testing dataset')
arg.add_argument('--batch_size',default=10,type=int,help='batch size')
arg.add_argument('--mode',default='train',type=str,help='train or testing')
arg.add_argument('--nepoch',default=10,type=int,help='nepoch')
arg.add_argument('--maxlen',default=30,type=int,help='max length for q1 and q2')
arg.add_argument('--disp_freq',default=20,type=int,help='display frequency')
arg.add_argument('--valid_freq',default=2000,type=int,help='validation frequency')
arg.add_argument('--test_freq',default=2000,type=int,help='testing frequency')
arg.add_argument('--save_freq',default=2000,type=int,help='saving frequency')

args=arg.parse_args()
print args

NEPOCH=args.nepoch
train_file=args.train_file
valid_file=args.valid_file
test_file=args.test_file
batch_size=args.batch_size
optimizer=args.optimizer
disp_freq=args.disp_freq
valid_freq=args.valid_freq
test_freq=args.test_freq
maxlen=args.maxlen
dropout=0.1
lr=0.001

def evaluate(data,model):
    for x,y,label in data:
        print model.test(x,y,label)


def train():
    log.info('loading dataset...')
    train_data=TextIterator(train_file,n_batch=batch_size,maxlen=maxlen)
    test_data=TextIterator(train_file,n_batch=batch_size,maxlen=maxlen)
    log.info('building models....')
    #model=RCNNModel()
    log.info('training start....')
    start=time.time()
    idx=0
    for epoch in xrange(NEPOCH):
        error=0
        for (x,xmask),(y,ymask),label in train_data:
            idx+=1
            cost=0
            print "="*20
            print x.shape
            print x
            print "="*20
            #cost=model.train(x,y,label,lr)
            error+=cost
            if np.isnan(cost) or np.isinf(cost):
                print 'Nan Or Inf detected!'
                return  -1
            if idx % disp_freq==0:
                log.info('epoch:',epoch,'idx:',idx,'cost:',error/disp_freq)
                error=0
    #test_cost = evaluate(test_data, model)
    test_cost=0
    log.info('Testing cost:', test_cost)
    log.info("Finished. Time = " +str(time.time()-start))

def test():
    pass

if __name__=="__main__":
    if args.mode=='train':
        train()
    elif args.mode=='testing':
        test()
