import time
import logging
import itertools
import os
from logging.config import fileConfig
fileConfig('logging_config.ini')
log=logging.getLogger()

from argparse import ArgumentParser
import numpy as np
from models import RCNNModel
from utils import TextIterator,save_model,load_model

arg=ArgumentParser(usage='argument tips',description='Not yet')
arg.add_argument('--cell',default='gru',type=str,help='recurrent modules')
arg.add_argument('--optimizer',default='adam',type=str,help='optimizer')
arg.add_argument('--train_file',default='data/train.txt',type=str,help='training dataset')
arg.add_argument('--valid_file',default='data/valid.txt',type=str,help='validation dataset')
arg.add_argument('--test_file',default='data/test.txt',type=str,help='testing dataset')
arg.add_argument('--batch_size',default=15,type=int,help='batch size')
arg.add_argument('--mode',default='train',type=str,help='train or testing')
arg.add_argument('--nepoch',default=10,type=int,help='nepoch')
arg.add_argument('--maxlen',default=50,type=int,help='max length for q1 and q2')
arg.add_argument('--sim',default='eucdian',type=str,help='similarity metrics')

arg.add_argument('--disp_freq',default=10,type=int,help='display frequency')
arg.add_argument('--valid_freq',default=2000,type=int,help='validation frequency')
arg.add_argument('--test_freq',default=20,type=int,help='testing frequency')
arg.add_argument('--save_freq',default=2000,type=int,help='saving frequency')
arg.add_argument('--dump_freq',default=10000,type=int,help='dunping frequency')

arg.add_argument('--model_dir', default='./model/parameters_176832.65.pkl', type=str, help='model dir to dump')
arg.add_argument('--goto_line', default=10, type=int, help='goto the specific line index')

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
dump_freq=args.dump_freq
maxlen=args.maxlen
model_dir=args.model_dir
goto_line=args.goto_line
sim=args.sim
dropout=0.1
lr=0.001
n_input=200
n_hidden=200
VOCABULARY_SIZE=200000


def evaluate(data,model):
    acc_list=[]
    cost_list=[]
    for  (x,xmask),(y,ymask),label  in data:
        cost,acc = model.predict(x, xmask, y, ymask, label)
        cost_list.append(cost)
        acc_list.append(acc)
    return  np.mean(list(itertools.chain.from_iterable(cost_list))),np.mean(list(itertools.chain.from_iterable(acc_list)))



def train():
    log.info('loading dataset...')

    log.info('building models....')
    model=RCNNModel(n_input=n_input,n_vocab=VOCABULARY_SIZE,n_hidden=n_hidden,cell='gru',optimizer=optimizer,dropout=dropout,sim=sim,maxlen=maxlen,batch_size=batch_size)
    log.info('training start....')
    start=time.time()

    if os.path.isfile(model_dir):
        print 'loading checkpoint parameters....',model_dir
        model=load_model(model_dir,model)

    idx=goto_line
    for epoch in xrange(NEPOCH):
        costs=0
        acc_list=[]
        train_data = TextIterator(train_file+".train."+str(epoch), n_batch=batch_size, maxlen=maxlen)
        valid_data = TextIterator(train_file+".valid."+str(epoch), n_batch=batch_size, maxlen=maxlen)
        for (x,xmask),(y,ymask),label in train_data:
            idx+=1
            if x.shape[-1]!=batch_size:
                continue
            cost,acc=model.train(x,xmask,y,ymask,label,lr)
            #print cost
            #projected_output,cost= model.test(x, xmask, y, ymask,label)
            #print "projected_output shape:", projected_output.shape
            ##print "cnn_output shape:",cnn_output.shape
            #print "cost:",cost
            costs+=cost
            acc_list.append(acc)
            if np.isnan(cost) or np.isinf(cost):
                print 'Nan Or Inf detected!'
                return  -1
            if idx % disp_freq==0:
                log.info('epoch: %d, idx: %d cost: %.3f, Accuracy: %.3f '%(epoch,idx,costs/disp_freq,np.mean(list(itertools.chain.from_iterable(acc_list)))))
                costs=0
                error_rate_list=[]
            if idx%dump_freq==0:
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)

        loss, acc = evaluate(valid_data, model)
        log.info('cost: %.3f, Accuracy: %.3f' % (loss,acc))

    log.info("Finished. Time = " +str(time.time()-start))

def test():
    pass

if __name__=="__main__":
    if args.mode=='train':
        train()
    elif args.mode=='testing':
        test()
