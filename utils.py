import numpy as np
import cPickle as pickle

def save_model(f,model):
    ps={}
    for p in model.params:
        ps[p.name]=p.get_value()
    pickle.dump(ps,open(f,'wb'))

def load_model(f,model):
    ps=pickle.load(open(f,'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model

class TextIterator(object):
    def __init__(self,source,n_batch,maxlen=None):

        self.source=open(source,'r')
        self.n_batch=n_batch
        self.maxlen=maxlen

        self.end_of_data=False

    def __iter__(self):
        return self


    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        batch_q1=[]
        batch_q2 = []
        batch_label = []
        try:
            while True:
                s=self.source.readline()
                if s=="":
                    raise IOError
                s=s.split('\t')
                if len(s)!=3:
                    raise IOError

                q1,q2,label=s
                q1=[int(w) for w in q1.split(' ')]
                q2 = [int(w) for w in q2.split(' ')]
                label=int(label)

                if self.maxlen and (len(q1)>self.maxlen or len(q2)>self.maxlen):
                    continue
                batch_q1.append(q1)
                batch_q2.append(q2)
                batch_label.append(label)
                if len(batch_q1)>=self.n_batch:
                    break
        except IOError:
            self.end_of_data=True

        if len(batch_q1)<=0 or len(batch_q2)<=0:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        return prepare_data(batch_q1),prepare_data(batch_q2),batch_label

def prepare_data(seqs_x):
    lengths_x=[len(s)-1 for s in seqs_x]
    n_samples=len(seqs_x)
    maxlen_x=np.max(lengths_x)

    x=np.zeros((maxlen_x,n_samples)).astype('int32')
    x_mask=np.zeros((maxlen_x,n_samples)).astype('float32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x[:-1]
        x_mask[:lengths_x[idx],idx]=1

    return x,x_mask


