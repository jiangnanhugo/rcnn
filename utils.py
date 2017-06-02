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

    def goto_line(self, line_index):
        for _ in range(line_index):
            self.source.readline()

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        batch_p=[]
        batch_q = []
        batch_label = []
        try:
            while True:
                s=self.source.readline()
                if s=="":
                    raise IOError
                s=s.split('\t')
                if len(s)!=3:
                    raise IOError

                p,q,label=s
                p=[int(w) for w in p.split(' ') if len(w)>0]
                q = [int(w) for w in q.split(' ') if len(w)>0]
                label=int(label)
                if self.maxlen and (len(p)>self.maxlen or len(q)>self.maxlen):
                    continue
                batch_p.append(p)
                batch_q.append(q)
                batch_label.append(label)
                if len(batch_p)>=self.n_batch:
                    break
        except IOError:
            self.end_of_data=True

        if len(batch_p)<=0 or len(batch_q)<=0:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        return prepare_data(batch_p,self.maxlen),prepare_data(batch_q,self.maxlen),np.asarray(batch_label,dtype='int32')

def prepare_data(seqs_x,maxlen=10):
    lengths_x=[len(s)-1 for s in seqs_x]
    n_samples=len(seqs_x)
    if maxlen>0:
        maxlen_x=maxlen

    x=np.zeros((maxlen_x,n_samples)).astype('int32')
    x_mask=np.zeros((maxlen_x,n_samples)).astype('float32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x[:-1]
        x_mask[:lengths_x[idx],idx]=1

    return x,x_mask


