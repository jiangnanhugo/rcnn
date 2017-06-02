import numpy as np
from random import shuffle
import operator
import csv
import cPickle as pickle
from collections import defaultdict


VOCABULARY_SIZE=200000

def build_dataset(train_filepath,test_filepath):
    vocab=defaultdict(int)
    reader=csv.reader(open(train_filepath))
    next(reader)
    for id, qid1,qid2,q1,q2,dup in reader:
        q1=q1.split()
        for w in q1:
            if len(w)==0: continue
            vocab[w]+=1
        q2 = q2.split()
        for w in q2:
            if len(w)==0: continue
            vocab[w] += 1
    reader = csv.reader(open(test_filepath))
    next(reader)
    for id,q1,q2 in reader:
        q1=q1.split()
        for w in q1:
            if len(w)==0: continue
            vocab[w]+=1
        q2 = q2.split()
        for w in q2:
            if len(w)==0: continue
            vocab[w] += 1
    print len(vocab)
    sorted_vocab = sorted(vocab.items(), cmp=lambda x, y: cmp(x[1], y[1]), reverse=True)

    sorted_vocab=sorted_vocab[:VOCABULARY_SIZE]
    word2idx=dict()
    for widx,(w,_) in enumerate(sorted_vocab):
        word2idx[w]=widx

    with open('word2idx.pkl','w')as f:
        pickle.dump(word2idx,f)


def rewrite_corpus(vocab_path,train_filepath,test_filepath):
    word2idx=pickle.load(open(vocab_path))
    reader=csv.reader(open(train_filepath))
    fw=open('train.txt','w')
    next(reader)
    for id, qid1,qid2,q1,q2,dup in reader:
        q1 = q1.split()
        sent1 = [str(word2idx[w] + 1) if w in word2idx else '0' for w in q1]
        q2 = q2.split()
        sent2 = [str(word2idx[w] + 1) if w in word2idx else '0' for w in q2]
        if len(q1)<1 or len(q2)<1:
            continue
        fw.write(" ".join(sent1) + "\t" + " ".join(sent2)+"\t" +str(dup)+ "\n")
    fw.close()
    reader = csv.reader(open(test_filepath))
    fw = open('test.txt', 'w')
    next(reader)
    for id, q1,q2 in reader:
        q1=q1.split()
        sent1=[str(word2idx[w]+1) if w in word2idx else '0' for w in q1 ]
        q2=q2.split()
        sent2 = [str(word2idx[w] + 1) if w in word2idx else '0' for w in q2]
        fw.write(" ".join(sent1)+"\t"+" ".join(sent2)+"\n")
    fw.close()


def kflod(train_filepath,k=6,ratio=0.8):
    data=open(train_filepath,'r').read().split('\n')
    for i in range(k):
        shuffle(data)
        fw=open('train_'+str(i)+train_filepath,'w')
        separeted=int(len(data)*ratio)
        for line in data[:separeted]:
            fw.write(line+'\n')
        fw.close()
        fw = open('valid_' + str(i) + train_filepath, 'w')
        for line in data[separeted:]:
            fw.write(line + '\n')
        fw.close()


#build_dataset('train.csv','test.csv')
#rewrite_corpus('word2idx.pkl','train.csv','test.csv')
kflod('train.txt')