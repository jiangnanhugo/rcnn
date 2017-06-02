#!/bin/sh
THEANO_FLAGS="floatX=float32,device=cuda0,mode=FAST_RUN,lib.cnmem=0.9"  python main.py --model_dir model/parameters_7809.02.pkl --goto_line 205500
