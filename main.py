import argparse
import os
import pickle

import torch
import numpy as np
import pandas as pd

from models import model_v3
from util_funs import seq2index, cutseqs, highest_x, visual_att
from util_att import evaluate, cal_attention

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MultiRM APP.")

    parser.add_argument("-s","--seqs",type=str)
    parser.add_argument('--model_weights',default='./model_weights/trained_model_51seqs.pkl',type=str)
    parser.add_argument('--embedding_path',default='./Embeddings/embeddings_12RM.pkl',type=str)
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help="used GPU")
    parser.add_argument('--top',type=int,default=3,help='top k consecutive nucleo based on attention weights')

    args = parser.parse_args()

    num_samples = 1

    RMs = ['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Psi','AtoI']
    num_task = len(RMs)                                                               # 12 modifications
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    original_length = len(args.seqs)
    assert len(args.seqs) >= 21
    if len(args.seqs) < 51:
        args.model_weights = './model_weights/trained_model_21seqs.pkl'
        length = 21
        thresholds = [0.006630,0.202159,0.125162,0.129332,0.101850,0.355195,
                      0.140955,0.299688,0.031190,0.181220,0.180271,0.350196]
    else:
        length = 51
        thresholds = [0.008291,0.106070,0.141367,0.110626,0.107747,0.295998,
                      0.126506,0.433442,0.032172,0.116947,0.189863,0.360362]
    cutted_seqs = cutseqs([args.seqs],length)


    embeddings_dict = pickle.load(open(args.embedding_path,'rb'))

    seqs_kmers_index = seq2index(cutted_seqs,embeddings_dict)

    seqs_kmers_index = torch.transpose(torch.from_numpy(seqs_kmers_index),0,1)

    model = model_v3(num_task=num_task,use_embedding=True).cuda()
    model.load_state_dict(torch.load(args.model_weights))

    # Evaluate and cal Attention weights
    attention_weights, y_preds = evaluate(model,seqs_kmers_index)
    total_attention = cal_attention(attention_weights)

    for i in range(num_samples):
        y_prob = [y_pred.detach().cpu().numpy()[i] for y_pred in y_preds]
        bool = [y >=t for y, t in zip(y_prob,thresholds)]
        print('**'*20+'Sample %d'%(i+1) + '**'*20)

        index_list = [i for i, e in enumerate(bool) if e == True]
        if index_list == []:
            print('There is no modification sites in that sequence')
            print()
            break
        for idx in index_list:
            y_label_pred = RMs[idx]
            print('The sequence is predicted as: %s with prob %.6f at threshold %.6f'
                   %(y_label_pred,y_prob[idx],thresholds[idx]))

            this_attention = total_attention[i,idx,:]
            position_dict = highest_x(this_attention,w=3)

            edge = (original_length-length) / 2

            starts = []
            ends = []
            scores = []
            for j in range(1,args.top+1):
                score, start, end = position_dict[j]
                starts.append(start+edge)
                ends.append(end+edge)
                scores.append(score)

            visual_att(args.seqs,starts,ends)
