import argparse
import os
import pickle

import torch
import numpy as np
import pandas as pd
import json

from models import model_v3
from util_funs import seq2index, cutseqs, highest_x,visualize,str2bool
from util_att import evaluate, cal_attention

cwd = os.getcwd()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MultiRM APP.")

    parser.add_argument("-s","--seqs",type=str)
    parser.add_argument('--model_weights',default='../Weights/MultiRM/trained_model_51seqs.pkl',type=str)
    parser.add_argument('--embedding_path',default='../Embeddings/embeddings_12RM.pkl',type=str)
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help="used GPU")
    parser.add_argument('--top',type=int,default=3,help='top k consecutive nucleo based on attention weights')
    parser.add_argument('--alpha',type=float,default=0.05,help='significant level')
    parser.add_argument('--att_window',type=int,default=3,help='length of sliding window to aggregate attention weights')
    parser.add_argument('--verbose',type=str2bool,default=False,help='whether showing detailed prediction results or not')
    parser.add_argument('--save',type=str2bool,default=False,help='save the prob, p-value, predicted label and attention matrix')
    parser.add_argument('--save_path',type=str,default=cwd,help='path of directory to store the result')
    parser.add_argument('--save_id',type=str,default='',help='JOBID (for the use of web sever)')

    args = parser.parse_args()

    args.seqs = args.seqs.upper().replace('U','T')                               # preprocessing user input
    num_samples = 1

    RMs = ['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Psi','AtoI']
    num_task = len(RMs)                                                               # 12 modifications
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    original_length = len(args.seqs)
    check_pos = original_length - 51 + 1
    assert original_length >= 51


    args.embedding_path = os.path.join(cwd,args.embedding_path)
    args.model_weights = os.path.join(cwd,args.model_weights)

    embeddings_dict = pickle.load(open(args.embedding_path,'rb'))

    model = model_v3(num_task=num_task,use_embedding=True).cuda()
    model.load_state_dict(torch.load(args.model_weights))

    neg_prob = pd.read_csv('neg_prob.csv',header=None,index_col=0)

    probs = np.zeros((num_task,check_pos))
    p_values = np.zeros((num_task,check_pos))
    labels = np.zeros((num_task,original_length))
    attention = np.zeros((num_task,original_length))

    print('Note: MultiRM does not make predictions for the first and \nlast 25nt of the input sequence.')
    print()
    str_out = []
    print('*'*24+'Reporting'+'*'*24)
    str_out.append('*'*24+'Reporting'+'*'*24)

    for pos in range(original_length-51+1):
        cutted_seqs = args.seqs[pos:pos+51]


        seqs_kmers_index = seq2index([cutted_seqs],embeddings_dict)

        seqs_kmers_index = torch.transpose(torch.from_numpy(seqs_kmers_index),0,1)



        # Evaluate and cal Attention weights
        attention_weights, y_preds = evaluate(model,seqs_kmers_index)
        total_attention = cal_attention(attention_weights)


        y_prob = [y_pred.detach().cpu().numpy()[0] for y_pred in y_preds]


        for k in range(num_task):
            bool = neg_prob.iloc[k,:] > y_prob[k]
            p_value = np.sum(bool) / len(bool)

            if p_value < args.alpha:
                labels[k,pos+25] = 1
            p_values[k,pos] = p_value
            probs[k,pos] = y_prob[k]



        index_list = [i for i, e in enumerate(labels[:,pos+25]) if e == 1]
        if index_list == []:
            if args.verbose:
                sentense = 'There is no modification site at %d '%(pos+26)
                print(sentense)
                str_out.append(sentense)
        else:
            for idx in index_list:
                if args.verbose:
                    sentense = '%s is predicted at %d with p-value %.4f and alpha %.3f'%(
                                     RMs[idx],pos+26,p_values[idx,pos],args.alpha)
                    print(sentense)
                    str_out.append(sentense)


                this_attention = total_attention[0,idx,:]
                position_dict = highest_x(this_attention,w=args.att_window)

                edge = pos

                starts = []
                ends = []
                scores = []
                for j in range(1,args.top+1):
                    score, start, end = position_dict[j]
                    starts.append(start+edge)
                    ends.append(end+edge)
                    scores.append(score)

                    attention[idx,start+edge:end+edge+1] = 1

    # print(attention)
    seperate_line = '*'*15+'Visualize modification sites'+'*'*14
    str_out.append(seperate_line)
    print(seperate_line)
    str_list_1 = visualize(args.seqs,labels,RMs)
    str_out += str_list_1
    print()
    print()
    str_out.append('\n')
    str_out.append('\n')
    seperate_line_2 = '*'*19+'Visualize Attention'+'*'*19
    print(seperate_line_2)
    str_out.append(seperate_line_2)
    str_list_2 = visualize(args.seqs,attention,RMs)
    str_out += str_list_2


    if args.save:

        if not os.path.exists(os.path.join(args.save_path,args.save_id)):
            os.mkdir(os.path.join(args.save_path,args.save_id))

        cols_name = [i+26 for i in range(len(args.seqs)-50)]
        cols_name_2 = [i for i in range(len(args.seqs))]

        with open(os.path.join(args.save_path,args.save_id,'../Results/visualization.json'),'w') as f:
            json.dump(str_out, f)

        # with open(os.path.join(args.save_path,args.save_id,'../Results/visualization.csv'), 'w') as f:
        #     for line in str_out:
        #         f.write("%s\n"%(line))

        pd.DataFrame(data=probs,index=RMs,columns=cols_name).to_csv(os.path.join(args.save_path,args.save_id,'../Results/probs.csv'),header=True)
        pd.DataFrame(data=p_values,index=RMs,columns=cols_name).to_csv(os.path.join(args.save_path,args.save_id,'../Results/p_values.csv'),header=True)
        pd.DataFrame(data=labels,index=RMs,columns=cols_name_2).to_csv(os.path.join(args.save_path,args.save_id,'../Results/pred_labels.csv'),header=True)
        pd.DataFrame(data=attention,index=RMs,columns=cols_name_2).to_csv(os.path.join(args.save_path,args.save_id,'../Results/attention.csv'),header=True)
