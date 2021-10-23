import torch
from torch import nn
import numpy as np
import pandas as pd

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to_onehot(seq):
    """
    Inputs:
        seq: RNA seqs in string capitalized form
    """
    length = len(seq)
    out = np.zeros((4,length))
    for i in range(length):
        if seq[i] == "A":
            out[0,i] = 1
        elif seq[i] == 'C':
            out[1,i] = 1
        elif seq[i] == 'G':
            out[2,i] = 1
        elif seq[i] == 'T' or seq[i] == 'U':
            out[3,i] = 1
        elif seq[i] == '-' or seq[i] == 'N':
            out[:,i] = np.array([0.25,0.25,0.25,0.25])
    return out

def word2index_(my_dict):
    word2index = dict()
    for index, ele in enumerate(list(my_dict.keys())):
        word2index[ele] = index

    return word2index

def index2word_(my_dict):
    index2word = dict()
    for index, ele in enumerate(list(my_dict.keys())):
        index2word[index] = ele
    return index2word

def mapfun(x,my_dict):
    if x not in list(my_dict.keys()):
        return None
    else:
        return word2index_(my_dict)[x]

def seq2index(seqs,my_dict,window=3,save_data=False):
    """
    Convert single RNA sequences to k-mers representation.
        Inputs: ['ACAUG','CAACC',...] of equal length RNA seqs
        Example: 'ACAUG' ----> [ACA,CAU,AUG] ---->[21,34,31]
    """

    num_samples = len(seqs)
    temp = []
    for k in range(num_samples):
        length = len(seqs[k])
        seqs_kmers = [seqs[k][i:i+window] for i in range(0,length-window+1)]
        temp.append(seqs_kmers)


    seq_kmers = pd.DataFrame(data = np.concatenate(temp,axis=0))

    # load pretained word2vec embeddings

    word2index = word2index_(my_dict)

    seq_kmers_index = seq_kmers.applymap(lambda x: mapfun(x,my_dict))


    return seq_kmers_index.to_numpy()

def cutseqs(seqs,length):
    """
    Cut the input RNA/DNA seqs into desired length
    """

    for i in range(len(seqs)):
        # make sure the length of RNA is odd
        assert len(seqs[i]) % 2 != 0
        mid_idx = len(seqs[i]) // 2 + 1
        radius = (length - 1) // 2
        seqs[i] = seqs[i][mid_idx-radius-1:mid_idx+radius]
    return seqs

def highest_score(a,w):
    """
    Inputs:
        a: a 1-D numpy array contains the scores of each position
        w: length of window to aggregate the scores
    """

    assert(len(a)>=w)

    best = -20000
    best_idx_start = 0
    best_idx_end =0
    for i in range(len(a)-w + 1):
        tmp = np.sum(a[i:i+w])
        if tmp > best:
            best = tmp
            best_idx_start = i
            best_idx_end = i + w - 1

    return best, best_idx_start, best_idx_end

def highest_x(a,w,p=1):
    """
    Inputs:
        a: a 1-D numpy array contains the scores of each position
        w: length of window to aggregate the scores
        p: length of padding when maximum sum of consecutive numbers are taken
    """

    lists = [{k:v for (k,v) in zip(range(len(a)),a)}]
    result = {}
    max_idx = len(a) -1
    count = 1
    condition = [True]
    while any(con is True for con in condition):
        starts = []
        ends = []
        bests = []

        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())


            start_idx = idx[0]

            if len(values) >= w:
                highest, highest_idx_start, highest_idx_end = highest_score(values,w)

                starts.append(highest_idx_start+start_idx)


                ends.append(highest_idx_end+start_idx)


                bests.append(highest)


        best_idx = max(zip(bests, range(len(bests))))[1]   # calculate the index of maximum sum

        cut_value = bests[best_idx]

        if starts[best_idx] - p >=0:
            cut_idx_start = starts[best_idx] - p
        else:
            cut_idx_start = 0

        if ends[best_idx] + p <=max_idx:
            cut_idx_end = ends[best_idx] + p
        else:
            cut_idx_end = max_idx

        result[count] = (cut_value,starts[best_idx],ends[best_idx])


        copy = lists.copy()

        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())

            start_idx, end_idx = idx[0], idx[-1]

            if len(values) < w:
                copy.remove(ele)
            else:
#                 print(cut_idx_start,cut_idx_end)
#                 print(start_idx,end_idx)
#                 print(values)
                if (cut_idx_end < start_idx) or (cut_idx_start > end_idx):

                    pass
                elif (cut_idx_start < start_idx) and (cut_idx_end >= start_idx):
                    copy.remove(ele)
                    values = values[cut_idx_end-start_idx+1:]
                    idx = idx[cut_idx_end-start_idx+1:]
                    ele = {k:v for (k,v) in zip(idx,values)}

                    if ele != {}:
                        copy.append(ele)

                elif (cut_idx_start >= start_idx) and (cut_idx_end <= end_idx):
                    copy.remove(ele)
                    values_1 = values[:cut_idx_start-start_idx]
                    idx_1 = idx[:cut_idx_start-start_idx]
                    ele_1 = {k:v for (k,v) in zip(idx_1,values_1)}

                    values_2 = values[cut_idx_end-start_idx+1:]
                    idx_2 = idx[cut_idx_end-start_idx+1:]
                    ele_2 = {k:v for (k,v) in zip(idx_2,values_2)}

                    if ele_1 != {}:
                        copy.append(ele_1)
                    if ele_2 != {}:
                        copy.append(ele_2)

                elif (cut_idx_start <= end_idx) and (cut_idx_end > end_idx):
                    copy.remove(ele)
                    values = values[:cut_idx_start-start_idx]
                    idx = idx[:cut_idx_start-start_idx]
                    ele = {k:v for (k,v) in zip(idx,values)}

                    if ele != {}:
                        copy.append(ele)

        lists = copy
#        print(lists)
        count = count + 1
        condition = [len(i)>=w for i in lists]
#        print(condition)

    return result


def visualize(raw_seq,weights,RMs):
    num_bp = len(raw_seq) // 50 + 1
    str_list = []
    for k in range(num_bp):
        start = 50*k
        end = np.min([50*(k+1),len(raw_seq)])
        cutted_seqs = raw_seq[start:end]
        # 58 characters
        title = '*'*24+'%3d-%3d nt' %(start+1,end) + '*'*23
        origin = '%-7s'%('Origin')+cutted_seqs
        print(title)
        print(origin)
        str_list.append(title)
        str_list.append(origin)
        for i in range(len(RMs)):
            weight = weights[i,:]
            new = ['-'] * (end-start)
            for j in range(start,end):
                if int(weight[j]) == 1:
                    new[j-start] = raw_seq[j]
            modif = '%-7s'%(RMs[i]+'*'*(6-len(RMs[i])))+''.join(new)
            print(modif)
            str_list.append(modif)

    return str_list
