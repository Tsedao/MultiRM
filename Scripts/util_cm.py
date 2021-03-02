import numpy as np
import pandas as pd
import h5py


from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap

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

def reduction_clustering(df,n_clusters=3,eps=0.3):
    # Y = PCA(n_components=2).fit_transform(df)
    Y = umap.UMAP(n_components=2).fit_transform(df)
    # clustering = SpectralClustering(n_clusters=n_clusters,assign_labels="discretize",random_state=0).fit(Y)

    clustering = DBSCAN(eps=eps,min_samples=10).fit(Y)

    cluster_labels = clustering.labels_
    return cluster_labels

def result_idx(results,df,cluster_label):
    """
    Get the correspond samples from each cluster along with its label and location
    in original rna seqs
        Inputs:
            results: list of (score, start_idx, end_idx) tuple with loc
            df: 20-dim vector
            cluster_label:label for each sample. shape: (num_samples,)
    """
    class_dict = {}
    for label in np.unique(cluster_label):
        index_list_df = list(df.index[cluster_label==label])
        out = []
        for string in index_list_df:
            string_list = string.split('_')
            sample_idx = int(string_list[0])
            rank_idx = int(string_list[1])

            score, start_idx, end_idx = results[sample_idx].get(rank_idx)

            # for not aligned seqs
            # out.append((sample_idx,start_idx,end_idx))

            # for aligned seqs
            out.append((score,sample_idx,rank_idx))
        class_dict[label] = out
    return class_dict

def pfm(seqs):
    """
    Inputs:
           seqs: list of seqs
    Outputs:
           A (4, x) matrix where x is the length of input seqs

           A
           C
           G
           T/U

    """
    length = len(seqs[0])
    out = np.zeros((4,length))
    for seq in seqs:
        #print(seq)
        for i in range(length):
            if seq[i] == 'A':
                out[0,i] += 1
            elif seq[i] == 'C':
                out[1,i] += 1
            elif seq[i] == 'G':
                out[2,i] += 1
            elif seq[i] =='T' or seq[i] =='U':
                out[3,i] += 1
            elif seq[i] =='-':
                pass
    return out

def pwm(pfm,bg=0.25):
    totals = []
    for i in range(pfm.shape[1]):
        totals.append(np.sum(pfm[:,i]))
    total = max(totals)
    p = (pfm + (np.sqrt(total) * 1/4)) / (total + (4 * (np.sqrt(total) * 1/4)))
    # p = np.log2(p/bg)
    return p

def read_seq(index_list,raw_seqs):
    """
        Inputs:
            Index_list: [(sample_idx, start_idx, end_idx)] a list of tuples in single class
            raw_seqs: original RNA sequences
    """
    seqs = []
    scores = []
    for (score, sample_idx, end_idx) in index_list:
        # seqs.append(raw_seqs[sample_idx][start_idx:end_idx+1])

        # for aligned seqs
        seqs.append(raw_seqs[3*sample_idx+end_idx-1])
        scores.append(score)
    return (seqs,score)

def read_seq_all(index_dict,raw_seqs):
    class_dict = {}
    score_dict = {}
    for k, v in index_dict.items():
        seqs, scores = read_seq(v,raw_seqs)
        class_dict[k], score_dict[k] = seqs, scores
    return (class_dict,score_dict)

def to_onehot(seq):
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
        elif seq[i] == '-':
            out[:,i] = np.array([0.25,0.25,0.25,0.25])
    return out


def helper(RM,nucleos,data_type,length,RM_name, download=False, w=5,k=3,p=1):
    """
    extract short seqs and relative scores
    """

    RM_sum = np.sum(RM,axis=1)

    num_samples = RM.shape[0]
    results = []                                 # (score,start_idx,end_idx)
    for i in range(num_samples):
        result = highest_x(RM_sum[i,:],w=w,p=p)
        results.append(result)

    short_seqs = []
    scores = []

    for j in range(num_samples):
        seq = results[j]
        new_dict ={}
        for i in range(1,k+1):
            start_idx = seq.get(i)[1]
            end_idx = seq.get(i)[2]
            score = seq.get(i)[0]
            # extract short seqs
            short_seq = ''.join(nucleos.iloc[j,start_idx:end_idx+1])

            short_seqs.append(short_seq)
            scores.append(score)

            if download:
                with open('../Seqs/%s_%s_%d_wid%d_top%d.csv'%(data_type,RM_name,length,w,k),'a') as file:

                    file.write(short_seq)
                    file.write('\n')

                with open('../Seqs/%s_%s_%d_wid%d_top%d_score.csv'%(data_type,RM_name,length,w,k),'a') as file:

                    file.write(str(score))
                    file.write('\n')
    return short_seqs, scores


def cal_consensus_motif_2(seqs,scores,eps=0.3):
    """
    Computing motifs of aligned sequences
    Input:
         Seqs: aligned sequenced
         scores: aggregated ig score over such a short aligned sequence
         eps: parameters for DBSCAN
    """
    data = []
    for i in range(len(seqs)):
        tmp = to_onehot(seqs[i]).T.flatten()
        data.append(tmp)

    df = pd.DataFrame(data=data,index=list(range(len(seqs))))

    class_labels = reduction_clustering(df,n_clusters=6,eps=eps)

    seqs_dict = {}
    scores_dict = {}
    for i in np.unique(class_labels):
        if i != -1:
            class_seqs = seqs[class_labels==i]
            class_scores = scores[class_labels==i]
            avg_scores = np.sum(class_scores) / len(class_seqs)
            seqs_dict[i] = list(class_seqs)
            scores_dict[i] = avg_scores
            print('class:%d score:%.5f'%(i,avg_scores))
            # print(class_seqs)

    # sort the class by ig score
    index = sorted(scores_dict,key=scores_dict.__getitem__,reverse=True)

    pwm_weights = []
    ig_score = []
    for idx in index:
        pwm_weights.append(np.expand_dims(pwm(pfm(seqs_dict[idx])),axis=0))
        ig_score.append(scores_dict[idx])

    consensus_motif = np.concatenate(pwm_weights,axis=0)

    return consensus_motif,ig_score
