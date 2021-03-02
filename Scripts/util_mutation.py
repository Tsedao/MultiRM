import numpy as np

def mutate_withinattention_single(seq,exclude):
    """
    Perform random mutation within attention area in a single sequence
    Input:
        Seq: RNA sequence in string type, i.e. 'ACCACGT'
        exclude: indices of two attention window, i.e. ((25,27),(2,4))
    """
    exclude1, exclude2 = exclude

    # choose mutation window
    choose_exclude = np.random.uniform(0,1)
    if choose_exclude <= 0.5:
        start, end = exclude1
    else:
        start, end = exclude2
    window_size = end - start + 1

    # choose mutation point
    choose_point = np.random.uniform(0,1)
    for i in range(1, window_size+1):
        if choose_point <= i / window_size:
            mutated_point = start+i-1
            break
    original_type = seq[mutated_point]


    # choose mutation type
    types = ['A','C','G','T']
    left_types = types.remove(original_type)

    choose_type = np.random.uniform(0,1)

    if choose_type <= 1 / 3:
        mutate_type = types[0]
    elif choose_type <= 2 / 3:
        mutate_type = types[1]
    else:
        mutate_type = types[2]
#     print(mutate_type)
#     print(seq[mutated_point])
#     print(mutated_point)
    seq[mutated_point] = mutate_type
    return seq


def mutate_outatt_single(seq, exclude):
    """
    Perform random mutation outside attention area in a single sequence
    Input:
        Seq: RNA sequence in string type, i.e. 'ACCACGT'
        exclude: indices of two attention window, i.e. ((25,27),(2,4))
    """
    exclude_low = exclude[np.argmin(exclude,axis=0)[0]]
    exclude_high = exclude[np.argmax(exclude,axis=0)[0]]

    # sanity check zero size window
    candidate_1 = (0, exclude_low[0]-1)
    candidate_2 = (exclude_low[1]+1, exclude_high[0]-1)
    candidate_3 = (exclude_high[1]+1, len(seq) -1)
    candidates = (candidate_1, candidate_2, candidate_3)

    window_size_1 = exclude_low[0]
    window_size_2 = exclude_high[0]-1 - exclude_low[1]
    window_size_3 = len(seq) - exclude_high[1] -1
    suitable_candidates = np.where(np.array((window_size_1,
                                    window_size_2,
                                    window_size_3))!=0)[0]
    # print(suitable_candidates)
    # choose mutation window
    choose_exclude = np.random.uniform(0,1)
    for k in range(0,len(suitable_candidates)):
        if choose_exclude <= (k+1) / len(suitable_candidates):
            start, end = candidates[suitable_candidates[k]]
            break
    window_size = end - start + 1

    # choose mutation point
    choose_point = np.random.uniform(0,1)
    # print("choose point %.6f, window_size %d"%(choose_point, window_size))
    for i in range(1, window_size+1):
        # print(i / window_size)
        if choose_point <= i / window_size:
            mutated_point = start+i-1
            break
    original_type = seq[mutated_point]


    # choose mutation type
    types = ['A','C','G','T']
    left_types = types.remove(original_type)

    choose_type = np.random.uniform(0,1)

    if choose_type <= 1 / 3:
        mutate_type = types[0]
    elif choose_type <= 2 / 3:
        mutate_type = types[1]
    else:
        mutate_type = types[2]
#     print(mutate_type)
#     print(seq[mutated_point])
#     print(mutated_point)
    seq[mutated_point] = mutate_type
    return seq
