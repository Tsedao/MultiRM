import numpy as np

def evaluate(model, input_x,model_path=None):
    """
    Calculate the attention weights and predicted probabilities
    """
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    y_pred = model(input_x)
    x = model.embed(input_x)
    output,(h_n,c_n) = model.NaiveBiLSTM(x)
    h_n = h_n.view(-1,output.size()[-1])
    context_vector,attention_weights = model.Attention(h_n,output)

    return attention_weights, y_pred

def cal_attention_every_class(attention_weights):
    length = attention_weights.shape[0]
    attention = np.zeros((1,length+2))
    for i in range(length+2):
        # unravel 3-mers attention
        if i == 0:
            attention[:,0] = attention_weights[0]
        elif i == 1:
            attention[:,1] = attention_weights[0] + attention_weights[1]
        elif i == length +1:
            attention[:,i] = attention_weights[i-2]
        elif i == length:
            attention[:,i] = attention_weights[i-2] + attention_weights[i-1]
        else:
            attention[:,i] = attention_weights[i-2]+attention_weights[i-1]+attention_weights[i]

    return attention

def cal_attention(total_attention_weights):
    """
    Unwarp the 3-mers inputs attention_weights and sum to single nucleotide
        Inputs: Attention weights shape [batch_size, length, num_class]
        Outputs: Unwarped Attention weights shape [batch_size, num_class, length+2]
    """
    num_class = total_attention_weights.shape[-1]
    length = total_attention_weights.shape[1] + 2
    num_samples = total_attention_weights.shape[0]
    total_attention = np.zeros((num_samples,num_class,length))
    for k in range(num_samples):
        tmp = []
        for i in range(num_class):
            tmp.append(cal_attention_every_class(total_attention_weights[k,:,i].detach().cpu().numpy()))
        tmp = np.concatenate(tmp,axis=0)

        total_attention[k,:] = tmp
    return total_attention
