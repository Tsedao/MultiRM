import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, sampler

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

class HDF5Dataset(Dataset):
    """
    Args:
        h5data_path(str): path of h5 file
        train(boolean): whether use train data or not
        transform(optional)
    """
    def __init__(self, h5data_path,train=True, transform=None):
        self.h5data = h5py.File(h5data_path,'r')
        self.transform = transform
        self.train = train
        self.train_x = np.array(self.h5data["train_in_seq"])
        self.train_y = np.array(self.h5data["train_out"])
        self.valid_x = np.array(self.h5data["valid_in_seq"])
        self.valid_y = np.array(self.h5data["valid_out"])

    def __getitem__(self, index):

        if self.train:
            x = self.train_x[index,...]
            y = self.train_y[index,...]
        else:
            x = self.valid_x[index,...]
            y = self.valid_y[index,...]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        y = torch.from_numpy(y)

        # convert datatype
        x = x.type('torch.cuda.FloatTensor')
        y = y.type('torch.cuda.FloatTensor')
        return (x, y)

    def __len__(self):
        if self.train:
            return self.train_x.shape[0]
        else:
            return self.valid_x.shape[0]
    # test purpose
    # def __len__(self):
    #     return 200

class RMdata(Dataset):


    def __init__(self, data_path, use_embedding,length,mode):
        """
        Inputs:
            mode: train, valid, test
        """
        self.data_path = data_path
        self.mode = mode
        self.use_embedding = use_embedding
        self.radius = length // 2

        if self.mode == 'train':
            self.train_x = pd.read_hdf(self.data_path,'train_in')
            self.train_y = pd.read_hdf(self.data_path,'train_out').to_numpy()
            self.valid_x = pd.read_hdf(self.data_path,'valid_in')
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()

            if self.use_embedding:
                print('Using pre-trained embeddings!'+'-' * 60)
                self.train_x = pd.read_hdf(self.data_path,'train_in_3_mers')
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                self.train_x = self.train_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                # print(self.train_x.shape[1])
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                # cropping the sequence one_hot encoding
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                # print(middle_index)
                self.train_x = self.train_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
        else:
            self.valid_x = pd.read_hdf(self.data_path,'valid_in')
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()
            self.test_x = pd.read_hdf(self.data_path,'test_in')
            self.test_y = pd.read_hdf(self.data_path,'test_out').to_numpy()

            if self.use_embedding:
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                self.test_x = pd.read_hdf(self.data_path,'test_in_3_mers')
                # cropping the sequence accroding to its length
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                # print(self.train_x.shape[1])
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                self.test_x = self.test_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                # cropping the sequence one_hot encoding
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                # print(middle_index)
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                # print(self.train_x.shape[1])
                self.test_x = self.test_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()

        self.class_name = list(pd.read_hdf(self.data_path,'test_out').columns)


    def __getitem__(self,index):
        if self.mode == 'train':
            x = self.train_x[index,...]
            y = self.train_y[index,...]
        elif self.mode == 'valid':
            x = self.valid_x[index,...]
            y = self.valid_y[index,...]
        elif self.mode == 'test':
            x = self.test_x[index,...]
            y = self.test_y[index,...]


        x = torch.from_numpy(x)
        y = torch.from_numpy(y)


        x = x.type('torch.cuda.FloatTensor')
        y = y.type('torch.cuda.FloatTensor')

        return (x, y)

    def __len__(self):
        if self.mode == 'train':
            return self.train_x.shape[0]
        elif self.mode == 'valid':
            return self.valid_x.shape[0]
        elif self.mode == 'test':
            return self.test_x.shape[0]


def load_RM_data(path,batch_size,length,use_embedding,balanced_sampler=False):

    train = RMdata(path,use_embedding=use_embedding,
                            length= length,mode='train')

    valid = RMdata(path,use_embedding=use_embedding,
                    length=length, mode='valid')

    if not balanced_sampler:
        train_loader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
    else:
        weights_train = make_weights_for_balanced_classes(train)
        # weights_valid = make_weights_for_balanced_classes(valid)

        weights_train = torch.cuda.DoubleTensor(weights_train)
        # weights_valid = torch.cuda.DoubleTensor(weights_valid)

        sampler_train = sampler.WeightedRandomSampler(weights_train, len(weights_train))
        # sampler_valid = sampler.WeightedRandomSampler(weights_valid, len(weights_valid))

        train_loader = DataLoader(dataset=train,batch_size=batch_size,sampler=sampler_train)
        # valid_loader = DataLoader(dataset=valid,batch_size=batch_size,sampler=sampler_valid)

    valid_loader = DataLoader(dataset=valid,batch_size=batch_size,shuffle=True)

    return train_loader, valid_loader


def make_weights_for_balanced_classes(dataset):
    X, y = dataset[:]
    num_examples = len(y)
    nclasses = len(y[1]) + 1
    count = np.zeros(nclasses)
    y = y.cpu().numpy()
    for i in range(num_examples):
        count[np.concatenate([np.squeeze(y[i,:]),np.array([0])])==1] += 1
    # negative class weight
    count[-1] = num_examples - np.sum([count[i] for i in range(nclasses)])
    weight_per_class = np.zeros(nclasses)
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * num_examples
    for i in range(num_examples):
        if not list(np.squeeze(y[i,:])) == list(np.zeros(len(y[1]))):
            weight[i] = np.mean(weight_per_class[np.concatenate([np.squeeze(y[i,:]),np.array([0])])==1])
        else:
            # negative cases
            weight[i] = weight_per_class[-1]
    return weight

def cal_precision(y_true, y_pred,eps=1e-7):
	true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
	predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + eps)
	return precision

def cal_recall(y_true, y_pred,eps=1e-7):
	true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
	possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
	recall = true_positives / (possible_positives + eps)
	return recall

def cal_accuary(y_true, y_pred):
    acc = torch.mean((torch.round(torch.clamp(y_pred,0,1))==y_true).type('torch.cuda.FloatTensor'))
    return acc

def precision_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based precision

    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not (y_pred[i] == 0).all():
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_pred_posi = np.sum(y_pred[i])
            result += n_true_posi / n_pred_posi
    return result / n_samples

def recall_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based recall

    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not (y_true[i] == 0).all():
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_ground_true = np.sum(y_true[i])
            result += n_true_posi / n_ground_true
    return result / n_samples

def f1_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based recall
    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not ((y_true[i] == 0).all() and (y_pred[i] == 0).all()):
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_ground_true = np.sum(y_true[i])
            n_pred_posi = np.sum(y_pred[i])
            f1 = 2*(n_true_posi) / (n_ground_true+n_pred_posi)
            result += f1
    return result / n_samples

def hamming_loss(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: hamming loss
    """
    n_samples = y_true.shape[0]
    n_classes = y_true.shape[1]
    loss = 0
    for i in range(n_samples):
        xor = np.sum((y_true[i] + y_pred[i]) % 2)
        loss += xor / n_classes
    return loss / n_samples


def cal_metrics(model_out,label,plot=False,class_names=None,plot_name=None):
    """
    Inputs:
        class_name: for plot purpose
    """
    from sklearn.metrics import recall_score,precision_score,roc_auc_score,roc_curve, average_precision_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve
    num_task = len(model_out)

    # threshold_list = [0.5 for i in range(num_task)]                              # thresholds standard
    threshold_list = [0.002887,0.004897,0.001442,0.010347,0.036834,0.028677,
                      0.009135,0.095019,0.001394,0.007883,0.113931,0.125591]     # thresholds for multirm #

    # threshold_list = [0.004554,0.014769,0.005969,0.043316,0.076438,0.091157,
    #                   0.121174,0.175164,0.006239,0.001260,0.051128,0.255274]    # thresholds for hmm


    # threshold_list = [0.007389,0.050478,0.046165,0.068021,0.088967,0.150652,    # thresholds for CNN+Lstm
    #                   0.080001,0.317348,0.003866,0.013430,0.090117,0.256765]
    metrics = {'recall':[],'precision':[],'accuracy':[],'auc':[],'auc_2':[],
                'sn':[],'sp':[],'acc_2':[],'mcc':[], 'ap':[], 'ap_2':[]}

    # auc_2: auc across all samples
    # auc: auc across one single class
    metrics_avg = {'recall':0, 'precision':0,'accuracy':0,'auc':0,'auc_2':0}

    # Compute ROC curve and ROC area for each class
    fpr,tpr = dict(), dict()
    fpr_2,tpr_2 = dict(), dict()

    precisions, recalls = dict(), dict()
    precisions_m, recalls_m = dict(), dict()

    label = label.cpu().numpy()
    Y_pred = np.zeros(label.shape)
    for i in range(num_task):
        y_true = label[:,i]
        y_pred = torch.clamp(model_out[i].cpu().detach(),0,1).numpy()
        y_pred = np.array([0 if instance < threshold_list[i] else 1 for instance in list(y_pred)])
        Y_pred[:,i] = y_pred
        y_score = model_out[i].cpu().detach().numpy()
        # if i==0:
            # print(y_pred[y_true==1])
        # recall = recall_score(y_true,y_pred,zero_division=1)
        # precision = precision_score(y_true,y_pred,zero_division=1)
        acc = np.mean(y_true==y_pred)
        # handle one_class problem

        # test binary auc
        auc = roc_auc_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])

        # test binary ap
        ap = average_precision_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])

        # test multiclass auc
        auc_2 = roc_auc_score(y_true,y_score)

        # test multi ap
        ap_2 = average_precision_score(y_true,y_score)

        fpr[i], tpr[i], thresholds = roc_curve(y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])
        fpr_2[i], tpr_2[i], thresholds_2 = roc_curve(y_true, y_score)

        precisions[i], recalls[i], _ = precision_recall_curve(y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])
        precisions_m[i], recalls_m[i], _ = precision_recall_curve(y_true, y_score)

        gmeans = np.sqrt(tpr_2[i] * (1-fpr_2[i]))

        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds_2[ix], gmeans[ix]))

        best_threshold = thresholds_2[ix]
        y_pred_new = np.array([0 if instance < best_threshold else 1 for instance in list(y_score)])

        # binary based confusion_matrix
        # tn, fp, fn, tp = confusion_matrix(y_true[i*100:(i+1)*100], y_pred_new[i*100:(i+1)*100]).ravel()

        # multiclass based confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()
        pp = tp+fn
        pn = tn+fp
        sensitivity = tp / pp
        specificity = tn / pn
        recall = sensitivity
        precision = tp / (tp + fp)
        acc_2 = (tp+tn) / (pp+pn)
        # mcc = acc_2 / np.sqrt((1+(fp-fn)/pp)*(1+(fn-fp)/pn))
        mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))


        # update dictionary

        metrics['auc_2'].append(auc_2)
        metrics['sn'].append(sensitivity)
        metrics['sp'].append(specificity)
        metrics['acc_2'].append(acc_2)
        metrics['mcc'].append(mcc)
        metrics['ap'].append(ap)
        metrics['ap_2'].append(ap_2)




        metrics['recall'].append(recall)
        metrics['precision'].append(precision)
        metrics['accuracy'].append(acc)
        metrics['auc'].append(auc)


        metrics_avg['recall'] += recall
        metrics_avg['precision'] += precision
        metrics_avg['accuracy'] += acc
        # metrics_avg['auc'] += auc

    precision_multi_ = precision_multi(label,Y_pred)
    recall_multi_ = recall_multi(label,Y_pred)
    f1_multi_ = f1_multi(label,Y_pred)
    hamming_loss_ = hamming_loss(label,Y_pred)

    print("precision multi: %f"%(precision_multi_))
    print("recall multi: %f"%(recall_multi_))
    print("f1 multi: %f"%(f1_multi_))
    print("hamming loss: %f"%(hamming_loss_))

    metrics_avg['recall'] /= num_task
    metrics_avg['precision'] /= num_task
    metrics_avg['accuracy'] /= num_task
    # metrics_avg['auc'] /= num_task

    print(plot)
    if plot:

        # define colors
        colors = [(39,64,139),(0,128,128),(31, 119, 180), (44, 160, 44), (152, 223, 138), (174, 199, 232),
                  (255, 127, 14), (255, 187, 120),(214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213)]

        for i in range(len(colors)):
            r, g, b = colors[i]
            colors[i] = (r / 255., g / 255., b / 255.)

        # modifying parameters for plot
        from math import sqrt
        golden_mean = (sqrt(5)-1.0)/2.0 #used for size=
        fig_width = 6 # fig width in inches
        fig_height = fig_width*golden_mean # fig height in inches
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['axes.titlesize'] = 10
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['font.family'] = 'serif'
        # params = {'axes.labelsize': 10, # fontsize for x and y labels (was 10)
        #       'axes.titlesize': 10,
        #       'font.size': 10,
        #       'legend.fontsize': 10,
        #       'xtick.labelsize': 8,
        #       'ytick.labelsize': 8,
        #       'text.usetex': False,
        #       'font.family': 'serif'
        #       }
        lw = 2
        #fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(13,4),gridspec_kw={'width_ratios': [1, 2.2]})

        # roc curve
        fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(fig_width*2+0.7,fig_height+0.1))

        # PR curve
        fig_2, axes_2 = plt.subplots(nrows=1,ncols=2,figsize=(fig_width*2+0.7,fig_height+0.1))
        # matplotlib.rcParams.update(params)
        # set color palettes

        for i, class_name in zip(range(num_task), class_names):

            axes[0].plot(fpr[i], tpr[i], color=colors[i],lw=lw)
            axes[0].plot([0, 1], [0, 1], 'k--', lw=lw)
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_ylim([0.0, 1.0])
            axes[0].tick_params(axis='x',which='both',top=False)
            axes[0].tick_params(axis='y',which='both',right=False)
            axes[0].set_aspect('equal', adjustable='box')
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC curves (binary)')

            axes_2[0].plot(recalls[i], precisions[i], color=colors[i],lw=lw)
            axes_2[0].plot([0, 1], [0.5, 0.5], 'k--', lw=lw)
            axes_2[0].set_xlim([0.0, 1.0])
            axes_2[0].set_ylim([0.45, 1.0])
            axes_2[0].tick_params(axis='x',which='both',top=False)
            axes_2[0].tick_params(axis='y',which='both',right=False)
            xmin, xmax = axes_2[0].get_xlim()
            ymin, ymax = axes_2[0].get_ylim()
            axes_2[0].set_aspect(abs((xmax-xmin)/(ymax-ymin)), adjustable='box')
            axes_2[0].set_xlabel('Recall')
            axes_2[0].set_ylabel('Precision')
            axes_2[0].set_title('PR curves (binary)')


            if class_name == 'Atol':
                class_name = 'A-to-I'
            elif class_name == 'hPsi':
                class_name = 'Psi'
            elif class_name[-1] == 'm':
                class_name = class_name[1:]
            else:
                # tmp = class_name[2:]
                # num = class_name[1]
                # class_name = 'm^{%s}%s'%(num,tmp)

                class_name = class_name[1:]

            axes[1].plot(fpr_2[i], tpr_2[i], color=colors[i],lw=lw,
                     label ='%s ($AUC_{b}$ = %.2f, $AUC_{m}$ = %.2f)'%(class_name,
                                metrics['auc'][i],metrics['auc_2'][i]))
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.0])
            axes[1].tick_params(axis='x',which='both',top=False)
            axes[1].tick_params(axis='y',which='both',right=False,left=False,labelleft=False)
            axes[1].set_aspect('equal', adjustable='box')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC curves (multiple)')

            axes_2[1].plot(recalls_m[i], precisions_m[i], color=colors[i],lw=lw,
                     label ='%s ($AP_{b}$ = %.2f, $AP_{m}$ = %.2f)'%(class_name,
                                metrics['ap'][i],metrics['ap_2'][i]))
            axes_2[1].set_xlim([0.0, 1.0])
            axes_2[1].set_ylim([0.0, 1.0])
            axes_2[1].tick_params(axis='x',which='both',top=False)
            axes_2[1].tick_params(axis='y',which='both',right=False,left=False,labelleft=True)
            xmin, xmax = axes_2[1].get_xlim()
            ymin, ymax = axes_2[1].get_ylim()
            axes_2[1].set_aspect(abs((xmax-xmin)/(ymax-ymin)), adjustable='box')
            axes_2[1].set_xlabel('Recall')
            axes_2[1].set_ylabel('Precision')
            axes_2[1].set_title('PR curves (multiple)')

        # Shrink current axis by 20%
        # box = axes[1].get_position()
        # print(box)
        # axes[1].set_position([box.x0, box.y0, box.x1-box.width * 0.5, box.height])
        # print(axes[1].get_position())
        axes[1].plot([0, 1], [0, 1], 'k--', lw=lw, label='no skill')
        axes_2[1].plot([0, 1], [0.04, 0.04], 'k--', lw=lw, label = 'no skill')

        # Put a legend to the right of the current axis
        axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1),borderaxespad=0.,frameon=False)
        axes_2[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1),borderaxespad=0.,frameon=False)

        fig.tight_layout()
        fig_2.tight_layout()

        fig.savefig('../Figs/roc_curve_%s.pdf'%(plot_name))
        fig_2.savefig('../Figs/precision_recall_curve_%s.pdf'%(plot_name))

        print('Successfully save figure to ../Figs/roc_curve_%s.pdf'%(plot_name))
        print('Successfully save figure to ../Figs/precision_recall_curve_%s.pdf'%(plot_name))


    return metrics,metrics_avg

def cal_metrics_sampling(model_out,label):

    from sklearn.metrics import recall_score,precision_score,roc_auc_score,roc_curve, average_precision_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve

    label = label.cpu().numpy()
    Y_pred = np.zeros(label.shape)
    num_task = len(model_out)
    metrics = {i : {'acc':[],'auc':[], 'ap':[], 'fprs':[],
                     'tprs':[],'precisions':[],'recalls':[]} for i in range(num_task)}
    total_num = 304661
    posi_num = np.array([1591, 1878,1471,2253,16346,3207,3696,65178,2447,1036,3137,52618])
    neg_num = total_num - posi_num
    ratio = np.round(neg_num / posi_num).astype(int)
    iterations = 2000

    for i in range(num_task):

        y_true_pos = label[label[:,i]==1,i]
        y_true_neg = label[label[:,i]!=1,i]
        y_pred = model_out[i].cpu().detach().numpy()
        y_pred_pos = y_pred[label[:,i]==1]
        y_pred_neg = y_pred[label[:,i]!=1]

        for iter in range(iterations):

            pos_num = len(label[:,i]==1)
            pos_idx = np.random.randint(0,len(y_true_pos),pos_num)
            neg_idx = np.random.randint(0, len(y_true_neg),pos_num*ratio[i])

            y_true = np.concatenate([y_true_pos[pos_idx], y_true_neg[neg_idx]])
            y_score = np.concatenate([y_pred_pos[pos_idx], y_pred_neg[neg_idx]])

            y_pred_label = y_score > 0.5
            acc = np.mean(y_true==y_pred_label)
            auc = roc_auc_score(y_true,y_score)
            ap = average_precision_score(y_true,y_score)

            fprs, tprs, thresholds = roc_curve(y_true, y_score)

            precisions, recalls, _ = precision_recall_curve(y_true, y_score)

            metrics[i]['acc'].append(acc)
            metrics[i]['auc'].append(auc)
            metrics[i]['ap'].append(ap)
            metrics[i]['fprs'] = fprs.tolist()
            metrics[i]['tprs'] = tprs.tolist()
            metrics[i]['precisions'] = precisions.tolist()
            metrics[i]['recalls'] = recalls.tolist()

    metrics_avg = dict()
    metrics_avg['acc'] = [np.mean(metrics[i]['acc']) for i in range(num_task)]
    metrics_avg['auc'] = [np.mean(metrics[i]['auc']) for i in range(num_task)]
    metrics_avg['ap'] = [np.mean(metrics[i]['ap']) for i in range(num_task)]

    return metrics, metrics_avg
