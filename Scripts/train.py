def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cal_loss_weight(dataset, beta=0.99999):
    data, label = dataset[:]
    total_example = label.shape[0]
    num_task = label.shape[1]
    labels_dict = dict(zip(range(num_task),[sum(label[:,i]) for i in range(num_task)]))
    keys = labels_dict.keys()
    class_weight = dict()

    # Class-Balanced Loss Based on Effective Number of Samples
    for key in keys:
        effective_num = 1.0 - beta**labels_dict[key]
        weights = (1.0 - beta) / effective_num
        class_weight[key] = weights

    weights_sum = sum(class_weight.values())

    # normalizing weights
    for key in keys:
        class_weight[key] = class_weight[key] / weights_sum * num_task

    return class_weight

def naive_loss(y_pred, y_true, loss_weight=None,ohem=False,focal=False):

    num_task = y_true.shape[-1]
    num_examples = y_true.shape[0]
    k = 0.7

    def binary_cross_entropy(x, y,focal=False):
        alpha = 0.75
        gamma = 2

        pt = x * y + (1 - x) * (1 - y)
        at = alpha * y + (1 - alpha)* (1 - y)

        # focal loss
        if focal:
            loss = -at*(1-pt)**(gamma)*(torch.log(x) * y + torch.log(1 - x) * (1 - y))
        else:
            loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
        return loss
    # loss = nn.BCELoss(reduction='sum') fail to double backwards
    loss_output = torch.zeros(num_examples).cuda()
    for i in range(num_task):
        if loss_weight:
            out = loss_weight[i]*binary_cross_entropy(y_pred[i],y_true[:,i],focal)
            loss_output += out
        else:
            loss_output += binary_cross_entropy(y_pred[i],y_true[:,i],focal)

    # loss = nn.MultiLabelSoftMarginLoss(weight=loss_weight,reduction='sum')
    # loss_output = loss(y_pred, y_true)

    # Online Hard Example Mining
    if ohem:
        val, idx = torch.topk(loss_output,int(k*num_examples))
        loss_output[loss_output<val[-1]] = 0

    loss = torch.sum(loss_output)

    return loss


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model,test_loader,loss_weight,use_embedding,use_uncertain_weighting,MultiTaskLossWrapper=None):

    with torch.no_grad():
        model.eval()
        test_loss = 0
        metrics_dict = {"acc":0,
                       "auc":0,
                        "ap":0}
        for x, y_true in test_loader:
            x, y_true = x.cuda(), y_true.cuda()
            # resize input x from 1001 features
            if not use_embedding:
                x = x.view(x.size(0),-1,4).transpose(1,2)

            y_pred = model(x)
            if use_uncertain_weighting:
                MultiTaskLossWrapper.eval()
                test_loss += MultiTaskLossWrapper(y_pred,y_true)
            else:
                test_loss += naive_loss(y_pred,y_true,loss_weight)

            acc = 0
            auc = 0
            ap = 0
            num_task = len(y_pred)
            for i in range(num_task):
                label = y_true.cpu().numpy()[:,i]
                y_score = y_pred[i].cpu().detach().numpy()
                y_pred_single = np.array([0 if instance < 0.5 else 1 for instance in y_score])

                acc += np.mean(y_pred_single==label)

                try:
                    auc += roc_auc_score(label,y_score)
                except ValueError:
                    pass

                try:
                    ap += average_precision_score(label,y_score)
                except ValueError:
                    pass


            metrics_dict['acc'] += acc / num_task
            metrics_dict["auc"] += auc / num_task
            metrics_dict["ap"] += ap / num_task

        num_examples = len(test_loader.dataset)
        test_loss /= num_examples
        num_batches = num_examples // test_loader.batch_size + 1
        metrics_dict['acc'] /= num_batches
        metrics_dict['auc'] /= num_batches
        metrics_dict['ap'] /= num_batches

    return test_loss, metrics_dict

def train(model,train_loader,test_loader,args):
    """

    """
    from time import time

    import csv
    if not os.path.exists(args.save_dir):
        print('%s does not exist, create it now'%(args.save_dir)+'-'*30)
        os.mkdir(args.save_dir)
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile,
                 fieldnames=['epoch', 'loss', 'val_loss', 'val_acc',
                              'val_precision','val_recall'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(),lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_decay)

    #lr_decay = lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.t_max)
    best_val_acc = 0
    best_val_loss = 50
    loss_weight_ = cal_loss_weight(train_loader.dataset) # dictionary
    # loss_weight = torch.Tensor(list(loss_weight.values())).cuda() # pytorch tensor

    # parepare weights parameters
    loss_weight = []
    for i in range(args.num_task):
        # initialize weights
        loss_weight.append(torch.tensor(loss_weight_[i],requires_grad=True, device="cuda"))
        # loss_weight_temp = torch.tensor(loss_weight_[i])
        # loss_weight_temp = loss_weight_temp.cuda()
        # loss_weight_temp.requires_grad = True
        # loss_weight.append(loss_weight_temp)

    # parepare uncertain weighting
    if args.use_uncertain_weighting:
        MutiTaskLoss = MultiTaskLossWrapper(args.num_task).cuda()
    else:
        MutiTaskLoss = None

    # optimizer for Grad Norm
    # optimizer_2 = Adam(loss_weight,lr=0.001) deprecated grad norm

    alph = 0.16  # hyperparameter of GradNorm

    print('Begin Training'+'-' * 70)
    for epoch in range(args.epochs):
        model.train()
        ti = time()
        training_loss = 0.0
        coef = 0

        for i, (x, y_true) in enumerate(train_loader):
            x, y_true = x.cuda(), y_true.cuda()

            # A 1 0 0 0
            # C 0 1 0 0
            # T/U 0 0 0 1
            # G 0 0 1 0
            # N 0 0 0 0

            if not args.use_embedding:
                x = x.view(x.size(0),-1,4).transpose(1,2)


            y_pred = model(x)

            #print('%d-batch'%i, loss_weight[0].is_leaf)
            if args.use_uncertain_weighting:
                loss = MutiTaskLoss(y_pred,y_true)
            else:
                loss = naive_loss(y_pred,y_true,loss_weight,args.OHEM,args.focal_loss)


            optimizer.zero_grad()

            # gradient clipping
            clip_value = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)


            loss.backward()
            # print(loss_weight[0].is_leaf)
            optimizer.step()
            # print(loss_weight[0].is_leaf)

            training_loss += loss.data


            # print(loss_weight[0].is_leaf)

            # Renormalizing the losses weights
            # coef = args.num_task/sum([loss for loss in loss_weight])
            # loss_weight = [coef*loss for loss in loss_weight]

            if i == 1:
                #sanity check y_pred
                print("Sanity Checking, at epoch%02d, iter%02d, y_pred is"%(epoch,i),
                        [y_pred[j][1].cpu().detach() for j in range(args.num_task)])
                print("Learning rate: %.16f" % optimizer.state_dict()['param_groups'][0]['lr'] )
                    # print("Gradient of weight: ", torch.autograd.grad(Lgrad,loss_weight[0]).detach().cpu())
            # print(loss_weight)

        lr_decay.step()


        # adjust learning rate by hand
        # if 19 <= epoch < 39:
        #     adjust_learning_rate(optimizer,optimizer.state_dict()['param_groups'][0]['lr'] / 10)
        # elif 39 <= epoch < 59:
        #     adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 2))
        # elif 59 <= epoch < 79:
        #     adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 3))
        # elif epoch >= 79:
        #     adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 4))


        #compute validation loss and acc
        val_loss, metrics_dict = test(model,test_loader,loss_weight,args.use_embedding,
                                      args.use_uncertain_weighting,MutiTaskLoss)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss.cpu().numpy() / len(train_loader.dataset),
                                val_loss=val_loss.detach().cpu().numpy(), val_acc=metrics_dict['acc'],
                                val_recall=metrics_dict["auc"],
                                val_precision=metrics_dict['ap']))

        print("===>Epoch %02d: loss=%.5f, val_loss=%.4f, val_acc=%.4f,\
                val_auc=%.4f, val_ap=%.4f, time=%ds"
              %(epoch,training_loss/len(train_loader.dataset),val_loss,
                 metrics_dict["acc"], metrics_dict["auc"],metrics_dict["ap"],
                  time()-ti))
        if metrics_dict['acc'] > best_val_acc and val_loss < best_val_loss:  # update best validation acc and save model
            best_val_acc = metrics_dict["acc"]
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()

    torch.save(model.state_dict(), args.save_dir + '/trained_model_%02dseqs.pkl' % args.length)
    print('Trained model saved to \'%s/trained_model.h5\'' % (args.save_dir))
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)

    return model


if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Naive Network on RBP.")
    parser.add_argument('--inputs',default='../Data/data_12RM.h5',type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--length', default=101,type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.8, type=float,
                        help="The value multiplied by lr at each epoch.Set a larger value for larger epochs")
    parser.add_argument('--t_max',default=5, type=int)
    parser.add_argument('--save_dir', default='../Results')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--gpu', type=int, default=[6,7], nargs='+', help="used GPU")
    parser.add_argument('--num_task',default=12, type=int)
    parser.add_argument('--grad_norm',default=False, type=str2bool, nargs='?',
                         help='activate grad norm')
    parser.add_argument('--mode',default='train',
                        help="Set the model to train or not")
    parser.add_argument('--use_embedding',default=False, type=str2bool, nargs='?',
                         help='activate embedding')
    parser.add_argument('--use_uncertain_weighting',default=False, type=str2bool, nargs='?',
                         help='activate uncertain weighting')
    parser.add_argument('--balanced_sampler',default=False,type=str2bool,nargs='?')
    parser.add_argument('--OHEM',default=False,type=str2bool,nargs='?')
    parser.add_argument('--focal_loss',default=False,type=str2bool,nargs='?')
    parser.add_argument('--hmm',default=False,type=str2bool,nargs='?')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    from util_layers import *
    from models import *
    from train_utils import *



    import torch
    import numpy as np
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, lr_scheduler

    from sklearn.metrics import roc_auc_score, average_precision_score

    # define model
    # model = NaiveNet(input_size = args.length, num_task=args.num_task) # CNN only
    # model = NaiveNet_v1(input_size = args.length, num_task=args.num_task) # CNN + LSTM + Attention
    # model = NaiveNet_v2(input_size = args.length,num_task=args.num_task) # CNN + LSTM
    # model = nn.DataParallel(NaiveCaps_v1(num_task=args.num_task))
    model = model_v3(num_task=args.num_task,use_embedding=args.use_embedding)
    # model = nn.DataParallel(model_v3(num_task=args.num_task,use_embedding=args.use_embedding))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model)

    if args.weights is not None:  # init the model weights with provided one
        print('Loading weights from %s' % (args.weights) +'-' * 40)
        model.load_state_dict(torch.load(args.weights))


    if args.hmm == True:
        args.use_embedding = False
    # train or test
    if args.mode=='train':
        # load data
        train_loader, valid_loader = load_RM_data(args.inputs,batch_size=args.batch_size,
                                                  length=args.length,
                                                  use_embedding=args.use_embedding,
                                                  balanced_sampler=args.balanced_sampler)
        train(model,train_loader,valid_loader,args)
    else:
        print('Loading test data'+'-' * 70)
        test_data = RMdata(data_path=args.inputs, length=args.length,
                           use_embedding=args.use_embedding, mode=args.mode)
        x_test, y_test = test_data[:]
        torch.cuda.empty_cache()
        # x_test, y_test = x_test.cuda(), y_test.cuda()

        if not args.use_embedding:
            x_test = x_test.view(x_test.size(0),-1,4).transpose(1,2)

        print('Begin testing %s set'%(args.mode)+'-' * 70)
        model.eval()

        # handle with CUDA memory issue
        try:
            y_pred = model(x_test.cuda())
        except RuntimeError:
            print('Catch RuntimeError, parepare to batch the test set'+ '-'* 50)
            batch_size = 1
            num_iter = x_test.shape[0] // batch_size

            x_test_tem = x_test[0:1*batch_size,...]
            y_pred = model(x_test_tem.cuda())
            for i in range(1, num_iter):
                x_test_tem = x_test[i*batch_size:(i+1)*batch_size,...]
                y_pred_tem = model(x_test_tem.cuda())
                for j in range(args.num_task):
                    y_pred[j] = torch.cat((y_pred[j].cpu().detach(),y_pred_tem[j].cpu().detach()),dim=0)

        class_names = test_data.class_name
        model_name = 'gen2vec'
        # evaluate the model
        metrics, metrics_avg = cal_metrics(y_pred, y_test, plot=True, class_names=class_names, plot_name=model_name)

        # metrics, metrics_avg = cal_metrics_sampling(y_pred,y_test)

        # performances_df = pd.DataFrame.from_dict(metrics)
        # performances_df['names'] = pd.Series(class_names)
        # performances_df.set_index('names',inplace=True)
        # performances_df.to_csv('./pf.csv')

        print('End testing'+'-' * 70)
        print()
        print('-'*35+"Result"+"-"*35)
        # print outcome
        import prettytable as pt
        tb = pt.PrettyTable()
        tb.field_names = ["metrics"] + class_names
        for key, values in metrics.items():
            tb.add_row([key]+values)
        print(tb)

        pf_save_path = "%s/pf.json" %(args.save_dir)
        import json
        with open(pf_save_path,'w') as fp:
            json.dump(metrics, fp)
        print("Storing performance results to %s/pf.json" %(args.save_dir))
