# external files
import numpy as np
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
from collections import Counter
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS

# internal files
from layer.cheb import *
from utils.Citation import *
from layer.geometric_baselines import *
from torch_geometric.utils import to_undirected
from utils.preprocess import geometric_dataset, load_syn
from utils.save_settings import write_log
from utils.hermitian import hermitian_decomp
from utils.symmetric_distochastic import desymmetric_stochastic

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="baseline--chebnet.")

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')

    parser.add_argument('-K', '--K', default=2, type=int)

    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=2, help='num of filters')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    #parser.add_argument('-to_undirected', '-tud', action='store_true', help='if convert graph to undirecteds')
    return parser.parse_args()

def acc(pred, label, mask):
    correct = int(pred[mask].eq(label[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc

def main(args):
    
    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, date_time)

    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    if load_func == 'WebKB':
        load_func = WebKB
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikiCS':
        load_func = WikiCS
        dataset = load_func(root=args.data_path)
    elif load_func == 'cora_ml':
        dataset = citation_datasets(root='../dataset/data/tmp/cora_ml/cora_ml.npz')
    elif load_func == 'citeseer_npz':
        dataset = citation_datasets(root='../dataset/data/tmp/citeseer_npz/citeseer_npz.npz')
    elif load_func == 'syn':
        dataset = load_syn(args.data_path + args.dataset, None)

    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    data.y = data.y.long()
    num_classes = (data.y.max() - data.y.min() + 1).detach().numpy()
    data = data.to(device)
    # normalize label, the minimum should be 0 as class index
    splits = data.train_mask.shape[1]
    if len(data.test_mask.shape) == 1:
        data.test_mask = data.test_mask.unsqueeze(1).repeat(1, splits)

    for split in range(splits):
        graphmodel = ChebModel(data.x.size(-1), num_classes, K=args.K, filter_num=args.num_filter, dropout=args.dropout)    
        model = nn.DataParallel(graphmodel)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            if early_stopping > 500:
                break
            ####################
            # Train
            ####################
            train_loss, train_acc = 0.0, 0.0

            # for loop for batch loading
            model.train()
            out = model(data)

            train_loss = F.nll_loss(out[data.train_mask[:,split]], data.y[data.train_mask[:,split]])
            pred_label = out.max(dim = 1)[1]
            train_acc = acc(pred_label, data.y, data.train_mask[:,split])

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            outstrtrain = 'Train loss: %.6f, acc: %.3f' % (train_loss.detach().item(), train_acc)
            #scheduler.step()
            
            ####################
            # Validation
            ####################
            model.eval()
            test_loss, test_acc = 0.0, 0.0
            
            out = model(data)
            pred_label = out.max(dim = 1)[1]            

            test_loss = F.nll_loss(out[data.val_mask[:,split]], data.y[data.val_mask[:,split]])
            test_acc = acc(pred_label, data.y, data.val_mask[:,split])

            outstrval = ' Test loss: %.6f, acc: %.3f' % (test_loss.detach().item(), test_acc)
            
            duration = "--- %.4f seconds ---" % (time.time() - start_time)
            log_str = ("%d / %d epoch" % (epoch, args.epochs))+outstrtrain+outstrval+duration
            #print(log_str)

            ####################
            # Save log csv
            ####################
            status = 'w'
            if os.path.isfile(log_path + '/log'+str(split)+'.csv'):
                status = 'a'
            with open(log_path + '/log'+str(split)+'.csv', status) as file:
                file.write(log_str)
                file.write('\n')

            ####################
            # Save weights
            ####################
            torch.save(model.state_dict(), log_path + '/model_latest'+str(split)+'.t7')
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model'+str(split)+'.t7')
            else:
                early_stopping += 1

        write_log(vars(args), log_path)

        ####################
        # Testing
        ####################
        model.load_state_dict(torch.load(log_path + '/model'+str(split)+'.t7'))
        model.eval()
        preds = model(data)
        pred_label = preds.max(dim = 1)[1]
    
        acc_train = acc(pred_label, data.y, data.val_mask[:,split])
        acc_test = acc(pred_label, data.y, data.test_mask[:,split])

        model.load_state_dict(torch.load(log_path + '/model_latest'+str(split)+'.t7'))
        model.eval()
        preds = model(data)
        pred_label = preds.max(dim = 1)[1]
    
        acc_train_latest = acc(pred_label, data.y, data.val_mask[:,split])
        acc_test_latest = acc(pred_label, data.y, data.test_mask[:,split])

        ####################
        # Save testing results
        ####################
        logstr = 'val_acc: '+str(np.round(acc_train, 3))+' test_acc: '+str(np.round(acc_test,3))+' val_acc_latest: '+str(np.round(acc_train_latest,3))+' test_acc_latest: '+str(np.round(acc_test_latest,3))
        print(logstr)
        with open(log_path + '/log'+str(split)+'.csv', status) as file:
            file.write(logstr)
            file.write('\n')
        torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)