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
from utils.preprocess import geometric_dataset, load_syn
from utils.save_settings import write_log
from utils.hermitian import hermitian_decomp
from utils.symmetric_distochastic import desymmetric_stochastic

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="MagNet Conv.")

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')

    parser.add_argument('--epochs', type=int, default=3000, help='training epochs')
    parser.add_argument('--q', type=float, default=0, help='q value for the phase matrix')
    parser.add_argument('--seed', type=int, default=0, help='random seed for training testing split/random graph generation')

    parser.add_argument('--K', type=int, default=2, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='how many layers of gcn in the model, only 1 or 2 layers.')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    
    parser.add_argument('-activation', '-a', action='store_true', help='if use activation function')
    parser.add_argument('--num_filter', type=int, default=1, help='num of filters')
    parser.add_argument('--to_radians', type=str, default='none', help='if transform real and imaginary numbers to modulus and radians')
    return parser.parse_args()
'''
def reset_model(model):
    del model
    torch.cuda.empty_cache()
    return
'''
def main(args):
    
    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, date_time)

    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    if load_func == 'WebKB':
        load_func = WebKB
    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
    elif load_func == 'WikiCS':
        load_func = WikiCS
    elif load_func == 'cora_ml':
        load_func = citation_datasets
    elif load_func == 'citeseer_npz':
        load_func = citation_datasets
    elif load_func == 'syn':
        load_func = load_syn

    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)

    _file_ = args.data_path+args.dataset+'/data'+str(args.q)+'_'+str(args.K)+'.pk'
    if os.path.isfile(_file_):
        data = pk.load(open(_file_, 'rb')) 
        L = data['L']
        X, label, train_mask, val_mask, test_mask = geometric_dataset(args.q, args.K, root=args.data_path+args.dataset, subset=subset,
                                dataset = load_func, load_only = True)
    else:
        X, label, train_mask, val_mask, test_mask, L = geometric_dataset(args.q, args.K, root=args.data_path+args.dataset, subset=subset,
                                dataset = load_func, load_only = False, save_pk = True)

    # normalize label, the minimum should be 0 as class index
    _label_ = label - np.amin(label)
    cluster_dim = np.amax(_label_)+1
    
    L_img = torch.from_numpy(L.imag).float().to(device)
    L_real = torch.from_numpy(L.real).float().to(device)
    label = torch.from_numpy(_label_[np.newaxis]).to(device)

    X_img = torch.from_numpy(X[np.newaxis]).float().to(device)
    X_real = torch.from_numpy(X[np.newaxis]).float().to(device)
  
    criterion = nn.NLLLoss()

    splits = train_mask.shape[1]
    if len(test_mask.shape) == 1:
        #data.test_mask = test_mask.unsqueeze(1).repeat(1, splits)
        test_mask = np.repeat(test_mask[:,np.newaxis], splits, 1)

    for split in range(splits):
        graphmodel = ChebNet(X_real.size(-1), L_real, L_img, K = args.K, label_dim=cluster_dim, layer = args.layer,
                                activation = args.activation, num_filter = args.num_filter, to_radians = args.to_radians, dropout=args.dropout)    
        model = nn.DataParallel(graphmodel)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_test_acc = 0.0
        train_index = train_mask[:,split]
        val_index = val_mask[:,split]
        test_index = test_mask[:,split]

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
            count, train_loss, train_acc = 0.0, 0.0, 0.0

            # for loop for batch loading
            count += int(X_real.size(0)*np.sum(train_index))

            model.train()
            preds = model(X_real, X_img)
            train_loss = criterion(preds[:,:,train_index], label[:,train_index])

            pred_label = preds.max(dim = 1)[1]
            train_acc = 1.0*((pred_label[:,train_index] == label[:,train_index])).sum().detach().item()/count

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            outstrtrain = 'Train loss: %.6f, acc: %.3f' % (train_loss.detach().item(), train_acc)
            #scheduler.step()
            ####################
            # Validation
            ####################
            model.eval()
            count, test_loss, test_acc = 0.0, 0.0, 0.0
            
            # for loop for batch loading
            count += int(X_real.size(0)*np.sum(val_index))
            preds = model(X_real, X_img)
            pred_label = preds.max(dim = 1)[1]
            
            test_loss = criterion(preds[:,:,val_index], label[:,val_index])
            test_acc = 1.0*((pred_label[:,val_index] == label[:,val_index])).sum().detach().item()/count

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
        preds = model(X_real, X_img)
        pred_label = preds.max(dim = 1)[1]
    
        count = int(X_real.size(0)*np.sum(val_index))
        acc_train = (1.0*((pred_label[:,val_index] == label[:,val_index])).sum().detach().item())/count

        count = int(X_real.size(0)*np.sum(test_index))
        acc_test = (1.0*((pred_label[:,test_index] == label[:,test_index])).sum().detach().item())/count

        model.load_state_dict(torch.load(log_path + '/model_latest'+str(split)+'.t7'))
        model.eval()
        preds = model(X_real, X_img)
        pred_label = preds.max(dim = 1)[1]
    
        count = int(X_real.size(0)*np.sum(val_index))
        acc_train_latest = (1.0*((pred_label[:,val_index] == label[:,val_index])).sum().detach().item())/count

        count = int(X_real.size(0)*np.sum(test_index))
        acc_test_latest = (1.0*((pred_label[:,test_index] == label[:,test_index])).sum().detach().item())/count

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