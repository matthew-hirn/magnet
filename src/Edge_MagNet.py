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
from torch_geometric.utils import to_undirected

# internal files
from layer.cheb import *
from utils.edge_data import generate_dataset, in_out_degree
from utils.preprocess import geometric_dataset, to_edge_dataset, load_edge_index
from utils.save_settings import write_log

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction of MagNet")

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')
    parser.add_argument('--drop_prob', type=float, default=0.4, help='random drop for testing edges')

    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=4, help='num of filters')
    parser.add_argument('-not_norm', '-n', action='store_false', help='if use normalized laplacian or not, default: yes')
    parser.add_argument('-random_walk', '-rw', action='store_true', help='if use random walk based laplacian to replace magnetic laplacian')

    parser.add_argument('--q', type=float, default=0, help='q value for the phase matrix')
    parser.add_argument('--K', type=int, default=2, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='how many layers of gcn in the model, only 1 or 2 layers.')
    parser.add_argument('-activation', '-a', action='store_true', help='if use activation function')
    parser.add_argument('--to_radians', type=str, default='none', help='if transform real and imaginary numbers to modulus and radians')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')
    parser.add_argument('-dgrees', '-d', action='store_true', help='if use in degree+outdegree as feature')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    return parser.parse_args()

def acc(pred, label):
    #print(pred.shape, label.shape)       
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
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
    elif load_func == 'Cora':
        edge_index = load_edge_index(file = 'cora.edges', path = '../dataset/cora/')
    elif load_func == 'Citeseer':
        edge_index = load_edge_index(file = 'citeseer.edges', path = '../dataset/citeseer/')
    elif load_func == 'Email':
        edge_index = load_edge_index(file = 'email-Eu-core.edges', path = '../dataset/email/')
    elif load_func == 'HepPh':
        edge_index = load_edge_index(file = 'HepPh.edges', path = '../dataset/cite-Hep/')
    elif load_func == 'HepTh':
        edge_index = load_edge_index(file = 'HepTh.edges', path = '../dataset/cite-Hep/')

    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)
        
    # load dataset
    if load_func not in ['Cora', 'Citeseer', 'Email', 'HepPh', 'HepTh']:
        data = dataset[0]
        edge_index = data.edge_index

    size = torch.max(edge_index).item()+1
    # generate edge index dataset
    _file_ = args.data_path+args.dataset.split('/')[0]+'_'+subset+'_'+str(args.drop_prob)+'.pk'
    if os.path.isfile(_file_):
        datasets = pk.load(open(_file_, 'rb'))
    else:
        datasets = generate_dataset(edge_index, _file_, splits = 10, test_prob = args.drop_prob)

    #feature = torch.cat((dataset[0].x.data, torch.ones(dataset[0].x.data.shape[0]).unsqueeze(-1)), axis=-1)
    for i in range(10):
        ########################################
        # get hermitian laplacian
        ########################################
        edges = datasets[i]['train']['positive']
        L = to_edge_dataset(args.q, edges, args.K, i, size, root=args.data_path+args.dataset, laplacian=True, norm=args.not_norm)       
        L_img = torch.from_numpy(L.imag).float().cuda()
        L_real = torch.from_numpy(L.real).float().cuda()

        # get feature
        #X_img = feature.unsqueeze(0).to(device)
        #X_real = feature.unsqueeze(0).to(device)
        
        if args.dgrees == True:
            X_img = in_out_degree(edges, size).cuda()
            X_real = in_out_degree(edges, size).cuda()
        else:
            X_img  = torch.ones(L_real.shape[-1]).unsqueeze(0).unsqueeze(-1).cuda()
            X_real = torch.ones(L_real.shape[-1]).unsqueeze(0).unsqueeze(-1).cuda()
        ########################################
        # initialize model and load dataset
        ########################################
        model = ChebNet_Edge(X_real.size(-1), L_real, L_img, K = args.K, label_dim = 2, layer = args.layer,
                                activation = args.activation, num_filter = args.num_filter, to_radians = args.to_radians, dropout=args.dropout).to(device)
        #model = nn.DataParallel(model)  
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        y_train = np.r_[np.zeros(len(datasets[i]['train']['positive'].T)),
                        np.ones(len(datasets[i]['train']['negative'].T))].astype('int')
        y_val   = np.r_[np.zeros(len(datasets[i]['validate']['positive'].T)),
                        np.ones(len(datasets[i]['validate']['negative'].T))].astype('int')
        y_test  = np.r_[np.zeros(len(datasets[i]['test']['positive'].T)),
                        np.ones(len(datasets[i]['test']['negative'].T))].astype('int')
        y_train = torch.from_numpy(y_train).cuda()
        y_val   = torch.from_numpy(y_val).cuda()
        y_test  = torch.from_numpy(y_test).cuda()

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            #if early_stopping > 500:
            #    break
            ####################
            # Train
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.train()
            out = model(X_real, X_img, 
                        datasets[i]['train']['positive'].T, 
                        datasets[i]['train']['negative'].T)
            
            train_loss = F.nll_loss(out, y_train)
            pred_label = out.max(dim = 1)[1]            
            train_acc  = acc(pred_label, y_train)
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            outstrtrain = 'Train loss: %.6f, acc: %.3f' % (train_loss.detach().item(), train_acc)
            
            ####################
            # Validation
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.eval()
            out = model(X_real, X_img, 
                        datasets[i]['validate']['positive'].T, 
                        datasets[i]['validate']['negative'].T)

            test_loss  = F.nll_loss(out, y_val)
            pred_label = out.max(dim = 1)[1]          
            test_acc   = acc(pred_label, y_val)

            outstrval = ' Test loss: %.6f, acc: %.3f' % (test_loss.detach().item(), test_acc)            
            duration = "--- %.4f seconds ---" % (time.time() - start_time)
            log_str = ("%d / %d epoch" % (epoch, args.epochs))+outstrtrain+outstrval+duration
            #print(log_str)
           
            ####################
            # Save log csv
            ####################
            status = 'w'
            if os.path.isfile(log_path + '/log'+str(i)+'.csv'):
                status = 'a'
            with open(log_path + '/log'+str(i)+'.csv', status) as file:
                file.write(log_str)
                file.write('\n')

            ####################
            # Save weights
            ####################
            torch.save(model.state_dict(), log_path + '/model_latest'+str(i)+'.t7')
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model'+str(i)+'.t7')
            else:
                early_stopping += 1

        write_log(vars(args), log_path)

        ####################
        # Testing
        ####################
        model.load_state_dict(torch.load(log_path + '/model'+str(i)+'.t7'))
        model.eval()
        out = model(X_real, X_img, 
                    datasets[i]['test']['positive'].T, 
                    datasets[i]['test']['negative'].T)
        pred_label = out.max(dim = 1)[1]
        #test_acc   = acc(pred_label[:len(datasets[i]['test']['positive'].T)], y_test[:len(datasets[i]['test']['positive'].T)])
        test_acc = acc(pred_label, y_test)

        model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
        model.eval()
        out = model(X_real, X_img, 
                    datasets[i]['test']['positive'].T, 
                    datasets[i]['test']['negative'].T)
        pred_label = out.max(dim = 1)[1]
        #test_acc_least = acc(pred_label[:len(datasets[i]['test']['positive'].T)], y_test[:len(datasets[i]['test']['positive'].T)])
        test_acc_least = acc(pred_label, y_test)

        ####################
        # Save testing results
        ####################
        logstr = 'test_acc: '+str(np.round(test_acc,3))+' test_acc_latest: '+str(np.round(test_acc_least,3))

        print(logstr)
        with open(log_path + '/log'+str(i)+'.csv', status) as file:
            file.write(logstr)
            file.write('\n')
        torch.cuda.empty_cache()

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)