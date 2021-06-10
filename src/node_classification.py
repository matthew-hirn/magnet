import os, sys

epochs = '3000'
for data in [
             'WebKB/Cornell', 'WebKB/Texas', 'WebKB/Wisconsin',
             'cora_ml/', 
             'citeseer_npz/',
             'syn/syn1', 
             'syn/syn2', 
             'syn/syn3'
            ]:
    for lr in [1e-3, 1e-2, 5e-3]:
        # MagNet
        log_path = data
        for num_filter in [16, 32, 64]:
            for q in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
                command = ('python sparse_MagNet.py ' 
                            +' --dataset='+data
                            +' --q='+str(q)
                            +' --num_filter='+str(num_filter)
                            +' --K=1'
                            +' --log_path='+str(log_path)
                            +' --layer=2'
                            +' --epochs='+epochs
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' -a')
                print(command)
                os.system(command)
        
        log_path = 'Sym_' + data
        for num_filter in [5, 15, 30]:
            command = ('python Sym_DiGCN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)
                
        log_path = 'GCN_' + data
        for num_filter in [16, 32, 64]:
            command = ('python GCN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)
            command = ('python GCN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --epochs='+epochs
                        +' --lr='+str(lr)
                        +' -tud')
            print(command)
            os.system(command)

        log_path = 'Cheb_' + data
        for num_filter in [16, 32, 64]:
            command = ('python Cheb.py ' 
                        +' --dataset='+data
                        +' --K=2'
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)

        log_path = 'SAGE_' + data
        for num_filter in [16, 32, 64]:
            command = ('python SAGE.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)
            command = ('python SAGE.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' -tud'
                        +' --epochs='+epochs)
            print(command)
            os.system(command)

        log_path = 'GAT_' + data
        for heads in [2, 4, 8]:
            for num_filter in [16, 32, 64]:
                command = ('python GAT.py ' 
                            +' --dataset='+data
                            +' --heads='+str(heads)
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)
                command = ('python GAT.py ' 
                            +' --dataset='+data
                            +' --heads='+str(heads)
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' -tud'
                            +' --epochs='+epochs)
                print(command)
                os.system(command)

        log_path = 'GIN_' + data
        for num_filter in [16, 32, 64]:
            command = ('python GIN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)
            command = ('python GIN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' -tud'
                        +' --epochs='+epochs)
            print(command)
            os.system(command)

        # K=10 following the original paper
        log_path = 'APPNP_' + data
        for num_filter in [16, 32, 64]:
            for alpha in [0.05, 0.1, 0.15, 0.2]: 
                command = ('python APPNP.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)
                command = ('python APPNP.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --epochs='+epochs
                            +' --lr='+str(lr)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' -tud')
                print(command)
                os.system(command)
        
        log_path = 'DiG_' + data
        for num_filter in [16, 32, 64]:
            for alpha in [0.05, 0.1, 0.15, 0.2]: 
                command = ('python Digraph.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)
    