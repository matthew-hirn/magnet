import os, sys

epochs = '3000'
for data in [
             'WebKB/Cornell/', 'WebKB/Texas/', 'WebKB/Wisconsin/', 
             'Email/', 
             'WikipediaNetwork/Chameleon/', 
             'WikipediaNetwork/Squirrel/',
             'Cora/', 
             'Citeseer/', 
             'WikiCS/', 
            ]:
    drop_prob = '0.5'
    if data in ['WikiCS/']:
        drop_prob = '0.8'

    for lr in [1e-3]:
        log_path = 'Edge_'+data[:-1]
        for num_filter in [16, 32, 64]:
            for q in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
                command = ('python Edge_MagNet.py ' 
                            +' --dataset='+data
                            +' --q='+str(q)
                            +' --num_filter='+str(num_filter)
                            +' --K=1'
                            +' -d'
                            +' --log_path='+str(log_path)
                            +' --layer=2'
                            +' --epochs='+epochs
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' -a')
                print(command)
                os.system(command)

        log_path = 'Edge_'+data[:-1]+'_SymDiGCN'
        for num_filter in [5, 15, 30]:
            command = ('python Edge_SymDiGCN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --drop_prob='+drop_prob
                        +' --lr='+str(lr)
                        +' -d'
                        +' --epochs='+epochs)
            print(command)
            os.system(command)

        log_path = 'Edge_'+data[:-1]+'_Cheb'
        for num_filter in [16, 32, 64]:
                command = ('python Edge_Cheb.py ' 
                            +' --dataset='+data
                            +' --K=2'
                            +' -d'
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)

        log_path = 'Edge_'+data[:-1]+'_GCN'
        for num_filter in [16, 32, 64]:
                command = ('python Edge_GCN.py ' 
                            +' --dataset='+data
                            +' -d'
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --epochs='+epochs
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' -tud')
                print(command)
                os.system(command)
                command = ('python Edge_GCN.py ' 
                            +' --dataset='+data
                            +' -d'
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --epochs='+epochs
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr))
                print(command)
                os.system(command)
        
        log_path = 'Edge_'+data[:-1]+'_SAGE'
        for num_filter in [16, 32, 64]:
                command = ('python Edge_SAGE.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)
                command = ('python Edge_SAGE.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --epochs='+epochs
                        +' --drop_prob='+drop_prob
                        +' --lr='+str(lr)
                        +' -tud')
                print(command)
                os.system(command)
        
        log_path = 'Edge_'+data[:-1]+'_GAT'
        for heads in [2, 4, 8]:
            for num_filter in [16, 32, 64]:
                command = ('python Edge_GAT.py ' 
                            +' --dataset='+data
                            +' --heads='+str(heads)
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command) 
                command = ('python Edge_GAT.py ' 
                            +' --dataset='+data
                            +' --heads='+str(heads)
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' -tud')
                print(command)
                os.system(command) 

        
        log_path = 'Edge_'+data[:-1]+'_GIN'
        for num_filter in [16, 32, 64]:
                command = ('python Edge_GIN.py ' 
                            +' --dataset='+data
                            +' -d'
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)
 
        log_path = 'Edge_'+data[:-1]+'_APPNP'
        for num_filter in [16, 32, 48]:
            for alpha in [0.05, 0.1, 0.15, 0.2]: 
                for K in [1, 5, 10]: 
                    command = ('python Edge_APPNP.py ' 
                                +' --dataset='+data
                                +' -d'
                                +' --num_filter='+str(num_filter)
                                +' --log_path='+str(log_path)
                                +' --drop_prob='+drop_prob
                                +' --lr='+str(lr)
                                +' --K='+str(K)
                                +' --alpha='+str(alpha)
                                +' --epochs='+epochs)
                    print(command)
                    os.system(command)
                    command = ('python Edge_APPNP.py ' 
                                +' --dataset='+data
                                +' -d'
                                +' --num_filter='+str(num_filter)
                                +' --log_path='+str(log_path)
                                +' --epochs='+epochs
                                +' --drop_prob='+drop_prob
                                +' --lr='+str(lr)
                                +' --K='+str(K)
                                +' --alpha='+str(alpha)
                                +' -tud')
                    print(command)
                    os.system(command)
 
        log_path = 'Edge_'+data[:-1]+'_Digraph'
        for num_filter in [16, 32, 64]:
            for alpha in [0.1, 0.05, 0.15, 0.2]: 
                command = ('python Edge_Digraph.py ' 
                            +' --dataset='+data
                            +' -d'
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --drop_prob='+drop_prob
                            +' --lr='+str(lr)
                            +' --alpha='+str(alpha)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)