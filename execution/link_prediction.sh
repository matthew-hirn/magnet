
cd ../src

# task 1
../../parallel -j4 --resume-failed --results ../Output/LinkPred1GCN    --joblog ../joblog/LinkPred1GCN    CUDA_VISIBLE_DEVICES=0 python ./Edge_GCN.py          --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred1Cheb   --joblog ../joblog/LinkPred1Cheb   CUDA_VISIBLE_DEVICES=0 python ./Edge_Cheb.py  --K 2  --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred1GIN    --joblog ../joblog/LinkPred1GIN    CUDA_VISIBLE_DEVICES=0 python ./Edge_GIN.py          --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred1SAGE   --joblog ../joblog/LinkPred1SAGE   CUDA_VISIBLE_DEVICES=0 python ./Edge_SAGE.py         --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j2 --resume-failed --results ../Output/LinkPred1Sym    --joblog ../joblog/LinkPred1Sym    CUDA_VISIBLE_DEVICES=0 python ./Edge_SymDiGCN.py     --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred1APPNP  --joblog ../joblog/LinkPred1APPNP  CUDA_VISIBLE_DEVICES=0 python ./Edge_APPNP.py        --epochs 3000 --num_filter {1} --dataset {2} --alpha {3} --K {4}               --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2 ::: 1 5 10 

../../parallel -j2 --resume-failed --results ../Output/LinkPred1DiG    --joblog ../joblog/LinkPred1DiG    CUDA_VISIBLE_DEVICES=0 python ./Edge_Digraph.py      --epochs 3000 --num_filter {1} --dataset {2} --alpha {3}                       --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2  

../../parallel -j2 --resume-failed --results ../Output/LinkPred1DiGib  --joblog ../joblog/LinkPred1DiGib  CUDA_VISIBLE_DEVICES=0 python ./Edge_Digraph.py      --epochs 3000 --num_filter {1} --dataset {2} --alpha {3} --method_name DiG_ib  --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 :::  6 11 21 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2

../../parallel -j2 --resume-failed --results ../Output/LinkPred1GAT    --joblog ../joblog/LinkPred1GAT    CUDA_VISIBLE_DEVICES=0 python ./Edge_GAT.py          --epochs 3000 --num_filter {1} --dataset {2} --heads {3}   --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 2 4 8  

../../parallel -j4 --resume-failed --results ../Output/LinkPred1Mag    --joblog ../joblog/LinkPred1Mag    CUDA_VISIBLE_DEVICES=0 python ./Edge_sparseMagnet.py --epochs 3000 --num_filter {1} --dataset {2} --q {3} --K=1 --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2 0.25

# task 2
../../parallel -j4 --resume-failed --results ../Output/LinkPred2GCN    --joblog ../joblog/LinkPred2GCN    CUDA_VISIBLE_DEVICES=0 python ./Edge_GCN.py          --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred2Cheb   --joblog ../joblog/LinkPred2Cheb   CUDA_VISIBLE_DEVICES=0 python ./Edge_Cheb.py  --K 2  --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred2GIN    --joblog ../joblog/LinkPred2GIN    CUDA_VISIBLE_DEVICES=0 python ./Edge_GIN.py          --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred2SAGE   --joblog ../joblog/LinkPred2SAGE   CUDA_VISIBLE_DEVICES=0 python ./Edge_SAGE.py         --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j2 --resume-failed --results ../Output/LinkPred2Sym    --joblog ../joblog/LinkPred2Sym    CUDA_VISIBLE_DEVICES=0 python ./Edge_SymDiGCN.py     --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred2APPNP  --joblog ../joblog/LinkPred2APPNP  CUDA_VISIBLE_DEVICES=0 python ./Edge_APPNP.py        --epochs 3000 --num_filter {1} --dataset {2} --alpha {3} --K {4}               --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2 ::: 1 5 10 

../../parallel -j2 --resume-failed --results ../Output/LinkPred2DiG    --joblog ../joblog/LinkPred2DiG    CUDA_VISIBLE_DEVICES=0 python ./Edge_Digraph.py      --epochs 3000 --num_filter {1} --dataset {2} --alpha {3}                       --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2  

../../parallel -j2 --resume-failed --results ../Output/LinkPred2DiGib  --joblog ../joblog/LinkPred2DiGib  CUDA_VISIBLE_DEVICES=0 python ./Edge_Digraph.py      --epochs 3000 --num_filter {1} --dataset {2} --alpha {3} --method_name DiG_ib  --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 :::  6 11 21 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2

../../parallel -j2 --resume-failed --results ../Output/LinkPred2GAT    --joblog ../joblog/LinkPred2GAT    CUDA_VISIBLE_DEVICES=0 python ./Edge_GAT.py          --epochs 3000 --num_filter {1} --dataset {2} --heads {3}   --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 2 4 8  

../../parallel -j4 --resume-failed --results ../Output/LinkPred2Mag    --joblog ../joblog/LinkPred2Mag    CUDA_VISIBLE_DEVICES=0 python ./Edge_sparseMagnet.py --epochs 3000 --num_filter {1} --dataset {2} --q {3} --K=1 --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --log_path LinkPred2 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2 0.25


# task 2 --num_class_link 3
../../parallel -j4 --resume-failed --results ../Output/LinkPred23GCN    --joblog ../joblog/LinkPred23GCN    CUDA_VISIBLE_DEVICES=0 python ./Edge_GCN.py          --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred23Cheb   --joblog ../joblog/LinkPred23Cheb   CUDA_VISIBLE_DEVICES=0 python ./Edge_Cheb.py  --K 2  --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred23GIN    --joblog ../joblog/LinkPred23GIN    CUDA_VISIBLE_DEVICES=0 python ./Edge_GIN.py          --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred23SAGE   --joblog ../joblog/LinkPred23SAGE   CUDA_VISIBLE_DEVICES=0 python ./Edge_SAGE.py         --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j2 --resume-failed --results ../Output/LinkPred23Sym    --joblog ../joblog/LinkPred23Sym    CUDA_VISIBLE_DEVICES=0 python ./Edge_SymDiGCN.py     --epochs 3000 --num_filter {1} --dataset {2} --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/   

../../parallel -j4 --resume-failed --results ../Output/LinkPred23APPNP  --joblog ../joblog/LinkPred23APPNP  CUDA_VISIBLE_DEVICES=0 python ./Edge_APPNP.py        --epochs 3000 --num_filter {1} --dataset {2} --alpha {3} --K {4}               --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2 ::: 1 5 10 

../../parallel -j2 --resume-failed --results ../Output/LinkPred23DiG    --joblog ../joblog/LinkPred23DiG    CUDA_VISIBLE_DEVICES=0 python ./Edge_Digraph.py      --epochs 3000 --num_filter {1} --dataset {2} --alpha {3}                       --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2  

../../parallel -j2 --resume-failed --results ../Output/LinkPred23DiGib  --joblog ../joblog/LinkPred23DiGib  CUDA_VISIBLE_DEVICES=0 python ./Edge_Digraph.py      --epochs 3000 --num_filter {1} --dataset {2} --alpha {3} --method_name DiG_ib  --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 :::  6 11 21 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2

../../parallel -j2 --resume-failed --results ../Output/LinkPred23GAT    --joblog ../joblog/LinkPred23GAT    CUDA_VISIBLE_DEVICES=0 python ./Edge_GAT.py          --epochs 3000 --num_filter {1} --dataset {2} --heads {3}   --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 2 4 8  

../../parallel -j4 --resume-failed --results ../Output/LinkPred23Mag    --joblog ../joblog/LinkPred23Mag    CUDA_VISIBLE_DEVICES=0 python ./Edge_sparseMagnet.py --epochs 3000 --num_filter {1} --dataset {2} --q {3} --K=1 --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 2 --num_class_link 3 --log_path LinkPred23 ::: 16 32 64 ::: WebKB/Cornell WebKB/Texas WebKB/Wisconsin cora_ml/ citeseer/  ::: 0.05 0.1 0.15 0.2 0.25
