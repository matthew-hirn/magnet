cd ../src

# Note: device could be changed, and lines could be split into different scripts

../../parallel -j10 --resume-failed --results ../Output/cyclic_Cheb --joblog ../joblog/cyclic_Cheb_joblog  CUDA_VISIBLE_DEVICES=0 python ./Cheb.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j20 --resume-failed --results ../Output/cyclic_DGCN --joblog ../joblog/cyclic_DGCN_joblog  CUDA_VISIBLE_DEVICES=4 python ./DGCN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j10 --resume-failed --results ../Output/cyclic_Digraph --joblog ../joblog/cyclic_Digraph_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j10 --resume-failed --results ../Output/cyclic_DigraphIB --joblog ../joblog/cyclic_DigraphIB_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py --method_name DiG_ib --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j10 --resume-failed --results ../Output/cyclic_APPNP --joblog ../joblog/cyclic_APPNP_joblog  CUDA_VISIBLE_DEVICES=5 python ./APPNP.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j4 --resume-failed --results ../Output/cyclic_GAT --joblog ../joblog/cyclic_GAT_joblog  CUDA_VISIBLE_DEVICES=3 python ./GAT.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j10 --resume-failed --results ../Output/cyclic_GCN --joblog ../joblog/cyclic_GCN_joblog  CUDA_VISIBLE_DEVICES=7 python ./GCN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j10 --resume-failed --results ../Output/cyclic_GIN --joblog ../joblog/cyclic_GIN_joblog  CUDA_VISIBLE_DEVICES=6 python ./GIN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j10 --resume-failed --results ../Output/cyclic_magnet1 --joblog ../joblog/cyclic_magnet1_joblog  CUDA_VISIBLE_DEVICES=1 python ./sparse_Magnet.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --K 1 --dropout 0.5 -a ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7

../../parallel -j10 --resume-failed --results ../Output/cyclic_SAGE --joblog ../joblog/cyclic_SAGE_joblog  CUDA_VISIBLE_DEVICES=6 python ./SAGE.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/cyclic --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7