cd ../src

# Note: device could be changed, and lines could be split into different scripts

../../parallel -j10 --resume-failed --results ../Output/ordered_Cheb --joblog ../joblog/ordered_Cheb_joblog  CUDA_VISIBLE_DEVICES=0 python ./Cheb.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j20 --resume-failed --results ../Output/ordered_DGCN --joblog ../joblog/ordered_DGCN_joblog  CUDA_VISIBLE_DEVICES=4 python ./DGCN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j10 --resume-failed --results ../Output/ordered_Digraph --joblog ../joblog/ordered_Digraph_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j10 --resume-failed --results ../Output/ordered_DigraphIB --joblog ../joblog/ordered_DigraphIB_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py --method_name DiG_ib --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j10 --resume-failed --results ../Output/ordered_APPNP --joblog ../joblog/ordered_APPNP_joblog  CUDA_VISIBLE_DEVICES=5 python ./APPNP.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j4 --resume-failed --results ../Output/ordered_GAT --joblog ../joblog/ordered_GAT_joblog  CUDA_VISIBLE_DEVICES=3 python ./GAT.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j10 --resume-failed --results ../Output/ordered_GCN --joblog ../joblog/ordered_GCN_joblog  CUDA_VISIBLE_DEVICES=7 python ./GCN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j10 --resume-failed --results ../Output/ordered_GIN --joblog ../joblog/ordered_GIN_joblog  CUDA_VISIBLE_DEVICES=6 python ./GIN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j10 --resume-failed --results ../Output/ordered_magnet1 --joblog ../joblog/ordered_magnet1_joblog  CUDA_VISIBLE_DEVICES=1 python ./sparse_Magnet.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --K 1 --dropout 0.5 -a ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05

../../parallel -j10 --resume-failed --results ../Output/ordered_SAGE --joblog ../joblog/ordered_SAGE_joblog  CUDA_VISIBLE_DEVICES=6 python ./SAGE.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/syn --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 -0.08 -0.05