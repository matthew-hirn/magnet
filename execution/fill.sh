cd ../src

# Note: device could be changed, and lines could be split into different scripts

../../parallel -j10 --resume-failed --results ../Output/fill_Cheb --joblog ../joblog/fill_Cheb_joblog  CUDA_VISIBLE_DEVICES=0 python ./Cheb.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j20 --resume-failed --results ../Output/fill_DGCN --joblog ../joblog/fill_DGCN_joblog  CUDA_VISIBLE_DEVICES=4 python ./DGCN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j10 --resume-failed --results ../Output/fill_Digraph --joblog ../joblog/fill_Digraph_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j10 --resume-failed --results ../Output/fill_DigraphIB --joblog ../joblog/fill_DigraphIB_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py --method_name DiG_ib --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j10 --resume-failed --results ../Output/fill_APPNP --joblog ../joblog/fill_APPNP_joblog  CUDA_VISIBLE_DEVICES=5 python ./APPNP.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j4 --resume-failed --results ../Output/fill_GAT --joblog ../joblog/fill_GAT_joblog  CUDA_VISIBLE_DEVICES=3 python ./GAT.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j10 --resume-failed --results ../Output/fill_GCN --joblog ../joblog/fill_GCN_joblog  CUDA_VISIBLE_DEVICES=7 python ./GCN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j10 --resume-failed --results ../Output/fill_GIN --joblog ../joblog/fill_GIN_joblog  CUDA_VISIBLE_DEVICES=6 python ./GIN.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j10 --resume-failed --results ../Output/fill_magnet1 --joblog ../joblog/fill_magnet1_joblog  CUDA_VISIBLE_DEVICES=1 python ./sparse_Magnet.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --K 1 --dropout 0.5 -a ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8

../../parallel -j10 --resume-failed --results ../Output/fill_SAGE --joblog ../joblog/fill_SAGE_joblog  CUDA_VISIBLE_DEVICES=6 python ./SAGE.py  --epochs 3000 --seed {1} --p_q {2} --log_path syn --dataset syn/fill --dropout 0.5 ::: 10 20 30 40 ::: 0.95 0.9 0.85 0.8