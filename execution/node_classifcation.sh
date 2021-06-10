cd ../src

# Note: device could be changed, and lines could be split into different scripts
.
../../parallel -j18 --resume-failed --results ../Output/magnet --joblog ../joblog/magnet_joblog  CUDA_VISIBLE_DEVICES=1 python ./sparse_Magnet.py  --epochs 3000 --lr {1} --num_filter {2} --q {3} --log_path telegram_magnet --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --K 1  --layer 2 --dropout 0.5 -a ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 0.0 0.05 0.1 0.15 0.2 0.25

../../parallel -j9 --resume-failed --results ../Output/Cheb --joblog ../joblog/Cheb_joblog  CUDA_VISIBLE_DEVICES=2 python ./Cheb.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_Cheb --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64

../../parallel -j9 --resume-failed --results ../Output/GCN --joblog ../joblog/GCN_joblog  CUDA_VISIBLE_DEVICES=2 python ./GCN.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_GCN --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64

../../parallel -j9 --resume-failed --results ../Output/GCN_tud --joblog ../joblog/GCN_tud_joblog  CUDA_VISIBLE_DEVICES=2 python ./GCN.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_GCN_tud --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ -tud --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64

../../parallel -j9 --resume-failed --results ../Output/Sym_DiGCN --joblog ../joblog/Sym_DiGCN_joblog  CUDA_VISIBLE_DEVICES=2 python ./Sym_DiGCN.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_Sym_DiGCN --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 5 15 20

../../parallel -j14 --resume-failed --results ../Output/GAT --joblog ../joblog/GAT_joblog  CUDA_VISIBLE_DEVICES=3 python ./GAT.py  --epochs 3000 --lr {1} --num_filter {2} --heads {3} --log_path telegram_GAT --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 2 4 8

../../parallel -j14 --resume-failed --results ../Output/GAT_tud --joblog ../joblog/GAT_tud_joblog  CUDA_VISIBLE_DEVICES=3 python ./GAT.py  --epochs 3000 --lr {1} --num_filter {2} --heads {3} --log_path telegram_GAT_tud --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ -tud --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 2 4 8

../../parallel -j14 --resume-failed --results ../Output/Digraph --joblog ../joblog/Digraph_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py  --epochs 3000 --lr {1} --num_filter {2} --alpha {3} --log_path telegram_Digraph --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 0.05 0.1 0.15 0.2

../../parallel -j14 --resume-failed --results ../Output/DigraphIB --joblog ../joblog/DigraphIB_joblog  CUDA_VISIBLE_DEVICES=4 python ./Digraph.py  --epochs 3000 --lr {1} --num_filter {2} --alpha {3} --log_path telegram_Digraph_IB --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --method_name DiG_ib --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 0.05 0.1 0.15 0.2

../../parallel -j9 --resume-failed --results ../Output/GIN --joblog ../joblog/GIN_joblog  CUDA_VISIBLE_DEVICES=4 python ./GIN.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_GIN --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64

../../parallel -j9 --resume-failed --results ../Output/GIN_tud --joblog ../joblog/GIN_tud_joblog  CUDA_VISIBLE_DEVICES=4 python ./GIN.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_GIN_tud --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ -tud --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64

../../parallel -j9 --resume-failed --results ../Output/SAGE --joblog ../joblog/SAGE_joblog  CUDA_VISIBLE_DEVICES=4 python ./SAGE.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_SAGE --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64

../../parallel -j9 --resume-failed --results ../Output/SAGE_tud --joblog ../joblog/SAGE_tud_joblog  CUDA_VISIBLE_DEVICES=4 python ./SAGE.py  --epochs 3000 --lr {1} --num_filter {2} --log_path telegram_SAGE_tud --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ -tud --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64

../../parallel -j14 --resume-failed --results ../Output/APPNP --joblog ../joblog/APPNP_joblog  CUDA_VISIBLE_DEVICES=5 python ./APPNP.py  --epochs 3000 --lr {1} --num_filter {2} --alpha {3} --log_path telegram_APPNP --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 0.05 0.1 0.15 0.2

../../parallel -j14 --resume-failed --results ../Output/APPNP_tud --joblog ../joblog/APPNP_tud_joblog  CUDA_VISIBLE_DEVICES=5 python ./APPNP.py  --epochs 3000 --lr {1} --num_filter {2} --alpha {3} --log_path telegram_APPNP_tud --dataset telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/ -tud --dropout 0.5 ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 0.05 0.1 0.15 0.2
