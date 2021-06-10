cd ../src

python ./sparse_Magnet.py -D --dataset telegram/telegram

python ./Sym_DiGCN.py -D --method_name DiG_ib --lr 0.1

python ./APPNP.py -D --dataset syn/fill

python ./GAT.py -D -NS --dataset syn/cyclic --lr 0.1

python ./Edge_sparseMagnet.py -D --task 1

python ./Edge_Cheb.py -D --task 2

python ./Edge_sparseMagnet.py -D --task 2 --num_class_link 3