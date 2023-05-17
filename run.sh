#!bash
mkdir baseline_exp
cd baseline_exp
mkdir bert_raw
mkdir EntLM
cd ..
for shot in 1 2 4 8 16 32
do
    cd baseline_exp/EntLM
    mkdir ${shot}shot
    cd ..
    cd ..
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 0 --device 2 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_0.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 1 --device 4 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_1.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 2 --device 5 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_2.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 3 --device 6 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_3.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 4 --device 7 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_4.out 
done
