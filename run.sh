#!bash
mkdir baseline_exp
cd baseline_exp
mkdir bert_raw
mkdir EntLM
cd ..
for shot in 4
do
    cd baseline_exp/EntLM
    mkdir ${shot}shot
    cd ..
    cd ..
    for seed in 0 1 2 3 4
    do
        nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed $seed  --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_${seed}.out
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 1 --device 1 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_1.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 2 --device 2 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_2.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 3 --device 3 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_3.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 4 --device 4 --shot $shot >>baseline_exp/EntLM/${shot}shot/matsciner_4.out 
    done
done
