#!bash
cd baseline_exp
mkdir not_othor
cd ..
for shot in 1 2 4 8 16 32
do
    cd baseline_exp/not_othor
    mkdir ${shot}shot
    cd ..
    cd ..
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/run_ner_dect.py --data_file_seed 0 --device 2 --shot $shot --proto_dim 160 --num_train_epochs 100 --model_logits_weight 10 >>baseline_exp/not_othor/${shot}shot/matsciner_0.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/run_ner_dect.py --data_file_seed 1 --device 4 --shot $shot --proto_dim 160 --num_train_epochs 100 --model_logits_weight 10 >>baseline_exp/not_othor/${shot}shot/matsciner_1.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/run_ner_dect.py --data_file_seed 2 --device 5 --shot $shot --proto_dim 160 --num_train_epochs 100 --model_logits_weight 10 >>baseline_exp/not_othor/${shot}shot/matsciner_2.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/run_ner_dect.py --data_file_seed 3 --device 6 --shot $shot --proto_dim 160 --num_train_epochs 100 --model_logits_weight 10 >>baseline_exp/not_othor/${shot}shot/matsciner_3.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/run_ner_dect.py --data_file_seed 4 --device 7 --shot $shot --proto_dim 160 --num_train_epochs 100 --model_logits_weight 10 >>baseline_exp/not_othor/${shot}shot/matsciner_4.out 
    sleep 5

    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 0 --device 2 --shot $shot >>baseline_exp/freeze_bert_poly/${shot}shot/matsciner_0.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 1 --device 4 --shot $shot >>baseline_exp/freeze_bert_poly/${shot}shot/matsciner_1.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 2 --device 5 --shot $shot >>baseline_exp/freeze_bert_poly/${shot}shot/matsciner_2.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 3 --device 6 --shot $shot >>baseline_exp/freeze_bert_poly/${shot}shot/matsciner_3.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/train_transformer.py --data_file_seed 4 --device 7 --shot $shot >>baseline_exp/freeze_bert_poly/${shot}shot/matsciner_4.out 
    # sleep 5

    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/baseline1/bert_ner_raw.py --seed 0 --device 2 --shot $shot >>/home/liwentao/Dec-Tuning-in-Mat/not_othor/${shot}shot/matsciner_0.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/baseline1/bert_ner_raw.py --seed 1 --device 4 --shot $shot >>/home/liwentao/Dec-Tuning-in-Mat/not_othor/${shot}shot/matsciner_1.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/baseline1/bert_ner_raw.py --seed 2 --device 5 --shot $shot >>/home/liwentao/Dec-Tuning-in-Mat/not_othor/${shot}shot/matsciner_2.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/baseline1/bert_ner_raw.py --seed 3 --device 6 --shot $shot >>/home/liwentao/Dec-Tuning-in-Mat/not_othor/${shot}shot/matsciner_3.out &
    # nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/baseline1/bert_ner_raw.py --seed 4 --device 7 --shot $shot >>/home/liwentao/Dec-Tuning-in-Mat/not_othor/${shot}shot/matsciner_4.out 
    # sleep 5
done
