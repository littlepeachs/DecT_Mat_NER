#!bash
dataset=('sst2' 'agnews' 'yahoo') #'rte' 'snli' 'mnli-m' 'mnli-mm' 

file_dir_1="128_proto_dim"
file_dir_2="no_ins_no_proto_ortho_norm"

cd experiment
mkdir $file_dir_1
sleep 1
cd $file_dir_1
mkdir $file_dir_2
cd $file_dir_2
cd ..
cd ..
cd ..
sleep 1

for i in 0 1
do
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/src/run_dect.py --calibration 1 --dataset ${dataset[i]} --seed 0 --device 0 >>experiment/$file_dir_1/$file_dir_2/${dataset[i]}_0.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/src/run_dect.py --calibration 1 --dataset ${dataset[i]} --seed 1 --device 1 >>experiment/$file_dir_1/$file_dir_2/${dataset[i]}_1.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/src/run_dect.py --calibration 1 --dataset ${dataset[i]} --seed 2 --device 2 >>experiment/$file_dir_1/$file_dir_2/${dataset[i]}_2.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/src/run_dect.py --calibration 1 --dataset ${dataset[i]} --seed 3 --device 3 >>experiment/$file_dir_1/$file_dir_2/${dataset[i]}_3.out &
    nohup /home/liwentao/miniconda3/envs/py38/bin/python /home/liwentao/Dec-Tuning-in-Mat/src/run_dect.py --calibration 1 --dataset ${dataset[i]} --seed 4 --device 4 >>experiment/$file_dir_1/$file_dir_2/${dataset[i]}_4.out 
    sleep 15
done
