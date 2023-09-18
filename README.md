# DecTNER
The source codes for EntLM.

## Dependencies:

Cuda 11.2, python 3.9.16

To install the required packages by following commands:

```
$ pip3 install -r requirements.txt
```

To download the pretrained bert-base-cased model:
```
$ python run_dect_ner.py
```

## Few-shot Experiment
The few-shot dataset has been constructed before in [1,2,4,8,16,32].
The scripts contains label-word selected before.
Run the few-shot experiments by editing the shot and seed in  with:
```
sh run_dect_ner.py
```
if you want to run the baseline experiment, please enter the corresponding content.
for example, baseline 1:
```
cd baseline1_Bert_raw
python bert_ner_raw.py
```
baseline 2:
```
cd baseline1_EntLM
python train_transformer.py
```


