# Assignment AIL 861

## Preprocessing 
<p> Run the following commands to perform preprocessing and tokenization of the data.: 

```bash
python preprocess.py
python create_vocab.py
python tokenizer.py
python save_fasttext_embedding.py
```

## Running experiments
All the training configuration files are present inside configs/
To run normal experiment (without gradient checkpointing or accumulation), use the following command: 
```bash
## Use pretrain.yml for model configuration 1 and pretrain2.yml for model configuration 2
python main.py --config_path configs/pretrain.yml  
```
To run gradient checkpointing experiments use the following command: 
```bash
python main_checkpoint.py --config_path configs/pretrain_checkpoint.yml
```
To run gradient accumulation experiments use the following command: 
```bash
## Choose number based on accumulation steps
python main_ga.py --config_path configs/pretrain_ga_{2/4/8}.yml 
```