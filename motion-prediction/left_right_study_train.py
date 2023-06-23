import os 

num_workers = 12

list_models = [ 'VectorNet', 'LSTM'] 
list_datasets = ['argoverse', 'nuscenes']
list_percent = [5, 10, 15, 25, 30, 35, 40, 45, 50] 

##################################################################################################################

# train base models
for dataset in list_datasets:
    for model in list_models: 

        # train normal right handed
        script = f'python train.py --num_workers {num_workers} --dataset {dataset} --model {model}'
        print(script)
        os.system(script)

        # train full left handed
        script = f'python train.py --num_workers {num_workers} --dataset {dataset} --model {model} --left_handed True'
        print(script)
        os.system(script)

##################################################################################################################

# finetune models
for percent in list_percent:
    for dataset in list_datasets:
        for model in list_models: 
            
            # finetuning
            if dataset == 'nuscenes': 
                epochs = 60
                percent_ = percent * 195/128
            else:   
                epochs = 15
                percent_ = percent

            # fine tune right handed models on left handed data
            script = f'python train.py --num_workers {num_workers} --dataset {dataset} --model {model} --left_handed True --small_dataset {percent_} --finetune {model}-MSE-K1-{dataset}-2s3s-right-100.0.pth --epochs {epochs}'
            print(script)
            os.system(script)
