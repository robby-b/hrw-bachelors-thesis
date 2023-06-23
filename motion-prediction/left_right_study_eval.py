import os 

num_workers = 12

train_dir = './trained_models/'
eval_dir = './logs/evaluations/'
directorys = ['3', '10', '42', '100', '123']
traffic_rules = ['right', 'left']

##################################################################################################################

# evaluate all models for all seeds on left and right hand data
for traffic_rule in traffic_rules:
    for directory in directorys:
        for file_name in os.listdir(train_dir+directory+'/'):    
            if os.path.isfile(train_dir+directory+'/'+file_name):
                
                # get model name and dataset name
                model_name = file_name.split('-')[0]
                dataset_name = file_name.split('-')[3]

                # create command
                script = f'python evaluate.py --model_path {train_dir+directory}/ --eval_by_direction True --num_workers {num_workers} --dataset {dataset_name} --model {model_name} --trained_model {file_name} --save_eval True'

                # set left hand options and save in seperate folders
                if traffic_rule == 'left': 
                    script += ' --left_handed True'
                    script += f' --save_path {eval_dir+directory}/left/'
                else:
                    script += f' --save_path {eval_dir+directory}/right/'

                # print and execute command
                print(script)
                os.system(script)
                                                                   