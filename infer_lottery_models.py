import glob,os
import json

folders = glob.glob('/fs/vulcan-projects/pruning_sgirish/classification/models/*/')

num_cmd = 0

for fol in folders:
	config = json.load(open(fol+'/config_dict.json','rb'))
	if 'kp' not in config.keys():
		config['kp'] = 1.0

	cmd = 'python main.py  --batch_size 128 --data /scratch0/ahmdtaha/imagenet/ --dataset imagenet --evaluate --gpu 0 ' \
	
	model_files = glob.glob(fol+'/*')

	if len(model_files) <= 2:
		continue

	model_path = [x for x in model_files if 'best' in x ][-1]
	arch = config['arch']

	save_path = 'lottery_infer_results/' + config['arch'] +'_keep_' + str(config['kp'])+'_'+fol.split('/')[-2] + '/'

	cmd += ' --arch ' + arch
	cmd += ' --resume '+ model_path
	cmd += ' --exp_name ' + save_path
	cmd += ' --only_val True'

	print(cmd)
	num_cmd+=1


print(num_cmd)

