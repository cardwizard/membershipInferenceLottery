"""
   Meta script to run script and generate vector probability data for different models.
"""
import glob
import os

#Path: arch_keep_percentage
#res_model_paths = {'model_b4bf649c52605ab7d32ba58c0242f335':'resnet18_'}

#res_model_paths = {'model_a76bfd8b9842925d5c5dad60349a6809':'resnet18_20', 'model_646a1905f1703fc12ad60c3b2b2cb61d':'resnet18_50',\
#'model_2fec83b9f616801fd7ffa4238590a47f':'resnet50_10','model_7dd1ac8f4cb4937a4ca0f2c54edf79ed':'resnet50_20',\
#'pretrained':'resnet50_100'}

model_root_folder = '/fs/vulcan-projects/pruning_sgirish/classification/models/'


base_cmd = 'python main.py  --batch_size 128 --dataset imagenet --evaluate --gpu 0 '

if os.path.exists('/scratch0/ahmdtaha/imagenet/'):
   data_path = '/scratch0/ahmdtaha/imagenet/'
elif os.path.exists('/scratch0/ahmdtaha/datasets/imagenet/'):
   data_path = '/scratch0/ahmdtaha/datasets/imagenet/'

base_cmd += ' --data ' + data_path

subsets = [10000,20000,30000,40000,50000]


num_cmd = 0
for i in res_model_paths:

   keep = int(float(res_model_paths[i].split('_')[-1]))

   cmd = base_cmd +''

   if 'pretrained' in i:
      cmd+= ' --pretrained '
   else:
      model_file = model_root_folder + i +'/model_best.pth.tar'
      cmd+= ' --resume ' + model_file

   if 'resnet18' in res_model_paths[i]:   
      arch = 'resnet18'
      cmd+=' --arch resnet18'
   else:
      arch = 'resnet50'
      cmd+=' --arch resnet50'


   for sub in subsets:
      subset_cmd = ' --subset ' + str(sub)
      if arch=='resnet18':
         save_name = 'res18_vectors/lottery_' + str(keep) +'/' + 'lottery_' + str(keep) + '_' + str(sub) +'/'
      else:
         save_name = 'res50_vectors/lottery_' + str(keep) +'/' + 'lottery_' + str(keep) + '_' + str(sub) +'/'

      save_name_cmd = ' --exp_name ' + save_name
      #cmd += ' --exp_name ' + save_name
      
      final_cmd = cmd+ subset_cmd + save_name_cmd 
      print('\n')
      print(final_cmd)
      os.system(final_cmd)
      print('\n')

      num_cmd +=1

print(num_cmd)





