#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


import helper
import trainer
import args
from dataloader import ImageFolderDataset

import pickle
import yaml

with open('config.yaml') as fout:
	config = yaml.load(fout,Loader=yaml.FullLoader)

def main(args):


	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count()
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		print(ngpus_per_node,args.gpu)
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
	args.gpu = gpu


	if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
		
		#if args.rank==0:

		tb_dir = args.exp_name+'/tb_logs/'
		ckpt_dir = args.exp_name + '/checkpoints/'

		helper.make_dir(args.exp_name)
		helper.make_dir(tb_dir)
		helper.make_dir(ckpt_dir)

		with open(args.exp_name+'/args.pkl','wb') as fout:
			pickle.dump(args,fout)

		print("writing to : ",tb_dir+'{}'.format(args.exp_name),args.rank,ngpus_per_node)
		writer = SummaryWriter(tb_dir, flush_secs=10)

	# suppress printing if not master
	if args.multiprocessing_distributed and args.gpu != 0:
		def print_pass(*args):
			pass
		builtins.print = print_pass

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)

	if args.pretrained:
		print('loading pretrained model')

	if args.dataset == 'tinyimagenet' or args.dataset=='tiny_imagenet':
		args.num_classes = 200

		print("Modifying resnet for tinyimagenet ",'pretrained = ',args.pretrained)
		model = models.__dict__[args.arch](pretrained=args.pretrained)

		model.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),stride=(2,2),padding=(3,3),bias=True)
		model.maxpool = nn.Sequential()
		model.fc = nn.Linear(in_features=model.fc.in_features,out_features=args.num_classes,bias=True)
		model.avgpool = nn.AdaptiveAvgPool2d(1)
		model.requires_grad_(True)
		img_size = (64,64)
		my_dataset = TinyImageFolderDataset(args)
	
		mean = config['tiny_mean']
		std = config['tiny_std']

	elif args.dataset == 'imagenet':
		args.num_classes = 1000
		print('loading Imagenet; pretrained = ',args.pretrained)
		model = models.__dict__[args.arch](pretrained=args.pretrained)
		img_size = (224,224)
		mean = config['imagenet_mean']
		std = config['imagenet_std']

		print('Generating random samples: subset of train,val')
		
		if args.subset_size is None:
			my_dataset = ImageFolderDataset(args,random_subset=False)
		else:
			print('random_subset mode:\n')
			my_dataset = ImageFolderDataset(args,random_subset=True)

		#train_loader = torch.utils.data.RandomSampler(args.data+'/train/',replacement=True,num_samples=10000)
		#val_loader = torch.utils.data.RandomSampler(args.data+'/val/',replacement=True,num_samples=10000)

	elif args.dataset =='cifar10':
		args.num_classes = 10
		print("loading cifar10")
		from models import resnet
		model = eval("resnet."+ args.arch+'('+str(args.num_classes)+')')
		my_dataset = CifarLoader(args)
		img_size = (32,32)
		mean = torch.tensor(config['cifar_mean']).cuda()
		std = torch.tensor(config['cifar_std']).cuda()

	my_dataset.get_loaders()

	if args.distributed:
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			model.cuda()
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
		print("no dataparallel on model.")
		# comment out the following line for debugging
		#raise NotImplementedError("Only DistributedDataParallel is supported.")
	#else:
		#raise NotImplementedError("Only DistributedDataParallel is supported.")
		model = nn.DataParallel(model).cuda()


	criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			model,optimizer,args = helper.load_checkpoint(args,model,optimizer)
		elif os.path.isfile(args.resume+'/checkpoints/model_best.pth.tar'):
			model,optimizer,args = helper.load_checkpoint(args,model,optimizer)
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	#USE this only when batch size is fixed. 
	#This takes time, but optimizes to crazy speeds once input is fixed. 
	cudnn.benchmark = True

	


	if args.evaluate:

		print('\n')
		sparsity = helper.print_network_sparsity(model)
		print('\n')

		if args.only_val: #only val
			val_top1,val_top5,val_loss = trainer.validate(my_dataset.val_loader, model, criterion, args,name='val')	
		else: #run both
			train_top1, train_top5, train_loss = trainer.validate(my_dataset.train_loader, model, criterion, args,name='train')
			val_top1,val_top5,val_loss = trainer.validate(my_dataset.val_loader, model, criterion, args,name='val')	

		save_name = args.exp_name + '/acc.txt'
		with open(save_name,'w') as fout:
			fout.write(str(val_top1.item()))		

		save_name = args.exp_name + '/sparsity.txt'
		with open(save_name,'w') as fout:
			fout.write(str(sparsity))		


		return 


	best_acc = 0
	if args.adv_free:
		global_noise_data = torch.zeros([args.batch_size, 3,img_size[0],img_size[1]]).cuda()

	for epoch in range(args.start_epoch, args.epochs):

		if args.distributed:
			my_dataset.train_sampler.set_epoch(epoch)

		helper.adjust_learning_rate(optimizer, epoch, args)

		if args.adv_free:
			clip_eps = config['adv_eps']
			step_size = config['adv_step_size']
			n_repeats = config['n_repeats']

			train_top1, train_top5, train_loss, global_noise_data = trainer.adv_train_free(my_dataset.train_loader, model,\
			 criterion,optimizer,global_noise_data,step_size,clip_eps,n_repeats,mean,std,epoch,args)
		else:
			train_top1, train_top5, train_loss = trainer.train(my_dataset.train_loader,model,criterion,optimizer,epoch,args)

		val_top1,val_top5,val_loss = trainer.validate(my_dataset.val_loader, model, criterion, args)

		#Logging.
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			writer.add_scalar("loss/train", train_loss, epoch)
			writer.add_scalar("top1/train", train_top1, epoch)
			writer.add_scalar("top5/train", train_top5, epoch)

			writer.add_scalar("loss/val", val_loss, epoch)
			writer.add_scalar("top1/val", val_top1, epoch)
			writer.add_scalar("top5/val", val_top5, epoch)

		#Save models.
		if val_top1 > best_acc:
			best_acc = val_top1
			is_best = True
		else:
			is_best = False

		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			helper.save_checkpoint(args,model,optimizer,epoch,is_best=is_best,periodic=False)

	with open(args.exp_name+'/acc.txt','w') as fout:
		fout.write(str(best_acc.item()))


if __name__ == '__main__':
	args = args.get_args()
	main(args)
