import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import helper
from main import *

class CustomDataset(data.Dataset):

	"""
		Your own custom dataloader. Fill in the functions. 
		
		__init__: get all the necessary arguments from here. 

		__len__: Should return the length of the entire dataset. 

		__getitem__(idx): Should return the item indexed by idx. 

	"""

	def __init__(self,**kwargs):
		pass

	def __len__(self):
		pass 

	def __getitem__(self,idx):
		pass 



class ImageFolderDataset(object):

	"""
		Returns train,val dataloaders adhering to ImageFolder instance rules.

		Input:
			args: Should be a dict containing:
				data: Path to train and val data marked as sub-folders with the same names. 
				distributed: Bool indicating if this mode is on/off.
				batch_size: Number of images in a batch.
				workers: num of threads for dataloader.

		Note: The augmentations used here are fixed and mostly follow standard imagenet architecture. 
			If you need something else, fucking write your own dataloader!

	"""

	def __init__(self, args,random_subset=False):
		super(ImageFolderDataset, self).__init__()
		self.args = args
		self.mean=[0.485, 0.456, 0.406]
		self.std=[0.229, 0.224, 0.225]
		self.train_sampler = None
		self.val_sampler = None

		self.normalize = transforms.Normalize(mean=self.mean,std=self.std)
		self.random_subset = random_subset

		print("Initialized dataloader with standard imagenet params")

	def get_loaders(self):

		self.traindir = os.path.join(self.args.data, 'train')
		self.valdir = os.path.join(self.args.data,'val')

		self.transforms = self.get_aug()
		self.val_transforms =  transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),\
			transforms.ToTensor(),self.normalize])

		self.train_dataset = datasets.ImageFolder(self.traindir,transform=self.transforms)
		self.val_dataset = datasets.ImageFolder(self.valdir,transform=self.val_transforms)

		if self.args.subset_size is not None:
			indices = torch.randperm(len(self.val_dataset))[:self.args.subset_size]

		print(self.random_subset,' random')
		if self.random_subset:
			#self.train_sampler = torch.utils.data.RandomSampler(self.traindir,replacement=True,num_samples=10000)
			#self.val_sampler = torch.utils.data.RandomSampler(self.valdir,replacement=True,num_samples=10000)

			#self.train_sampler = torch.utils.data.SubsetRandomSampler(indices=indices)
			#self.val_sampler = torch.utils.data.SubsetRandomSampler(indices=indices)

			#breakpoint()
			self.val_dataset = torch.utils.data.Subset(self.val_dataset, indices)
			self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
			#print('random subset')
			#print(self.train_sampler,self.val_sampler)

		elif self.args.distributed:
			self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)

		self.train_loader = torch.utils.data.DataLoader(
			self.train_dataset, batch_size=self.args.batch_size, shuffle=(self.train_sampler is None),
			num_workers=self.args.workers, pin_memory=True, sampler=self.train_sampler, drop_last=True)

		# self.val_loader = torch.utils.data.DataLoader(self.val_dataset,batch_size=self.args.batch_size,\
		# 	shuffle=False,num_workers=self.args.workers, pin_memory=True)

		print('changing val sampler')
		self.val_loader = torch.utils.data.DataLoader(self.val_dataset,batch_size=self.args.batch_size,\
			shuffle=False,num_workers=self.args.workers, pin_memory=True,sampler=self.val_sampler)


		return self.train_loader,self.val_loader


	def get_aug(self):


		augmentation = [
		transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
		transforms.RandomApply([
			transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
		], p=0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		self.normalize]

		augmentation = transforms.Compose(augmentation)

		return augmentation