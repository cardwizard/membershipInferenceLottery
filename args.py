import argparse
import torchvision.models as models

import socket
from contextlib import closing

def find_free_port():
	with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
		s.bind(('', 0))
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		return s.getsockname()[1]

def get_args():
	# Arguement Parser

	model_names = sorted(name for name in models.__dict__
		if name.islower() and not name.startswith("__")
		and callable(models.__dict__[name]))

	parser = argparse.ArgumentParser(description="Add more options if necessary")

	parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
						choices=model_names,
						help='model architecture: ' +
							' | '.join(model_names) +
							' (default: resnet18)')

	parser.add_argument('--dataset',default=None,help='chosor from tinyimagenet,cifar10 and imagenet')
	#parser.add_argument('--custom_model',action='store_true')
	parser.add_argument("--num_classes", default=1000, type=int)
	parser.add_argument('--pretrained',action='store_true')

	parser.add_argument('--adv_free',action='store_true')
	parser.add_argument('--adv_eval',action='store_true')
	parser.add_argument('--evaluate',action='store_true')
	parser.add_argument('--evaluate_topk',action='store_true')

	#Options related to distributed training.
	parser.add_argument('--world-size', default= 1, type=int, #changed default from -1
						help='number of nodes for distributed training')
	parser.add_argument('--rank', default=0, type=int, #changed default from -1
						help='node rank for distributed training')
	parser.add_argument('--dist-url', default='tcp://127.0.0.1:{port}'.format(port=find_free_port()), type=str,
						help='url used to set up distributed training')
	parser.add_argument('--dist-backend', default='nccl', type=str,
						help='distributed backend')
	parser.add_argument('--seed', default=None, type=int,
						help='seed for initializing training. ')
	parser.add_argument('--gpu', default=None, type=int,
						help='GPU id to use.')
	parser.add_argument('--multiprocessing-distributed', action='store_true',
						help='Use multi-processing distributed training to launch '
							 'N processes per node, which has N GPUs. This is the '
							 'fastest way to use PyTorch for either single node or '
							 'multi node data parallel training')
	parser.add_argument('--distributed',type=bool,default=False)


	parser.add_argument("--batch_size", default=256, type=int)	
	parser.add_argument('--debug', action='store_true', help='debug')
	parser.add_argument("--aug",help='add extra augmentations',action='store_true')

	# optimization
	parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
	parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
						help='where to decay lr, can be a list')
	parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
	parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
	parser.add_argument('--workers', type=int, default=16, help='num of workers to use')
	parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
	parser.add_argument('--start_epoch', type=int, default=0, help='number of training epochs')

	parser.add_argument("--local_rank", type=int)
	parser.add_argument('--cos', action='store_true',
					help='use cosine lr schedule')
	parser.add_argument('--schedule',default=None,nargs='+',
					help='use custom schedule, where lr is decreased by 0.1 at those schedules.')

	

	# resume, save,folders stuff
	parser.add_argument('--data',default='data/',type=str,help='Data directory')
	parser.add_argument('--model_path',default='',type=str,help='Path to model for inference.')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
					   help='path to latest checkpoint (default: none)')
	parser.add_argument('--exp_name', type=str, default='output/exp',
						help='experiment name, used to store everything related to this experiment')
	parser.add_argument('--save_freq',default=None,help='Use this to set value for periodic \
		saving of all ckpts.',type=int)
	
	parser.add_argument('--output',type=str,default='output/exp')

	#Attack
	parser.add_argument('--eps',help='model eps',default=0.031,type=float)
	parser.add_argument('--attack',help='attack_name',default=None)
	parser.add_argument('--iters',help='attack steps',default=20,type=int)

	parser.add_argument('--freq',help='dct freq to attack',default=0,type=int)
	parser.add_argument('--freq_range',help='dct freq range to attack',default=None,nargs='+')
	parser.add_argument('--freq_list',help='dct freq list to attack',default=None,nargs='+')

	parser.add_argument('--viz_freq',help='save freq attack viz in output dir',action='store_true')
	parser.add_argument('--target_acc',help='stop accuracy for variable_iters.py ',default=0.10,type=float)
	parser.add_argument('--deviation',help='deviation for blocking or replacing',default=1.0,type=float)
	parser.add_argument('--block_f',help='Block frequencies instead of replace with mean while using\
		freq_correct.py',action='store_true')

	#Extra training arguments.
	parser.add_argument('--drop_range',default=None,nargs='+',help='give a range of freq to drop randomly.')
	parser.add_argument('--drop_f',help='Randomly drop frequencies.',action='store_true')
	parser.add_argument('--drop_rate',help='Rate/percentage of freq to be dropped.',type=float,default=0.5)
	parser.add_argument('--set_freq_mean',action='store_true',help='set frequencies to mean and train (instead of dropping')

	#Viz
	parser.add_argument('--viz',action='store_true',help='vizualize segnet results')
	parser.add_argument('--viz_att',action='store_true',help='vizualize attention stuff')
	parser.add_argument('--viz_cam',action='store_true',help='vizualize attention stuff')

	#Fine tune inverse
	parser.add_argument('--seg_model',type=str,help='path to segnet inverse model')
	parser.add_argument('--class_model',type=str,help='path to classifier model')

	parser.add_argument('--sparse_loss',action='store_true',help='add sparse loss to attention',default=1e-3)
	parser.add_argument('--lamb',type=float,help='lambda to balance between inverse and classifier models',default=1e-3)
	
	#Add inverse stuff. 
	parser.add_argument('--cam_attention',action='store_true',help='option to attach Channelwise atttention in the beginning')

	#Initial attention model by Vatsal.
	parser.add_argument('--seg_attention',action='store_true',help='option to have Channelwise atttention, initial vatsal code.')


	#Options for block rep classifier with attention and attacking it.
	parser.add_argument('--img_size',type=int,default=128,help='img size for both train and test.')
	parser.add_argument('--rgb',action='store_true',help='data mode')
	parser.add_argument('--ycb',action='store_true',help='data mode')

	parser.add_argument('--bottleneck',action='store_true',help='attach attention to bottleneck')
	parser.add_argument('--begin',action='store_true',help='attach attention to beginning')

	parser.add_argument('--source_img_size',type=int,default=None,help='img size for both train and test.')
	parser.add_argument('--target_img_size',type=int,default=None,help='img size for both train and test.')
	#Attack transfer arguments.
	parser.add_argument('--topk_dataset',type=int,default=None,help='Choose from topk freq to attack: taken from average dataset')
	parser.add_argument('--topk',type=int,default=None,help='Choose from topk freq to attack: per image. Slow')

	parser.add_argument('--avg_attn_path',type=str,default=None,help='path to average attention map to choose the topk from')

	parser.add_argument('--target_adv_model',type=str,default='output/tiny_imagenet/resnet18_adv/checkpoints/model_best.pth.tar',\
		help='path to trained adv model that is being attacked')

	parser.add_argument('--subset_size',type=int,default=None,help='subset to choose from val set.')

	parser.add_argument('--only_val',type=bool,default=False,help='Run only on validation set')


	args = parser.parse_args()

	return args


