import os
import torch
import torch.nn as nn
import numpy as np
import  pickle
import matplotlib.pyplot as plt
import cv2
import math
import imageio
import matplotlib
import matplotlib.cm as mpl_color_map
import yaml
import torchvision.transforms as transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    elif args.schedule is not None:
        args.schedule = list(map(int,args.schedule))
        for milestone in args.schedule:
            if milestone==epoch:
                lr = lr*0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def check_create_dir(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)


def load_checkpoint(args,model,optimizer=None):

    """ 
        Checkpoint format:
                keys: state_dict,optimizer,saved_epoch,

    """

    print("=> loading checkpoint '{}'".format(args.resume))
    
    if args.gpu is None:
        checkpoint = torch.load(args.resume)
    else:
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)

    args.start_epoch = checkpoint['epoch']
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        optimizer = None

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))


    return model,optimizer,args


def save_checkpoint(args,model,optimizer,epoch,is_best=False,periodic=False,custom_name=None):

    """ 
        Checkpoint format:
                keys: state_dict,optimizer,saved_epoch,
                Periodic: Save every epoch separately.
    """

    # if periodic:
    #     filename = args.exp_name + '/checkpoints/' + 'epoch_'+str(epoch)+'.pth'
    # else:
    #     filename = args.exp_name + '/checkpoints/' + 'current.pth'

    # if custom_name is not None:
    #     filename =  args.exp_name + '/checkpoints/' + custom_name

    if is_best:
        filename = args.exp_name + '/checkpoints/' + 'model_best.pth.tar'
    else:
        filename = args.exp_name + '/checkpoints/' + 'current.pth'    
    
    state = {'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
    torch.save(state, filename)

def make_dir(name):
    if not os.path.exists(name):
        os.makedirs(name,exist_ok=True)

def get_all_dct(og_img,device='cuda'):

    # start = time.time()
    images_freq = []
    for i in range(64):
        im = dct.batch_dct(og_img,n_freq=(i,i),device=device)
        #im = get_freq_image(og_img,n_freq=(i,i),device=device)
        im = im.detach()
        images_freq.append(im)
    images_freq = torch.stack(images_freq)
    images_freq = torch.transpose(images_freq,1,0)

    return images_freq      


def load_pickle(filename):
    return pickle.load(open(filename,'rb'))

def save_pickle(filename,data):
    with open(filename,'wb') as fout:
        pickle.dump(data,fout)


def get_all_dct(og_img,device='cuda'):

    images_freq = []
    for i in range(64):
        im = dct.batch_dct(og_img,n_freq=(i,i),device=device)
        im = im.detach()
        images_freq.append(im)
    images_freq = torch.stack(images_freq)
    images_freq = torch.transpose(images_freq,1,0)

    return images_freq      

def get_all_freq(og_img,device='cuda'):

    images_freq = []
    for i in range(64):
        im = get_freq_image(og_img,n_freq=(i,i),device=device)
        im = im.detach().cpu()
        images_freq.append(im)

    images_freq = torch.stack(images_freq)
    images_freq = torch.transpose(images_freq,1,0)

    return images_freq

def to_numpy(img):
    """
        Takes in tensor in 1,C,H,W format and returns H,W,C format in numpy
    """

    img = torch.squeeze(img)
    img = img.cpu().detach().numpy()
    img = np.transpose(img,(1,2,0))

    return img

def to_tensor(img,device='cuda'):
    """
        Takes in numpy image of H,W,C and converts to tensor of C,H,W
    """
    img = torch.from_numpy(img)
    H,W,C = img.shape

    #img = img.permute([2,1,0])
    img = img.permute([2,0,1])

    return img


def save_tensor(tensor,filename='temp.png'):
    """
        Tensor is in C,H,W. Convert to H,W,C numpy and save.
    """
    tensor = torch.squeeze(tensor)
    img = to_numpy(tensor)
    cv2.imwrite(filename,img*255.0)

def convert_to_grayscale(im_as_arr):
    """
    copied from: https://github.com/utkuozbulak/pytorch-cnn-visualizations/tree/master/src 
    Cool repo.
    
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def process_inp_grad(inp_grad,inverse_norm=None):
    """
        Inverse the normalization, convert to HWC numpy format. 
    """
    inp_grad = torch.squeeze(inp_grad)
    if inverse_norm is not None:
        inp_grad = inverse_norm(inp_grad)
    inp_grad = inp_grad.permute(1,2,0)
    inp_grad = inp_grad.cpu().detach().numpy() 
    return inp_grad

def normalize_array(array):
    array = array - array.min()
    array /= array.max()
    return array

def save_results(acc,args,analysis):
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with open(args.output+'/'+'args.pkl','wb') as fout:
        pickle.dump(args,fout)

    with open(args.output+'/'+'analysis.pkl','wb') as fout:
        pickle.dump(analysis,fout)


    with open(args.output+'/acc.txt','w') as fout:
        fout.write(str(acc))
    print('Saved')  

def visualize(freq_image,adv_img,img,args,inverse_norm=None):

    for i in range(5):

        f_img = process_inp_grad(freq_image[i])

        f_img = normalize_array(f_img)
        f_img = convert_to_grayscale(f_img.transpose(2,1,0))
        f_img = np.squeeze(f_img)
        
        if inverse_norm is not None:
            adversarial_img = to_numpy(inverse_norm(adv_img[i]))
            clean_img = to_numpy(inverse_norm(img[i]))
        else:
            adversarial_img = to_numpy(adv_img[i])
            clean_img = to_numpy(img[i])

        fig, axes = plt.subplots(nrows=1, ncols=3)

        axes[0].imshow(clean_img,cmap='binary')
        axes[0].title.set_text('clean_img')
        axes[0].axis('off')
        
        axes[1].imshow(adversarial_img,cmap='binary')
        axes[1].title.set_text('adv_img')
        axes[1].axis('off')

        axes[2].imshow(f_img,cmap='binary')
        axes[2].title.set_text('Freq gradients ')
        axes[2].axis('off')

        filename = args.output+'/'+'attack_viz_'+str(i)+'.png'

        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight')


def plot_help(x_val,y_val,plt_type='plot',title=None,xlabel=None,ylabel=None,\
    xlim=None,ylim=None,save_name='temp.png',axhline=None):

    if plt_type=='plot':
        plt.plot(x_val,y_val,marker='o')
    elif plt_type=='bar':
        plt.bar(x_val,y_val)

    plt.title(title)   
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    
    if axhline is not None:
        plt.axhline(y=axhline,color='red')

    plt.savefig(save_name)
    plt.close()

def visualize_segnet(args,output,images):
    save_folder = args.exp_name + '/viz/'
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)


    num_examples = min(10,args.batch_size)
    if args.debug:
        num_examples = args.batch_size - 1

    for i in range(num_examples):
        out = to_numpy(output[i])
        img = to_numpy(images[i])
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(img)
        axes[0].set_title('Original image')
        axes[1].imshow(out)
        axes[1].set_title('Learnt inverse image')
        #plt.imshow()
        plt.savefig(save_folder+str(i)+'.png')
        plt.close()


def get_plot(img,out,attn):

    # img = to_numpy(img)
    # out = to_numpy(out)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].imshow(img)
    axes[0].set_title('Original image')
    axes[1].imshow(out)
    axes[1].set_title('Learnt inverse')
    
    axes[2].imshow(attn)
    #axes[2].colorbar()
    axes[2].set_title('Attn')

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #plt.close('all')
    plt.close()

    return image

def visualize_segnet_cam(args,attention,output,images):

    save_folder = args.exp_name + '/viz/'
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)

    num_examples = min(10,args.batch_size)
    if args.debug:
        num_examples = args.batch_size - 1

    plt_images = []
    attention = attention.cpu().squeeze().detach().numpy()

    for i in range(num_examples):

        out = to_numpy(output[i])
        img = to_numpy(images[i])
        att = attention[i].reshape(8,8)

        fig, axes = plt.subplots(nrows=1, ncols=3)
        axes[0].imshow(img)
        axes[0].set_title('Original image')
        axes[1].imshow(out)
        axes[1].set_title('Learnt inverse')
        
        axes[2].imshow(att)
        #axes[2].colorbar()
        axes[2].set_title('Attn')

        plt.savefig(save_folder+'/attention_'+str(i)+'png')
        plt.close()

    #     print(img.shape,out.shape,att.shape)

    #     plt_images.append(get_plot(img,out,att))

    # plt_images = np.array(plt_images)
    # print(plt_images.shape)
    # #plt.imshow(plt_images[0])
    # #plt.savefig(save_folder+'/temp.png')

    # for it,pl in enumerate(plt_images):
    #     #imageio.mimsave(save_folder+'/attention_'+str(it)+'.gif', pl, fps=1)
    #     plt.imshow(pl)
    #     plt.savefig(save_folder+'/attention_'+str(it)+'png')


def visualize_class_attention(args,attention,images):

    reconvert = coeff_shuffle.CoefficientShuffler(3,direction='blocks')

    with open('config.yaml') as fout:
        config = yaml.load(fout,Loader=yaml.FullLoader)
    mean = config['tiny_ycb_dct_mean']
    std = config['tiny_ycb_dct_std']

    inv_normalize = transforms.Normalize(mean = [ -mean[x]/std[x] for x in range(len(mean)) ],
    std = [1/std[x] for x in range(len(std))])

    save_folder = args.exp_name + '/viz/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)

    if images.shape[1]!=3: #too many channels
        images = reconvert(images)
        images = torch.stack([inv_normalize(images[x]) for x in range(len(images))])

        images = dct.batch_idct(images,images.device)
        images = dct.to_rgb(images,images.device)


    num_examples = min(10,args.batch_size)
    for i in range(num_examples):

        img = to_numpy(images[i])
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(img)
        axes[0].set_title('Original image')
        
        temp = axes[1].imshow(attention[i].detach().cpu().numpy().squeeze())
        axes[1].set_title('Attn')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        temp.set_norm(norm)

        fig.colorbar(temp, ax=axes[1])

        plt.savefig(save_folder+'/attention_'+str(i)+'.png')
        plt.close('all')





def vis_seg_attention(args,attention,attn_matrix,output,images):
    save_folder = args.exp_name + '/viz/'
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)

    num_examples = min(10,args.batch_size)
    if args.debug:
        num_examples = args.batch_size - 1

    for i in range(num_examples):

        out = to_numpy(output[i])
        img = to_numpy(images[i])
        fig, axes = plt.subplots(nrows=1, ncols=4)
        axes[0].imshow(img)
        axes[0].set_title('Original image')
        axes[1].imshow(out)
        axes[1].set_title('Learnt inverse')
        
        axes[2].imshow(attn_matrix[i].detach().cpu().numpy().squeeze())
        #axes[2].colorbar()
        axes[2].set_title('Attn Matrix')

        axes[3].imshow(attention[i].detach().cpu().numpy().squeeze())
        #axes[3].colorbar()
        axes[3].set_title('Attn')
        #plt.colorbar()

        plt.savefig(save_folder+'/attention_'+str(i)+'.png')
        plt.close()


def load_classifier(model,args):

    if os.path.isfile(args.class_model):
        model_file = args.class_model
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    if args.gpu is None:
        checkpoint = torch.load(model_file)
    else:
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(model_file, map_location=loc)

    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
       model = nn.DataParallel(model).cuda()
       model.load_state_dict(checkpoint['state_dict'])
    print('loaded ')

    return model

def load_inverse(model,args):

    if os.path.isfile(args.seg_model):
        model_file = args.seg_model
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    if args.gpu is None:
        checkpoint = torch.load(model_file)
    else:
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(model_file, map_location=loc)

    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    
    return model



# def zigZag(arr):
#     rows, columns = len(arr), len(arr[0])
#     result = [[] for i in range(rows + columns - 1)]

#     for i in range(rows):
#         for j in range(columns):
#             sum = i + j
#             if (sum % 2 == 0):

#                 # add at beginning
#                 result[sum].insert(0, arr[i][j])
#             else:

#                 # add at end of the list
#                 result[sum].append(arr[i][j])
#     return result

# def plot_activations(gate_activations_dist,file_type=''):
    
#     #gate_activations_dist = np.load(filename)
#     y = gate_activations_dist[:64].reshape((8, 8))
#     cb = gate_activations_dist[64:128].reshape((8, 8))
#     cr = gate_activations_dist[128:].reshape((8, 8))

#     plt.figure(1, figsize = (32, 32))
#     plt.subplot(411)
#     ax = sns.heatmap(y, linewidth=0.5, cmap="OrRd", square=True)

#     plt.subplot(412)
#     ax = sns.heatmap(cb, linewidth=0.5, cmap="OrRd", square=True)

#     plt.subplot(413)
#     ax = sns.heatmap(cr, linewidth=0.5, cmap="OrRd", square=True)

#     plt.subplot(414)
#     list_a = list(np.arange(64))
#     list_b = [x for sublist in zigZag(np.asarray(list_a).reshape((8, 8))) for x in sublist]
#     list_c = [list_b.index(m) for m in list_a]

#     #List_c gives the frequency order for the 8x8 block. 

#     ax = sns.heatmap(np.asarray(list_c).reshape((8, 8)), linewidth=0.5, cmap="OrRd", square=True, annot=True, annot_kws={"size": 18})
#     # ax = sns.heatmap(np.arange(64).reshape((8, 8)), linewidth=0.5, cmap="OrRd", square=True, annot=True, annot_kws={"size": 18})
#     # plt.show()
#     plt.savefig('heatmap'+file_type+'.svg')
#     plt.close('all')

def convert_mean_std(mean,std):
    mean = torch.tensor(mean).cuda()
    std = torch.tensor(std).cuda()
    mean = mean.reshape(1,3,1,1)
    std = std.reshape(1,3,1,1)
    return mean,std

def plot_average_attn(clean_attention_dist,args):

    output = args.exp_name+'/avg_attn'
    make_dir(output)

    clean_attention_dist = clean_attention_dist.reshape(8,8)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    temp_1 = axes.imshow(clean_attention_dist)
    axes.set_title('clean attention averaged')

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    temp_1.set_norm(norm)
    fig.colorbar(temp_1, ax=axes)
    fig.suptitle(' average attention ')
    plt.savefig(output+'/average_attention'+'.png')

    np.save(output+'/avg_clean_attn.npy',clean_attention_dist)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def print_network_sparsity(model):
    num_params = num_prunable_params = num_zeros = 0
    # for layer in model.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         num_prunable_params += layer.weight.numel()
    #         num_zeros += int(torch.sum(layer.weight==0).item())
    #         num_mask_zeros += int(torch.sum(layer.mask==0).item())
    #     for name, param in layer.named_parameters():
    #         num_params += param.numel()

    for name, param in model.named_parameters():
        #if name.split("module.")[-1] in mask:
        num_prunable_params += param.numel()
        num_zeros += int(torch.sum(param==0).item())
        num_params += param.numel()

    print("Prune percentage: {:.2f}% Network sparsity: {:.2f}%"\
          .format(num_zeros/num_prunable_params*100, num_zeros/num_params*100))

    sparsity = num_zeros/num_prunable_params*100
    return sparsity