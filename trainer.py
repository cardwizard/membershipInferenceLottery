from main import *
import helper

def train(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = helper.AverageMeter('Time', ':6.3f')
	data_time = helper.AverageMeter('Data', ':6.3f')
	losses = helper.AverageMeter('Loss', ':.4e')
	top1 = helper.AverageMeter('Acc@1', ':6.2f')
	top5 = helper.AverageMeter('Acc@5', ':6.2f')
	progress = helper.ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top5],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	softmax_list = []
	target_list = []


	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu,non_blocking=True)

		output = model(images)
		loss = criterion(output, target)

		softmax_out = torch.nn.functional.softmax(output, dim=1)
		softmax_list.append(softmax_out.cpu().detach())


		acc1, acc5 = helper.accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), images[0].size(0))
		top1.update(acc1[0], images[0].size(0))
		top5.update(acc5[0], images[0].size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		target_list.append(target.clone().cpu().detach())


		if i % 10 == 0:
			progress.display(i)


	softmax_list = torch.stack(softmax_list[:-1:])
	target_list = torch.stack(target_list[:-1:])

	n1,n2,cl = softmax_list.shape
	target_list = target_list.view(n1*n2)
	softmax_list = softmax_list.view(n1*n2,cl)


	helper.make_dir(args.exp_name)

	torch.save(softmax_list,args.exp_name+'/softmax_training.pt')
	torch.save(target_list,args.exp_name+'/target_training.pt')


	return top1.avg, top5.avg,losses.avg   

def validate(val_loader, model, criterion, args,name='val'):
	batch_time = helper.AverageMeter('Time', ':6.3f')
	losses = helper.AverageMeter('Loss', ':.4e')
	top1 = helper.AverageMeter('Acc@1', ':6.2f')
	top5 = helper.AverageMeter('Acc@5', ':6.2f')
	progress = helper.ProgressMeter(
		len(val_loader),
		[batch_time, losses, top1, top5],
		prefix='Test: ')

	# switch to evaluate mode
	model.eval()
	softmax_list = []
	target_list = []

	with torch.no_grad():
		end = time.time()
		#print(len(val_loader),args.batch_size)

		for i, (images, target) in enumerate(val_loader):

			if args.gpu is not None:
				images = images.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(images)
			loss = criterion(output, target)

			softmax_out = torch.nn.functional.softmax(output, dim=1)
			softmax_list.append(softmax_out.cpu().detach())
			target_list.append(target.clone().cpu().detach())

			# measure accuracy and record loss
			acc1, acc5 = helper.accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0], images.size(0))
			top5.update(acc5[0], images.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				progress.display(i)

		# TODO: this should also be done with the helper.ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

	softmax_list = torch.stack(softmax_list[:-1:])
	target_list = torch.stack(target_list[:-1:])

	n1,n2,cl = softmax_list.shape
	target_list = target_list.view(n1*n2)
	softmax_list = softmax_list.view(n1*n2,cl)


	helper.make_dir(args.exp_name)

	if name=='val':
		torch.save(softmax_list,args.exp_name+'/softmax_validation.pt')
		torch.save(target_list,args.exp_name+'/target_validation.pt')
	else:
		torch.save(softmax_list,args.exp_name+'/softmax_training.pt')
		torch.save(target_list,args.exp_name+'/target_training.pt')


	return top1.avg, top5.avg,losses.avg  


def adv_train_free(train_loader, model, criterion,optimizer,global_noise_data,\
	step_size,clip_eps,mean,std,epoch,args):

	batch_time = helper.AverageMeter('Time', ':6.3f')
	data_time = helper.AverageMeter('Data', ':6.3f')
	losses = helper.AverageMeter('Loss', ':.4e')
	top1 = helper.AverageMeter('Acc@1', ':6.2f')
	top5 = helper.AverageMeter('Acc@5', ':6.2f')
	progress = helper.ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top5],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	#Bounds for clamping, if not in 0-1 range. 
	lower,upper = -mean/std, (1-mean)/std
	lower = lower.reshape(1,c,1,1)
	upper = upper.reshape(1,c,1,1)

	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu,non_blocking=True)

		for j in range(configs.ADV.n_repeats):
			# Ascend on the global noise
			noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
			in1 = input + noise_batch
			
			#in1.clamp_(0, 1.0)
			#in1.sub_(mean).div_(std)
			in1 = torch.where(in1 < lower,lower,torch.where(in1> upper,upper,in1))

			output = model(in1)
			loss = criterion(output, target)
			
			prec1, prec5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(prec1[0], input.size(0))
			top5.update(prec5[0], input.size(0))

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()

			# Update the noise for the next iteration
			pert = step_size*torch.sign(noise_batch.grad)

			global_noise_data[0:input.size(0)] += pert.data
			global_noise_data.clamp_(-clip_eps,clip_eps)

			optimizer.step()
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

		if i % 10 == 0:
			progress.display(i)

		# TODO: this should also be done with the helper.ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

		return top1.avg, top5.avg,losses.avg, global_noise_data