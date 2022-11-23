
from kaggle_hubmap_v2 import *
from daformer import *
from coat_plus import *



#################################################################

class RGB(nn.Module):
	IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
	IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]
	
	def __init__(self, ):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
		self.register_buffer('std', torch.ones(1, 3, 1, 1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
	
	def forward(self, x):
		x = (x - self.mean) / self.std
		return x


class Net(nn.Module):
	
	
	def __init__(self,
	             encoder=coat_parallel_small_plus1,
	             decoder=daformer_conv1x1,
	             encoder_cfg={},
	             decoder_cfg={},
	             ):
		super(Net, self).__init__()
		decoder_dim = decoder_cfg.get('decoder_dim', 320)
		
		# ----
		self.rgb = RGB()
		
		self.encoder = encoder(
			#drop_path_rate=0.3,
		)
		encoder_dim = self.encoder.embed_dims
		# [64, 128, 320, 512]
		
		self.decoder = decoder(
			encoder_dim=encoder_dim,
			decoder_dim=decoder_dim,
		)
		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim, 1, kernel_size=1),
		)
		self.aux = nn.ModuleList([
			nn.Conv2d(decoder_dim, 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
		])
	 
	def forward(self, batch):
		
		x = batch['image']
		x = self.rgb(x)
		
		B, C, H, W = x.shape
		encoder = self.encoder(x)
		#print([f.shape for f in encoder])
		
		last, decoder = self.decoder(encoder)
		logit = self.logit(last)

		# logit = F.interpolate(logit, size=(H,W),mode='bilinear',align_corners=False, antialias=True)
		logit = F.interpolate(logit, size=(H,W),mode='bilinear',align_corners=False)
		
		output = {}
		if 'loss' in self.output_type:
			
			output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['mask'])
			for i in range(4):
				output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](decoder[i]),batch['mask'], H, W)
		if 'inference' in self.output_type:
			output['probability'] = torch.sigmoid(logit)
			
		return output

 
def criterion_aux_loss(logit, mask, H, W):
	# logit = F.interpolate(logit, size=(H, W),mode='bilinear',align_corners=False, antialias=True)
	logit = F.interpolate(logit, size=(H, W),mode='bilinear',align_corners=False)
	loss = F.binary_cross_entropy_with_logits(logit,mask)
	return loss 
 
 

 


def run_check_net():
	batch_size = 2
	image_size = 800
	
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}
	
	net = Net().cuda()
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k, v in batch.items():
		print('%32s :' % k, v.shape)
	
	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
	run_check_net()
