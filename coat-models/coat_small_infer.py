#https://www.kaggle.com/datasets/kozodoi/timm-pytorch-image-models
#https://www.kaggle.com/datasets/forcewithme/0916hubmapcoat
#https://www.kaggle.com/datasets/vad13irt/efficientnet-pytorch
#https://www.kaggle.com/datasets/vad13irt/segmentation-models-pytorch
#https://www.kaggle.com/datasets/vad13irt/pretrained-models-pytorch
import os,gc
import sys
sys.path.append('../input/0916hubmapcoat/coat')
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append("../input/pretrained-models-pytorch")
sys.path.append("../input/efficientnet-pytorch")
sys.path.append('../input/segmentation-models-pytorch')
import numpy as np
from common  import *
from lib.net.lookahead import *
from model_coat_daformer import *
from dataset import *
from tqdm import tqdm

import torch.cuda.amp as amp

# TRAIN = './data/hubmap/train_images/'
DEBUG = False
if DEBUG:
    TEST = '/kaggle/input/hubmap-organ-segmentation/train_images'
    TEST_DF = '/kaggle/input/hubmap-organ-segmentation/train.csv'
else:
    TEST = '../input/hubmap-organ-segmentation/test_images'
    TEST_DF = '/kaggle/input/hubmap-organ-segmentation/test.csv'
    
# MODELS = [f'/kaggle/input/hubmap0830upernetswin/fold{i}.pth' for i in range(4)]
MODELS = [
    '../input/0916hubmapcoat/small-fold0-top0.pth',
    '../input/0916hubmapcoat/small-fold0-top1.pth',
    '../input/0916hubmapcoat/small-fold0-top3.pth',
    #'../input/0916hubmapcoat/small-fold1-top0.pth',
    #'../input/0916hubmapcoat/small-fold1-top1.pth',
    '../input/0916hubmapcoat/small-fold2-top0.pth',
    '../input/0916hubmapcoat/small-fold2-top1.pth',
    '../input/0916hubmapcoat/small-fold2-top2.pth',
    '../input/0916hubmapcoat/small-fold3-top0.pth',
    '../input/0916hubmapcoat/small-fold3-top1.pth',
    '../input/0916hubmapcoat/small-fold3-top2.pth',
    '../input/hubmap0921/small_fold4_0921_top0.pth',
    '../input/hubmap0921/small_fold4_0921_top1.pth',
    '../input/hubmap0921/small_fold4_0921_top2.pth',
    #'../input/0916hubmapcoat/small_fold4_top0.pth',
    #'../input/0916hubmapcoat/small_fold4_top1.pth',
         ]



best_thrs = [
    {'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    {'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
{'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    {'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
{'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
{'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
{'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
{'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
     'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    {'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
                 'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    {'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
                 'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    {'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
                 'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    {'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
                 'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    #{'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
    #             'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
    #{'HPA': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}, 
    #             'Hubmap': {'kidney': 0.5,'prostate': 0.5,'largeintestine': 0.5,'spleen': 0.3,'lung': 0.05}},
            ]
weights = [1]*len(MODELS)
assert len(weights) == len(MODELS)
VOTE_THRESHOLD = 0.5

IMAGE_SIZE = 768
BATCH_SIZE = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_df = pd.read_csv(TEST_DF)

is_tta=True

from kaggle_hubmap_v2 import *
from daformer import *
from coat import *

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
                #  encoder=coat_lite_medium,
                encoder=coat_parallel_small,
                decoder=daformer_conv1x1,
                encoder_cfg={},
                decoder_cfg={},
                 ):
        super(Net, self).__init__()
        decoder_dim = decoder_cfg.get('decoder_dim', 320)
        self.output_type = ['inference', 'loss']
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

        x = batch
        x = self.rgb(x)

        B, C, H, W = x.shape
        encoder = self.encoder(x)
        # print([f.shape for f in encoder])

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

class HubmapDataset(Dataset):
    def __init__(self, df, image_size=IMAGE_SIZE):

        self.df = df
        self.length = len(self.df)
        self.image_size = image_size
        
    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
            string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        id = d['id']
        organ = organ_to_label[d.organ]

        image = cv2.imread('../input/hubmap-organ-segmentation/test_images/%d.tiff'%id, cv2.IMREAD_COLOR)
        image = image.astype(np.float32)/255

        s = d.pixel_size/0.4 * (image_size/3000)

        image = cv2.resize(image,dsize=(self.image_size,self.image_size),interpolation=cv2.INTER_LINEAR)

        r ={}
        r['index']= index
        r['id'] = id
        r['organ'] = torch.tensor([organ], dtype=torch.long)
        
        r['imsize'] = d['img_height']
        r['source'] = d['data_source']
        
        r['image'] = image_to_tensor(image)
        
        if is_tta:
            r['image1'] = torch.flip(r['image'], [1])
            r['image2'] = torch.flip(r['image'], [2])                        
                                    
        
        return r

test_dataset = HubmapDataset(test_df)
test_loader = DataLoader(
    test_dataset,
    sampler = SequentialSampler(test_dataset),
    batch_size  = 1,
    drop_last   = False,
    num_workers = 1,
    pin_memory  = False,
#     collate_fn = null_collate,
)

models = []
for path in MODELS:
    state_dict = torch.load(path,map_location=torch.device('cuda:0'))['state_dict']
    model = Net().cuda()
    model.load_state_dict(state_dict)
#     model.load_state_dict({k.replace('module.', ''):v for k, v in state_dict.items()})
    model.to(device)
    model.eval()
    models.append(model)
del state_dict

organs_dict = {1: 'kidney', 2: 'prostate', 3: 'largeintestine', 4: 'spleen', 5: 'lung'}

rles, names = [], []
for t, batch in tqdm(enumerate(test_loader)):

    # one iteration update  -------------
    batch_size = len(batch['index'])
    batch['image'] = batch['image'].cuda()
    if is_tta:
        batch['image1'] = batch['image1'].cuda()
        batch['image2'] = batch['image2'].cuda()
    batch['organ'] = batch['organ'].cuda()

    preds = np.zeros((batch['imsize'][0], batch['imsize'][0]), dtype=np.float)
    source = batch['source']
    organ = organs_dict[int(batch['organ'][0])]
    for i, net in enumerate(models): 
        
        net.output_type = ['inference']
        probability = net(batch['image'])['probability'].float().data.cpu().numpy().squeeze(0)
#         batch['image'] = batch['image'].cpu()
        if is_tta:

            probability1 = net(batch['image1'])['probability'].squeeze(0)
            probability1 = torch.flip(probability1, [1]).float().data.cpu().numpy()
#             batch['image1'] = batch['image1'].cpu()
            probability2 = net(batch['image2'])['probability'].squeeze(0)
            probability2 = torch.flip(probability2, [2]).float().data.cpu().numpy()
            probability2 =  probability2
#             batch['image2'] = batch['image2'].cpu()
            output = (probability+probability1+probability2)/3.
#             output = probability
        else:
            output = probability.float().data.cpu().numpy()
        output = output.squeeze()
        output = cv2.resize(output, dsize=(int(batch['imsize'][0]), int(batch['imsize'][0])))
        output = (output>best_thrs[i][source[0]][organ]).astype(np.int8)
        
        preds += weights[i] * output/sum(weights)
    
    preds = (preds>VOTE_THRESHOLD).astype(np.int8)
#     if organ == 'lung':
#         preds = (preds>=0.1).astype(np.int8)
#     else:
#         preds = (preds>=0.5).astype(np.int8)
    
    rle = rle_encode(preds)
    rles.append(rle)
    names.append(int(batch['id'][0]))

df = pd.DataFrame({'id':names,'rle':rles})
df.to_csv('coat_small_infer.csv',index=False)
