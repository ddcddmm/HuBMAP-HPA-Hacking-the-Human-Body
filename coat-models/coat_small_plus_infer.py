import os
import sys
sys.path.append('../input/0918hubmapcoat/coat')
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append("../input/pretrained-models-pytorch")
sys.path.append("../input/efficientnet-pytorch")
sys.path.append('../input/segmentation-models-pytorch')
import numpy as np
from common  import *
from lib.net.lookahead import *
from model_coatplus_daformer import *
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
    
MODELS = [
    '../input/hubmap0921/smallplus_fold0_768_top2.pth',
    '../input/0918hubmapcoat/smallplus_fold1_cv820.pth',
    '../input/coat-weight-ishikei/exp004_fold2_coat_p1_size768_05559.pth',
    '../input/coat-weight-ishikei/exp004_fold3_00004998.pth',
    '../input/0918hubmapcoat/smallplus_fold4_top0.pth',
        ]

IMAGE_SIZE = 768
BATCH_SIZE = 1
TH = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_df = pd.read_csv(TEST_DF)

is_tta= True


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

        x = batch
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
    model.load_state_dict({k.replace('module.', ''):v for k, v in state_dict.items()})
    model.to(device)
    model.eval()
    models.append(model)
del state_dict

hpa_thrs =   {'kidney': 0.5,
              'prostate': 0.5,
              'largeintestine': 0.5,
              'spleen': 0.4,
              'lung': 0.3}

hub_thrs = {'kidney': 0.45,
          'prostate': 0.45,
          'largeintestine': 0.45,
          'spleen': 0.3,
          'lung': 0.2}

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
    
    source = batch['source']
    
    preds = np.zeros((batch['imsize'][0], batch['imsize'][0]), dtype=np.float)
    
    for net in models: 
        #probability = net(batch['image'])['probability'].float().data.cpu().numpy().squeeze(0)
        net.output_type = ['inference']
        probability = net(batch['image'])['probability'].float().data.cpu().numpy().squeeze(0)
        #batch['image'] = batch['image'].cpu()
        if is_tta:

            probability1 = net(batch['image1'])['probability'].squeeze(0)
            probability1 = torch.flip(probability1, [1]).float().data.cpu().numpy()
            #batch['image1'] = batch['image1'].cpu()
            probability2 = net(batch['image2'])['probability'].squeeze(0)
            probability2 = torch.flip(probability2, [2]).float().data.cpu().numpy()
            probability2 =  probability2
            #batch['image2'] = batch['image2'].cpu()
            output = (probability+probability1+probability2)/3.
#             output = probability
        else:
            output = probability.float().data.cpu().numpy()
        
        output = output.squeeze()
        output = cv2.resize(output, dsize=(int(batch['imsize'][0].numpy()), int(batch['imsize'][0].numpy())))
        preds += output/len(models)
        
    organ = organs_dict[int(batch['organ'][0])]
    if source == 'HPA':
        preds = (preds>hpa_thrs[organ]).astype(np.int8)
    else:
        preds = (preds>hub_thrs[organ]).astype(np.int8)
    
    
    rle = rle_encode(preds)
    rles.append(rle)
    names.append(int(batch['id'][0]))

df = pd.DataFrame({'id':names,'rle':rles})
df.to_csv('coat_small_plus_infer.csv',index=False)
