import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tifffile as tiff 
from tqdm.auto import tqdm

def rle2mask(mask_rle, shape=(1600,256)):

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T
test = pd.read_csv('/kaggle/input/hubmap-organ-segmentation/test.csv')
p1  = pd.read_csv('coat_small_infer.csv')
p2  = pd.read_csv('./negative_infer.csv')
p3  = pd.read_csv('coat_small_plus_infer.csv')

test['rle1'] = test['id'].map(p1.set_index('id')['rle'])
test['rle2'] = test['id'].map(p2.set_index('id')['rle'])
test['rle3'] = test['id'].map(p3.set_index('id')['rle'])
def rle_encode_less_memory(img):
    #the image should be transposed
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

test_rows = []

# Iterate over all test images
for row_idx, row in tqdm(test.iterrows(), total=len(test)):
    try:
        mask1 = rle2mask(row['rle1'], (row['img_height'], row['img_width']))
    except:
        mask1 = np.zeros((row['img_width'], row['img_width']), dtype='uint8')
        
    try:
        mask2 = rle2mask(row['rle2'], (row['img_height'], row['img_width']))
    except:
        mask2 = np.zeros((row['img_width'], row['img_width']), dtype='uint8')
    
    try:
        mask3 = rle2mask(row['rle3'], (row['img_height'], row['img_width']))
    except:
        mask3 = np.zeros((row['img_width'], row['img_width']), dtype='uint8')

        
    mask_binary = np.where((mask1*1+mask2*0.5+mask3*0.5)>=1, 1, 0).astype(np.int8)

    test_rows.append({
        'id': row['id'],
        'rle': rle_encode_less_memory(mask_binary)
    })

test_df = pd.DataFrame(test_rows)
test_df.to_csv('submission.csv', index=False)

if len(test_df)==1:
    import matplotlib.pyplot as plt
    import cv2

    def rle_decode(mask_rle: str, img_shape: tuple = None) -> np.ndarray:
        seq = mask_rle.split()
        starts = np.array(list(map(int, seq[0::2])))
        lengths = np.array(list(map(int, seq[1::2])))
        assert len(starts) == len(lengths)
        ends = starts + lengths
        img = np.zeros((np.product(img_shape),), dtype=np.uint8)
        for begin, end in zip(starts, ends):
            img[begin:end] = 1
        return img.reshape(img_shape)

    fig = plt.figure(figsize=(18, 60))
    img = plt.imread('../input/hubmap-organ-segmentation/test_images/10078.tiff')
    mask = rle_decode(rle_encode_less_memory(mask_binary), img_shape=(2023, 2023)).T
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j, _ in enumerate(contours):
        cv2.drawContours(img, contours, j, color=(0, 255, 0), thickness=15)
    plt.imshow(img)
    fig.tight_layout()
