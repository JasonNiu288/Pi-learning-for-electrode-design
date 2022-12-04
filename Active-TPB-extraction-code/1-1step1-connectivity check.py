import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import tifffile as tiff
from skimage.measure import label, regionprops
import os


folder_name = '5767_image_folder'
img_path = './' + folder_name + '/'
img_pad_path = folder_name + '_padding_connected/'
img_pad_cut_side_path = folder_name + '_padding_connected_cut_side/'



def Connectivity_check(img):
    
    vals = np.unique(img)
    data_3D = np.empty([len(vals)*2, img.shape[0], img.shape[1], img.shape[2]])

    img_connected = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    for cnt, phs in enumerate(list(vals)):
        img1 = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        img1[img == phs] = 1  
        data_3D[cnt, :, :,:] = img1[:,:,:]

        label_im = label(data_3D[cnt, :, :,:], connectivity=1, background=0 )
        
        print(f'Pixel/Phase {phs}: {np.max(label_im)} regions')
        props = regionprops(label_im)

        store_label_count = np.zeros((2000,2))
        for i in range(len(props)):
            store_label_count[i,0] = i
            store_label_count[i,1] = props[i].area

        index = np.unravel_index(store_label_count.argmax(), store_label_count.shape)

        label_im[label_im>index[0]+1]=0
        label_im[label_im<index[0]+1]=0
        label_im[label_im == (index[0]+1)]=1

        data_3D[cnt+len(vals), :, :, :] = data_3D[cnt, :, :, :] - label_im
        data_3D[cnt, :, :, :] = label_im
         
        print(f'connected cells {np.sum(label_im)}')
        print(f'isolated cells {np.sum(data_3D[cnt+len(vals), :, :, :])}')

                
    # convert one-hot to greyscale image
    c1 = data_3D[0, :, :, :] #pore 0
    c2 = data_3D[1, :, :, :] #Ni 128
    c3 = data_3D[2, :, :, :] #YSZ 255

    c1_isolated = data_3D[3, :, :, :] #pore 0
    c2_isolated = data_3D[4, :, :, :] #Ni 128
    c3_isolated = data_3D[5, :, :, :] #YSZ 255

    img_connected = c1*80 + c1_isolated*0 + c2*128 + c2_isolated*0 + c3*255 + c3_isolated*0

    return img_connected



images_path_list = os.listdir(img_path)

os.makedirs(str(img_pad_path), exist_ok=True)
os.makedirs(str(img_pad_cut_side_path), exist_ok=True)



for idx, img_name in enumerate(images_path_list):
    print(f'----padding image {idx}')
    img_idx_path = img_path + '/' + img_name
    print(img_name)
    img = tiff.imread(img_idx_path)
    img_pad = np.pad(img, ((0,0),(1, 1),(1, 1)), 'constant', constant_values=((0, 0), (0,0), (0, 0))) # padding around 4 sides as air
    img_connected = Connectivity_check(img_pad)

    img_cut_connected_padding = img_connected[0:64, 1:65, 1:65]

    
    tiff.imsave(img_pad_cut_side_path + img_name, img_cut_connected_padding.astype(np.uint8), bigtiff=False) #save img for CNN
    tiff.imsave(img_pad_path + 'pad_connected_' + img_name , img_connected.astype(np.uint8), bigtiff=False)  #save img for OpenFOAM topoSetDict





                        

            

