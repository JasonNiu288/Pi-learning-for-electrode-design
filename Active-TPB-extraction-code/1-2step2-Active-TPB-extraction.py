import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
import os

def adjacent_cells(img, cell_index):
    x0 = cell_index[0]
    y0 = cell_index[1]
    z0 = cell_index[2]
    
    adjacent_cells = np.empty([6,1])
   # cell bottom 1
    if  x0 - 1 >=0:   
        adjacent_cells[0,] = img[x0 - 1, y0, z0] 
    else:
        adjacent_cells[0,] = -2 
   # cell top 2
    if  x0 + 1 >=img.shape[0]:
        adjacent_cells[1,] = -2
    else:
        adjacent_cells[1,] = img[x0 + 1, y0, z0] 


   # cell left 3
    if  y0 - 1 >=0:
        adjacent_cells[2,] = img[x0, y0 - 1, z0] 
    else:
        adjacent_cells[2,] = -2 
   # cell right 4
    if  y0 + 1 >=img.shape[1]:
        adjacent_cells[3,] = -2
    else:
        adjacent_cells[3,] = img[x0, y0 + 1, z0] 


   # cell left 5
    if  z0 - 1 >=0:
        adjacent_cells[4,] = img[x0, y0, z0 - 1] 
    else:
        adjacent_cells[4,] = -2 
   # cell right 6
    if  z0 + 1 >=img.shape[2]:
        adjacent_cells[5,] = -2
    else:
        adjacent_cells[5,] = img[x0, y0, z0+ 1] 


    return adjacent_cells

"""
img_connected_path ='./final_image_folder_connected/SOFC_39_40_85.tif'
img = tiff.imread(img_connected_path)
index = np.array([0,0,0])
out = adjacent_cells(index, img)
print(out)
"""




def Active_TPB_extract(img, pixel_value_pore, pixel_value_Ni, pixel_value_YSZ):
     # important: pixel value Ni and pixel value YSZ need to be checked in Connected-components-test.py !!!

    Active_TPB_count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                if(img[i, j, k]==pixel_value_pore):       # pore has least cells, the most efficient
                    index = np.array([i,j,k])
                    neighbour_pixel = adjacent_cells(img, index)
                    status_Ni = pixel_value_Ni in neighbour_pixel
                    status_YSZ = pixel_value_YSZ in neighbour_pixel

                    if status_Ni and status_YSZ:
                        Active_TPB_count +=1
    
    print(f'active TPB cells are {Active_TPB_count}')
    return Active_TPB_count






folder_name = '3D-trainingData-SOFC-epsilon-15-20_padding_connected_cut_side'
img_connected_path = './' + folder_name + '/'
pixel_value_pore = 80
pixel_value_Ni = 128
pixel_value_YSZ = 255

images_path_list = os.listdir(img_connected_path)

os.makedirs(str(img_connected_path), exist_ok=True)

final_csv = pd.DataFrame(columns=["img_name","TPB_length"])

for idx, img_name in enumerate(images_path_list):
    print(f'----processing image {idx} {img_name}')
    img_idx_path = img_connected_path + '/' + img_name
    img = tiff.imread(img_idx_path)


    TPB_length = Active_TPB_extract(img, pixel_value_pore, pixel_value_Ni, pixel_value_YSZ)

    new_row = {'img_name': img_name, 'TPB_length': TPB_length}
    final_csv = final_csv.append(new_row, ignore_index=True)


#final_csv.to_csv(r'./fake_images-epoch=500-TPB-loss_connected.csv', index = False, header=True)
final_csv.to_csv(r'./' + folder_name +'_TPB_length.csv', index = False, header=True)

