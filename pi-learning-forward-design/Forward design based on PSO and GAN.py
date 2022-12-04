import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile as tiff
import os
import matplotlib
from matplotlib import pyplot as plt
import copy
import Networks_3DCNN

from skimage.measure import label, regionprops


pth_WGAN_state = './WGAN-netG.pkl'
CNN_3D_physics_state = './CNN_physics-J_69.pkl'
DNN_J_state = './DNN_J_modelFC_epoch_1000.pkl'

img_optimum_path = './img_optimum'

os.makedirs(str(img_optimum_path), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 16 # Dimensions of our input noise(to our generator function)
nc = 3
num_classes = 1
gen_embedding = 320
features_gen = 64
lz =4 
img_size = 64
fixed_class_label = torch.full([1], 0).to(device)

# Parameters of DNN
input_size = 1
H1 = 500
output_size = 1




class Generator(nn.Module):
    def __init__(self, channels_noise, nc, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.embed_size = embed_size
        self.gen = nn.Sequential(
            self._block(channels_noise + int(embed_size/4/4/4), features_g * 8, 4, 2, 2),
            self._block(features_g * 8, features_g * 4, 4, 2, 2), 
            self._block(features_g * 4, features_g * 2, 4, 2, 2),  
            self._block(features_g * 2, features_g * 1, 4, 2, 2),  
            nn.ConvTranspose3d(features_g * 1,nc,kernel_size = (4, 4, 4),stride= (2, 2, 2),padding= (3, 3, 3)),
            nn.Softmax(dim=1),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, labels):
        #print(f'x.shape:{x.shape}')
        x_len = x.shape[2]
        #print(f'x_len = {x_len}')
        #print(self.embed(labels).shape)
        embedding = self.embed(labels).reshape(x.shape[0], int(self.embed_size/4/4/4), x_len, x_len, x_len)
        #print('embeding shape',embedding.shape)
        #print('x shape',x.shape)
        x = torch.cat([x, embedding], dim=1)
        #print('x shape',x.shape)
        return self.gen(x)

class FCN_test(nn.Module):
        def __init__(self,input_size, H1, output_size):
            super(FCN_test, self).__init__()

            self.fc1 = nn.Linear(input_size, H1)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(H1, H1)
            self.fc3 = nn.Linear(H1, output_size)
            self.dropout = nn.Dropout(0.2)
 
        def forward(self, x):
            x = self.fc1(x)
            x = self.act1(x)
            

            x = self.fc2(x)
            x = self.act1(x)
            

            x = self.fc3(x)
            #x = self.dropout(x)
            
            

            x = torch.flatten(x,1)
            
          
            return x




def Connectivity_check(img):
    
    vals = np.unique(img)
    data_3D = np.empty([len(vals)*2, img.shape[0], img.shape[1], img.shape[2]])

    img_connected = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    for cnt, phs in enumerate(list(vals)):
        img1 = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        img1[img == phs] = 1  
        data_3D[cnt, :, :,:] = img1[:,:,:]

        label_im = label(data_3D[cnt, :, :,:], connectivity=1, background=0 )
        
        #print(f'Pixel/Phase {phs}: {np.max(label_im)} regions')
        props = regionprops(label_im)

        store_label_count = np.zeros((10000,2))
        for i in range(len(props)):
            store_label_count[i,0] = i
            store_label_count[i,1] = props[i].area

        index = np.unravel_index(store_label_count.argmax(), store_label_count.shape)

        label_im[label_im>index[0]+1]=0
        label_im[label_im<index[0]+1]=0
        label_im[label_im == (index[0]+1)]=1

        data_3D[cnt+len(vals), :, :, :] = data_3D[cnt, :, :, :] - label_im
        data_3D[cnt, :, :, :] = label_im
         
        #print(f'connected cells {np.sum(label_im)}')
        #print(f'isolated cells {np.sum(data_3D[cnt+len(vals), :, :, :])}')

                
    # convert one-hot to greyscale image
    c1 = data_3D[0, :, :, :] #pore 0
    c2 = data_3D[1, :, :, :] #Ni 128
    c3 = data_3D[2, :, :, :] #YSZ 255

    c1_isolated = data_3D[3, :, :, :] #pore 0
    c2_isolated = data_3D[4, :, :, :] #Ni 128
    c3_isolated = data_3D[5, :, :, :] #YSZ 255

    img_connected = c1*80 + c1_isolated*0 + c2*128 + c2_isolated*0 + c3*255 + c3_isolated*0

    return img_connected


def OneHotTensorToTif_ForConnectivity(img_tensor):
    
    img_tensor = img_tensor.detach().cpu().numpy()

    img_mg = np.zeros(img_tensor.shape[1:])


    c1 = np.array(img_tensor[0])  # channel 1
    c2 = np.array(img_tensor[1])  # channel 2
    c3 = np.array(img_tensor[2])  # channel 3

    img_mg[(c1>c2)&(c1>c3)] = 0  #material one
    img_mg[(c2>c1)&(c2>c3)] = 128 #material two
    img_mg[(c3>c1)&(c3>c2)] = 255 #material two

    #tiff.imsave(img_connected_path + str(idx) + '.tif', img_mg.astype(np.uint8), bigtiff=False)
    return img_mg


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
    
    #print(f'active TPB cells are {Active_TPB_count}')
    return Active_TPB_count





#load pre-trained Generator
gen = Generator(z_dim, nc, features_gen, num_classes, img_size, gen_embedding).to(device)

gen.load_state_dict(torch.load(pth_WGAN_state))
gen = gen.to(device)
gen.eval()

#load pre-trained DNN
model_FC = FCN_test(input_size, H1, output_size)
model_FC = model_FC.to(device)
model_FC.load_state_dict(torch.load(DNN_J_state))

#load pre-trained physics-CNN
cnn = Networks_3DCNN.ConvNet(nc=1, cnn_dim=3)
CNN_physics = cnn().to(device)
CNN_physics.load_state_dict(torch.load(CNN_3D_physics_state))
CNN_physics.eval()


def J_objective(x):

    latent_noise = np.reshape(x, (1, z_dim, lz, lz, lz))
    latent_noise = torch.from_numpy(latent_noise)
    latent_noise = latent_noise.type(torch.FloatTensor).to(device)

    fake = gen(latent_noise, fixed_class_label)

    #1-Logic-driven-optimisation
    np_fake = OneHotTensorToTif_ForConnectivity(fake[0,:,:,:,:])
    img_pad = np.pad(np_fake, ((0,0),(1, 1),(1, 1)), 'constant', constant_values=((0, 0), (0,0), (0, 0)))
    img_connected = Connectivity_check(img_pad)
    img_connected = img_connected[0:64, 1:65, 1:65]
    
    
    #predict J through DNN
    TPB_length_fake_theore = Active_TPB_extract(img_connected, pixel_value_pore=80, pixel_value_Ni=128, pixel_value_YSZ=255)
    active_TPB_tensor = torch.full((1,1), TPB_length_fake_theore, dtype = torch.float).to(device)
    predicted_J = model_FC(active_TPB_tensor)
    J_fake = predicted_J.detach().cpu().numpy()
    

    return -J_fake.item()   



log_PSO  = pd.DataFrame(columns=["i", "gbest"])


N = 1000  
D = 1024  
T = 1000  
c1, c2 = 2, 2
w = 0.8
Xmax, Xmin = 1, 0
Vmax, Vmin = 1, 0


x = np.random.rand(N, D) * (Xmax - Xmin) + Xmin
v = np.random.rand(N, D) * (Vmax - Vmin) + Vmin



p = copy.deepcopy(x)
pbest = np.ones(N)
for i in range(N):
    pbest[i] = J_objective(x[i, :])
    pass

g = np.ones(D)
gbest = np.inf
for i in range(N):
    if pbest[i] < gbest:
        g = p[i, :]
        gbest = pbest[i]
        pass
    pass
gb = np.ones(T)

for i in range(T):
    
        
    for j in range(N):
        if J_objective(x[j, :]) < pbest[j]:
            p[j, :] = x[j, :]
            pbest[j] = J_objective(x[j, :])
            pass  # end if
        if pbest[j] < gbest:
            g = p[j, :]
            gbest = pbest[j]
            pass  # end if
       
        v[j, :] = w * v[j, :] + c1 * np.random.rand() * (p[j, :] - x[j, :]) + c2 * np.random.rand() * (
                g - x[j, :])
        x[j, :] = x[j, :] + v[j, :]
      
        for k in range(D):
            if v[j, k] > Vmax or v[j, k] < Vmin:
                v[j, k] = np.random.rand() * (Vmax - Vmin) + Vmin
                pass  # end if
            if x[j, k] > Xmax or x[j, k] < Xmin:
                x[j, k] = np.random.rand() * (Xmax - Xmin) + Xmin
                pass  # end if
            pass  # end k
        pass  # end j
    gb[i] = gbest
    if i%10 == 0:
       print(f'Iteration {i}/{T} gbest = {gb[i]:.4f}')
       new_row = {'Iteration': i, 'prediction': gb[i]}
       log_PSO = log_PSO.append(new_row, ignore_index=True)
    pass  # end i


    if i%10==0:
        log_PSO.to_csv(img_optimum_path + f'/PSO_log_csv_{i}' + '.csv', index=False, header=True)


        np.savetxt(img_optimum_path + f"/optim_design_parameters_{i}.csv", g, delimiter=",")
        #########test J_objective  AND Save the Optimum############
        pos_test = g


        latent_noise = np.reshape(pos_test, (1, z_dim, lz, lz, lz))
        latent_noise = torch.from_numpy(latent_noise)
        latent_noise = latent_noise.type(torch.FloatTensor).to(device)


        fake_optim = gen(latent_noise, fixed_class_label)



        img_without_connectivity_check = OneHotTensorToTif_ForConnectivity(fake_optim[0,:,:,:,:])

        img_pad = np.pad(img_without_connectivity_check.copy(), ((0,0),(1, 1),(1, 1)), 'constant', constant_values=((0, 0), (0,0), (0, 0)))
        img_connected = Connectivity_check(img_pad)
        img_connected = img_connected[0:64, 1:65, 1:65]


        tiff.imsave(img_optimum_path +  '/' + f'img_without_connectivity_check_{i}.tif', img_without_connectivity_check.astype(np.uint8), bigtiff=False)




print("optimal particle position", g)
print("best function value", gb[-1])

plt.figure()
plt.plot(gb)
plt.xlabel("Iteration")
plt.ylabel("function value")
plt.show()