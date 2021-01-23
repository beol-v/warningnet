import torch
import PIL
import numpy as np

import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
import math

def generate_perturbation_image(image, task_ind, perturbation_type, random=False, gaussian_sigma=0.0, spatial_factor=1, sensor_voltage='1v', K=3):

    original_image = []
    first_image = []
    second_image = []
    third_image = []

    for i, im in enumerate(image):
        ind = task_ind + i
        original_image.append(im[-2:])
        if perturbation_type == 'gaussian':
            first_image.append(get_gaussian_noise_image(im, rand_sig=random, sigma=0.05, seed=ind)[-K:])
            second_image.append(get_gaussian_noise_image(im, rand_sig=random, sigma=0.10, seed=ind)[-K:])
            third_image.append(get_gaussian_noise_image(im, rand_sig=random, sigma=0.15, seed=ind)[-K:])            
        elif perturbation_type == 'spatial':
            first_image.append(get_spatial_control_image(im, rand_sample=random, subsample_factor=2, seed=ind)[-K:])
            second_image.append(get_spatial_control_image(im, rand_sample=random, subsample_factor=3, seed=ind)[-K:])
            third_image.append(get_spatial_control_image(im, rand_sample=random, subsample_factor=4, seed=ind)[-K:])            
        elif perturbation_type == 'sensor':
            first_image.append(get_sensor_noise_image(im, rand_voltage=random, voltage='1v', seed=ind)[-K:])
            second_image.append(get_sensor_noise_image(im, rand_voltage=random, voltage='0p8v', seed=ind)[-K:])
            third_image.append(get_sensor_noise_image(im, rand_voltage=random, voltage='0p6v', seed=ind)[-K:])

    original_image = torch.cat(original_image, 0)
    first_image = torch.cat(first_image, 0)
    second_image = torch.cat(second_image, 0)
    third_image = torch.cat(third_image, 0)            
    
    return torch.cat((original_image, first_image, second_image, third_image),0) if perturbation_type != 'sensor' else torch.cat((first_image, second_image, third_image),0)

def get_gaussian_noise_image(image, rand_sig=False, sigma = 0.0, seed=1234):
    noisy_image = (image * 2.0 / 255.0) - 1.0 # ~ [-1,1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if rand_sig:
        sigma = torch.randn(1) + 0.15

    #print("sigma: {}".format(sigma))     

    noise = torch.randn(noisy_image.size()) * sigma * 2
    noisy_image = noisy_image.cuda()
    noise = noise.cuda()

    noisy_image = noisy_image + noise
    noisy_image = torch.clamp(noisy_image, -1.0, 1.0)
    noisy_image = (noisy_image + 1.0) / 2.0 * 255 # ~ [0,255]

    return noisy_image


def get_spatial_control_image(image, rand_sample=False, subsample_factor=1, seed=1234):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if rand_sample:
        subsample_factor = np.random.randint(low=1, high=5) * 2
    #print("spatial subsample_factor: {}".format(subsample_factor))

    if subsample_factor != 1:
        b, c, height, width = image.shape
        new_height = int(height/subsample_factor)
        new_width = int(width/subsample_factor)
    
        downsample = torch.nn.AvgPool2d((subsample_factor, subsample_factor), stride=subsample_factor)
        d_image = downsample(image)
        ud_image = F.upsample(d_image, size=(height, width), mode='nearest')
    else: ud_image = image

    return ud_image



def get_sensor_noise_image(image, voltage='1v', sigma='1p5', rand_voltage=False, seed=1234):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # image = image.cpu()

    if rand_voltage:
        vol = np.random.randint(low=0, high=4)
        if vol == 0: voltage = '0p9v'
        elif vol == 1: voltage = '0p8v'
        elif vol == 2: voltage = '0p7v'
        elif vol == 3: voltage = '0p6v'

    #print("voltage: {}".format(voltage))

    CURRENT_LVL = ["20n", "50n", "100n", "200n", "300n", "400n", "500n", "600n", "800n", "1u", "1p2u", "1p4u"]
    INTENSITY_LVL = {
        '1v': [6, 14, 25, 45, 64, 82, 99, 116, 147, 178, 207, 235]
    }

    def load_csv_into_int_list(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            csv_list = list(reader)
        csv_list = [ [ int(np.around(float(j))) for j in i ] for i in csv_list ]
        return csv_list


    out = torch.zeros_like(image[0], dtype=torch.long)
    out = torch.unsqueeze(out, 0)

    mc_iter_pix = torch.randint_like(image[0], low=0, high=99, dtype=torch.int32)

    for im in image:

        ## Load intensity error matrix of 13 curr level, 100 mc
        intensity_error = []
        curr_pix = torch.zeros(im.shape) ## (?, ?, 3)
        curr_pix = torch.unsqueeze(curr_pix, dim=0) ## (1, ?, ?, 3)

        for curr_id, curr in enumerate(CURRENT_LVL):
             ## Load intensity error matrix
            err = load_csv_into_int_list('../sensor_noise/noise_sigma_1p5_{}_25c/noise_{}A_processed.csv'.format(voltage, curr))
            err = [ e for i in err for e in i ]
            err = [ e - INTENSITY_LVL['1v'][curr_id] for e in err ] 
            intensity_error.append(err)  ## (13, 100)

           ##
            img = torch.unsqueeze(torch.abs(im - INTENSITY_LVL['1v'][curr_id]),dim=0)
            curr_pix = torch.cat([curr_pix, img], dim=0)
        
        intensity_error = torch.tensor(intensity_error)
        curr_pix = curr_pix[1:, :, :, :] ## (?, ?, 3, 13)
        curr_pix = torch.argmin(curr_pix, dim=0)
        curr_pix = curr_pix.type(torch.int32)
      
        #mc_iter_pix = torch.randint_like(im, low=0, high=99, dtype=torch.int32)
      
        noise = intensity_error[curr_pix.type(torch.long), mc_iter_pix.type(torch.long)]
        noise = torch.unsqueeze(noise, 0)

        out = torch.cat((out, noise), 0)

    out = out[1:].type(torch.float)
    image = image + out

    return image.cuda()


def load_task_behavior(perturb_type, task_behavior_dir):
    task_behavior_clean = _load_task_behavior(task_behavior_dir, perturbation_type='clean')
    task_behavior_first = _load_task_behavior(task_behavior_dir, perturbation_type=perturb_type, gaussian_sigma=0.05, spatial_subsample_factor=4, sensor_voltage='1v')
    task_behavior_second = _load_task_behavior(task_behavior_dir, perturbation_type=perturb_type, gaussian_sigma=0.10, spatial_subsample_factor=8, sensor_voltage='0p8v')
    task_behavior_third = _load_task_behavior(task_behavior_dir, perturbation_type=perturb_type, gaussian_sigma=0.15, spatial_subsample_factor=12, sensor_voltage='0p6v')    
    return task_behavior_clean, task_behavior_first, task_behavior_second, task_behavior_third


def _load_task_behavior(task_behavior_dir, perturbation_type='clean', gaussian_sigma=0.15, sensor_voltage='1v', spatial_subsample_factor=1):
    if perturbation_type == 'clean':
        csv_name = task_behavior_dir + "/clean.csv"

    elif perturbation_type == 'gaussian':
        gs = '{:1d}p{:02d}'.format(int(gaussian_sigma), int(gaussian_sigma * 100))
        csv_name = task_behavior_dir + "/gaussian_sigma_{:s}.csv".format(gs)

    elif perturbation_type == 'sensor':
        csv_name = task_behavior_dir + "/sensor_voltage_{:s}.csv".format(sensor_voltage)

    elif perturbation_type == 'spatial':
        csv_name = task_behavior_dir + "/spatial_subsample_{:d}.csv".format(spatial_subsample_factor)
    print("csv_name: {}".format(csv_name))

    out_list = []
    with open(csv_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        out_list = np.asarray([line for line in reader], dtype=np.float)[0]

    return out_list





def generate_labels(task_behavior_ind, num_batch, perturbation_type, task_behavior_clean, task_behavior_first, task_behavior_second, task_behavior_third):

    clean_score = []
    first_score = []
    second_score = []
    third_score = []

    for i in range(num_batch):
        ind = task_behavior_ind + i
        clean_score.append(task_behavior_clean[ind])
        first_score.append(task_behavior_first[ind])
        second_score.append(task_behavior_second[ind])
        third_score.append(task_behavior_third[ind])

    clean_score = torch.FloatTensor(clean_score)
    first_score = torch.FloatTensor(first_score)
    second_score = torch.FloatTensor(second_score)
    third_score = torch.FloatTensor(third_score)

    if perturbation_type == 'gaussian' or perturbation_type == 'spatial':
        score = torch.unsqueeze(torch.cat((clean_score, first_score, second_score, third_score), 0),-1) # [4*num_batch, 1]

    elif perturbation_type == 'sensor':
        score = torch.unsqueeze(torch.cat((first_score, second_score, third_score), 0),-1) # [3*num_batch, 1]

    score = score/score[0][0] if score[0][0] != 0 else torch.ones_like(score)
    score = torch.clamp(score/score[0][0], 0.0, 1.0)

    return score


def denorm(image):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    image = image[0].clone().detach().cuda()
    c, h, w = image.shape
    image = torch.reshape(image, (h, w, c))
    image = (image * std) + mean
    image = torch.unsqueeze(torch.reshape(image, (c, h, w)), 0)

    return image

def norm(image):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    image = image[0].clone().detach().cuda()
    c, h, w = image.shape
    image = torch.reshape(image, (h, w, c))
    image = (image - mean)/std
    image = torch.unsqueeze(torch.reshape(image, (c, h, w)), 0)

    return image

def resize_image(image, image_size):
    _, c, h, w = image.shape
    image = transforms.ToPILImage()(image[0].cpu()).convert("RGB")
    image = transforms.Resize((image_size,image_size))(image)
    image = transforms.ToTensor()(image)
    #image = torch.unsqueeze(torch.reshape(image, (c, image_size, image_size)), 0)
    return image


def get_photocurr(image):

    im_float = image.type(torch.float32)
    INTENSITY_LVL = torch.tensor([[6, 14, 25, 45, 64, 82, 99, 116, 147, 178, 207, 235],  # 0.7V
                              [6, 14, 25, 45, 64, 82, 99, 116, 147, 178, 207, 235]], dtype=torch.float32)  # 1V

    diff_pix = torch.zeros([len(INTENSITY_LVL[0]), image.shape[0], image.shape[1], image.shape[2]]).type(torch.uint8)

    for i in range(len(INTENSITY_LVL[0])):
        diff_pix[i] = torch.abs(im_float - INTENSITY_LVL[:, i][0])

    photocurr_image = torch.argmin(diff_pix, dim=0)
    photocurr_image = torch.squeeze(photocurr_image)

    return photocurr_image


def cal_sensor_energy(image, ind_voltage, low_voltage='1v'):

    # sigma = 1.5
    POWER = {}
    POWER['1v'] = torch.tensor([1.314, 1.833, 2.600, 4.077, 5.405, 6.688, 7.908, 9.104, 11.319, 13.453, 15.484, 17.422])
    POWER['0p9v'] = torch.tensor([0.874, 1.30, 1.92, 3.1, 4.14, 5.16, 6.13, 7.04, 8.74, 10.4, 12.0, 13.5])
    POWER['0p8v'] = torch.tensor([0.602, 0.921, 1.41, 2.34, 3.13, 3.89, 4.60, 5.30, 6.63, 7.84, 9.02, 10.1])
    POWER['0p7v'] = torch.tensor([0.342, 0.575, 0.950, 1.62, 2.23, 2.77, 3.30, 3.82, 4.75, 5.62, 6.47, 7.25])
    POWER['0p6v'] = torch.tensor([0.370, 0.565, 0.857, 1.345, 1.768, 2.140, 2.536, 2.901, 3.493, 4.142, 4.745, 5.245])

    if ind_voltage == 0:
        voltage = '1v'
    elif ind_voltage == 1:
        voltage = low_voltage

    image = (torch.squeeze(image) + 1.0) * 255.0 / 2.0
    photocurr_image = get_photocurr(image)

    photocurr_image_1d = photocurr_image.view(-1)

    power_image_1d = POWER[voltage][photocurr_image_1d]
    power_image = torch.reshape(power_image_1d, image.shape)
    tot_power = power_image[0, :, :].sum() + power_image[1, :, :].sum() * 2 + power_image[2, :, :].sum()
    tot_power = tot_power * 1e-9
    #print
    tot_energy = tot_power * 1e-3

    return tot_energy

