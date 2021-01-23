import os, glob
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
# from dataset import Dataset
# from voc_eval import voc_eval
import numpy as np
import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
import cv2
from matplotlib.lines import Line2D
from lib.scheduler import AverageMeter

import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from model.model_warningnet import *
from lib import evaluation
import csv
import add_perturbation as addP



class Solver(object):
    def __init__(self, dataset, dataloader, path_to_data_dir, path_to_task_behavior_dir, args, CLASSES=None, num_K=3):
        super().__init__()

        self.args = args
        if not dataloader:
            self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        else:
            self.dataloader = dataloader
        self.dataset = dataset
        self._path_to_data_dir = path_to_data_dir
        self.task_behavior_dir = path_to_task_behavior_dir
        self.model_save_dir = args.checkpoint
        self.CLASSES = CLASSES
        self.epoch_size = len(self.dataset)

        # Step size.
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.lr_update_step = args.lr_update_step

        # Build the model
        self.build_model(self.args.task)

        self.load_task_behavior = addP.load_task_behavior
        self.generate_perturbation_image = addP.generate_perturbation_image
        self.generate_labels = addP.generate_labels
        self.cal_sensor_energy = addP.cal_sensor_energy
        self.denorm = addP.denorm
        self.norm = addP.norm
        self.resize_image = addP.resize_image

        if self.args.mode == 'train':
            self.task_behavior_clean, self.task_behavior_first, self.task_behavior_second, self.task_behavior_third = self.load_task_behavior(self.args.perturbation_type, self.task_behavior_dir)


    def build_model(self, task):
        """Create a generator and a discriminator."""

        self.D = WarningNet() 

        params = []
        parameter_dict = dict(self.D.named_parameters())

        for name, param in parameter_dict.items():
            if param.requires_grad:
                print("param: {}".format(name))
                params += [{'params': [param], 'lr': 1e-3, 'weight_decay': 0.0005}]

        self.d_optimizer = torch.optim.Adam(params)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.9999)
        self.print_network(self.D, 'D')
        self.D.cuda()


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        D_model_dict = self.D.state_dict()
        D_pretrained_dict = torch.load(D_path, map_location=lambda storage, loc: storage)
        D_model_dict.update(D_pretrained_dict)
        self.D.load_state_dict(D_model_dict['state_dict'])

        self.scheduler_d.load_state_dict(D_model_dict['scheduler'])


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.d_optimizer.zero_grad()


    def confidence_loss(self, logit, target):
        criterion = torch.nn.MSELoss(size_average=False)
        return criterion(logit, target)



    def train(self) -> Dict[int, float]:

        # Start training from scratch or resume training.
        start_iters = 0
        if self.args.resume_iters:
            start_iters = self.args.resume_iters
            self.restore_model(self.args.resume_iters)
           
        data_iter = None
        task_behavior_ind = 0
        losses = AverageMeter()

        # Start training.
        print('Start training...')

        start_time = time.time()
        for iters in range(start_iters, self.args.num_iters):
            if not (data_iter) or (task_behavior_ind < self.args.batch_size):
                data_iter = iter(self.dataloader)

            _, image, _, _, _ = next(data_iter)
            for i in range (1, self.args.batch_size):
                if (task_behavior_ind + i) == self.epoch_size:
                    data_iter = iter(self.dataloader)
                _, ib, _, _, _ = next(data_iter)
                image = torch.cat((image, ib),0)
            image = image.cuda()


            # add perturbation
            x = self.generate_perturbation_image(image, task_behavior_ind, self.args.perturbation_type, K=2)
            x = x / 255.0 * 2.0 - 1.0

            score = self.generate_labels(task_behavior_ind, self.args.batch_size, self.args.perturbation_type,
                                            self.task_behavior_clean, self.task_behavior_first, self.task_behavior_second, self.task_behavior_third)
            
            x = x.cuda() 
            score = score.cuda()

            task_behavior_ind = (task_behavior_ind + self.args.batch_size) % self.epoch_size

            if score[1][0] == 0.: continue

            self.D.train()
            loss = {}
    
            out = self.D(x[::2], x[1::2])

            d_loss = self.confidence_loss(out, score)
            losses.update(d_loss.cpu().detach().item())

            if (iters+1) % self.args.lr_update_step == 0: self.d_optimizer.zero_grad()
            d_loss.backward()

            torch.nn.utils.clip_grad_value_(self.D.parameters(), 1e-2)
            if (iters+1) % self.args.lr_update_step == 0: self.d_optimizer.step()
            if (iters+1) % self.args.lr_update_step == 0: self.scheduler_d.step()

            # Logging.
            loss['loss'] = d_loss.item()

            # Print out training information.
            if (iters+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                current_lr = self.scheduler_d.get_lr()[0]
                log = "Elapsed [{}], Iteration [{}/{}], Learning Rate = {}".format(et, iters+1, self.args.num_iters, current_lr)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)


            if (iters+1) % (15 * self.log_step) == 0:
                losses.reset()

            # Save model checkpoints.
            if (iters+1) % self.model_save_step == 0:
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(iters+1))

                D_state = {'iteration': iters+1,
                           'state_dict': self.D.state_dict(),
                           'opt': self.d_optimizer.state_dict(),
                           'scheduler': self.scheduler_d.state_dict(),
                           }

                torch.save(D_state, D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))



    def test(self):

        self.restore_model(self.args.test_iters)
        self.D.eval()        

        avg_out = 0
        tot_energy = 0
        tot_first_energy = 0
        tot_second_energy = 0
        tot_third_energy = 0

        feedback_ctrl = False # Set to low voltage on initial frame

        start_time = time.time()
        with torch.no_grad():
            data_iter = iter(self.dataloader)
            for iters in range(int(len(self.dataloader)/self.args.batch_size)):
                image = []
                for i in range (self.args.batch_size):
                    _, ib, _, _, _ = next(data_iter)
                    image.append(ib)
                image = torch.cat(image, 0).cuda() 

                # add perturbation
                x = self.generate_perturbation_image(image, iters*self.args.batch_size, self.args.perturbation_type, K=2)
                x = x / 255.0 * 2.0 - 1.0
                x = x.cuda()
               
                out = self.D(x[::2], x[1::2])
                avg_out += torch.FloatTensor([torch.sum(out[:self.args.batch_size]),torch.sum(out[self.args.batch_size:self.args.batch_size*2]),torch.sum(out[self.args.batch_size*2:self.args.batch_size*3]),torch.sum(out[self.args.batch_size*3:])])

                if self.args.perturbation_type == 'sensor':               
                    if self.args.feedback: # use the output as a feedback to control the sensor voltage
                        if feedback_ctrl: 
                            ind = 0 
                            print("High V")
                        else: 
                            ind = 1
                            print("Low V")
                        energy = self.cal_sensor_energy(x[ind*2:ind*2+1], ind, '0p6v')
                        
                        if out[ind][0] < self.args.warningnet_thrsh: 
                            feedback_ctrl = True # Set to nominal voltage
                        else:
                            feedback_ctrl = False # Set to low voltage
                        tot_energy += energy
    
                    else:
                        first_energy = self.cal_sensor_energy(x[1:2], 1, '1v')
                        second_energy = self.cal_sensor_energy(x[3:4], 1, '0p8v')
                        third_energy = self.cal_sensor_energy(x[5:6], 1, '0p6v')
                        tot_first_energy += first_energy
                        tot_second_energy += second_energy
                        tot_third_energy += third_energy
 
                if iters % 1000 == 0: print("{}-th iteration processing...".format(iters))

        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        print("Time for Inference: {}".format(et))

        avg_out = avg_out / (len(self.dataloader))
        print("avg_out: {}".format(avg_out))

        if self.args.feedback:
            tot_energy = tot_energy / (len(self.dataloader))
            print("tot_energy: {}".format(tot_energy))
        else:
            tot_first_energy = tot_first_energy / (len(self.dataloader))
            tot_second_energy = tot_second_energy / (len(self.dataloader))
            tot_third_energy = tot_third_energy / (len(self.dataloader))          
            
            print("tot_first_energy: {}".format(tot_first_energy))
            print("tot_second_energy: {}".format(tot_second_energy))
            print("tot_third_energy: {}".format(tot_third_energy))




    def test_with_model(self, recognition_model):
        from model.model_activity_recognition import Model

        self.restore_model(self.args.test_iters)

        self.D.eval()

        all_image_ids, all_detection_bboxes, all_detection_labels, all_detection_probs = [], [], [], []  
        det_boxes = [[] for _ in range(len(self.dataset.CLASSES))]
        gt_boxes = []
        num_nominal = 0
        num_low = 0

        tot_energy = 0

        correct = 0.0

        feedback_ctrl = False # Set to low voltage on initial frame

        start_time = time.time()
        with torch.no_grad():
            data_iter = iter(self.dataloader)
            for iters in range(len(self.dataloader)):
                if iters > 3: break
                image_id_batch, image_batch, scale_batch, boxes_batch, labels_batch = next(data_iter)

                image_id = image_id_batch[0]
                image = image_batch[0].cuda()
                scale = scale_batch[0]
                boxes = boxes_batch[0]
                labels = labels_batch[0]
                _, _, height, width = image.size()

                # add perturbation
                x = self.generate_perturbation_image(image.unsqueeze(0), iters*self.args.batch_size, self.args.perturbation_type, K=3)
                x = x / 255.0 * 2.0 - 1.0
                first_image = x[-9:-6]
                second_image = x[-3:]

                y = second_image

                out = self.D(torch.unsqueeze(y[-2],0),torch.unsqueeze(y[-1],0))
                
                if self.args.feedback:
                    if out[0][0] < self.args.warningnet_thrsh: 
                        feedback_ctrl = True
                        ind = 0
                        print("Nominal")
                        x = first_image
                        num_nominal += 1

                    else:
                        feedback_ctrl = False       
                        ind = 1
                        print("Low Voltage/Resolution")
                        x = second_image
                        num_low += 1


                if self.args.perturbation_type == 'sensor':
                    if feedback_ctrl:
                        energy = self.cal_sensor_energy(first_image[-1:], 0) + self.cal_sensor_energy(second_image[-1:], 1, '0p6v')
                    else: energy = self.cal_sensor_energy(second_image[-1:], 1, '0p6v')
                elif self.args.perturbation_type == 'spatial':
                    if feedback_ctrl: energy = 1
                    else: energy = 1.0/4**2
                print('energy: ', energy)
                tot_energy += energy


                def task_forward(image, Model, model):
                    iimage = image / 255.0
                    forward_input = Model.ForwardInput.Eval(iimage)
                    forward_output = recognition_model.eval().forward(forward_input)
                    return forward_output
        
                forward_output = task_forward(x, Model, recognition_model)
                model = recognition_model

                detection_bboxes = forward_output.detection_bboxes # Normalized first before saving
                detection_scores = forward_output.detection_scores

                # Perform NMS for evaluation
                detection_bboxes, detection_labels, detection_probs = model.NMS_FN(detection_bboxes, detection_scores, height, width, nms_thresh=0.3)

                detection_bboxes = detection_bboxes.numpy() / scale
                detection_labels = detection_labels.cpu().numpy().astype(np.int32)
                detection_probs = detection_probs.numpy()

                if self.args.feedback:
 
                    boxes = boxes.numpy() / scale
                    labels = labels.numpy().astype(np.int32)
                    labels = labels - 1
    
                    for label in range(len(self.CLASSES)):
                        det_boxes[label].append(np.zeros((0, 5), dtype=np.float32))
    
                    for box, label, score in zip(detection_bboxes, detection_labels, detection_probs):
                        if(label > 0): # Ignore background detections
                            detection = np.hstack((box, score))
                            det_boxes[label-1][-1] = np.vstack((det_boxes[label-1][-1], detection))
   
                    gt_targets = np.hstack((boxes, labels[:, np.newaxis]))
                    gt_boxes.append(gt_targets)

                    det_boxes = [[] for _ in range(len(self.CLASSES))] 
                
                    for label in range(len(self.CLASSES)):
                        det_boxes[label].append(np.zeros((0, 5), dtype=np.float32))
                
                    for box, label, score in zip(detection_bboxes, detection_labels, detection_probs):
                        if(label > 0): # Ignore background detections
                            detection = np.hstack((box, score))
                            det_boxes[label-1][-1] = np.vstack((det_boxes[label-1][-1], detection))

                    x = x / 255.0 
                    im_to_plot_np = x[0].clone().cpu().numpy()
                    im_to_plot_np = np.transpose(im_to_plot_np, (1, 2, 0)) # (K, 300, 300, 3)
                    #im_to_plot_np = (im_to_plot_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    im_to_plot_np = np.clip(im_to_plot_np, 0, 1)
                    im_to_plot = (im_to_plot_np*255).astype(np.uint8).copy()
                
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    for label_ind, label in enumerate(self.CLASSES):
                        for detection in det_boxes[label_ind][0]:
                            box = detection[:4]
                            score = detection[-1]
                            if score > 0.5 and label_ind+1 == labels[0].numpy().astype(np.int32):
                                print('correct')
                                correct += 1

                            if (score < 0.5):
                                continue
                            else:
                                print("det: {}, {}, {}".format(label, box, score))
                                box_center = (box[:2] + box[2:4]) / 2
                                box_center_i = [int(b) for b in box_center]
                                cv2.rectangle(im_to_plot, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                                txt = '{}: {:.3f}'.format(label, score)
                                cv2.putText(im_to_plot, txt, (box[0], box[1]), font, 0.50, (255, 255, 255), 2)


                    if iters % 1000 == 0:
                        print("{}-th iter Evaluated".format(iters))

        tot_energy = tot_energy / (len(self.dataloader))
        print("tot_energy: {}".format(tot_energy))

        if self.args.feedback:
            mAP, ap_all, ap_strs = evaluation.evaluate_detections(gt_boxes, det_boxes, self.dataset.CLASSES, iou_thresh=0.5)
            print('Mean AP {}\n'.format(mAP))
            print('nominal frames: {}, low frames: {}'.format(num_nominal, num_low))                
