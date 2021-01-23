import argparse
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from backbone.interface import Interface
# from dataset import Dataset
from solver import Solver
from data import jhmdb
from data.augmentations import Augmentation


def str2bool(v):
    return v.lower() in ('true')


def _main(args, path_to_task_behavior: str, path_to_checkpoint: str, path_to_data_dir: str, path_to_task_model_path: str):


    torch.set_num_threads(1)

    # dataset
    if args.dataset == 'jhmdb':
        dataset = jhmdb.JHMDB21Detection_Tubelet
        AnnotationTransform = jhmdb.AnnotationTransform
        detection_collate_tubelet = jhmdb.detection_collate_tubelet
        CLASSES = jhmdb.CLASSES
        mode = 'train' if args.mode == 'train' else 'test'
        dataset = dataset(path_to_data_dir, mode, Augmentation(),
                                 AnnotationTransform(), input_type='rgb',
                                 full_test=True, num_K=args.K, split=args.split)
        data_loader = DataLoader(dataset, 1, num_workers=8,
                             shuffle=False, collate_fn=detection_collate_tubelet, pin_memory=True)   

    else:
        raise NotImplementedError


    solver = Solver(dataset, data_loader, path_to_data_dir, path_to_task_behavior, args, CLASSES=CLASSES, num_K=args.K)


    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    elif args.mode == 'test_with_model':
        def optimistic_restore(model, ckpt):
            model_dict = model.state_dict()
            pretrained_dict = torch.load(ckpt)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict['state_dict'])

        if args.task == 'activity':
            from model.model_activity_recognition import Model as RecognitionModel
            backbone = Interface.from_name('resnet101')(pretrained=False)
            task_model = RecognitionModel(backbone,
                              num_classes=len(CLASSES)+1,
                              args = args,
                              num_K=args.K,
                              offset_flag=True,
                              global_feat_flag=True,
                              rnn_flag=True).cuda()
            optimistic_restore(task_model, path_to_task_model_path)
            # task_model = task_model.cuda()
        
            print("Loaded Model {}".format(path_to_task_model_path))
            task_model.eval()
            solver.test_with_model(task_model)
        else:
            raise NotImplementedError




if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        # parser.add_argument('-b', '--backbone', choices=['vgg16', 'resnet101', 'deformresnet101', 'mobilenet'], required=True, help='name of backbone model')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('--task_behavior_dir', default='./data', help='path to data directory')
        parser.add_argument('--task_model_path', default='./checkpoints', help='path to data directory')
        # parser.add_argument('-r', '--results_dir', default='./results', help='path to results directory')
        parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
        parser.add_argument('--K', default=3, type=int, required=True, help='Size of subsequences')
        parser.add_argument('--task', choices=['activity', 'object', 'cls'], required=True, help='Dataset to train on')
        parser.add_argument('--dataset', choices=['jhmdb', 'coco', 'imagenet'], required=True, help='Dataset to train on')
        # parser.add_argument('--custom_dataset', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('--split', default=1, type=int, help='Which split of the dataset to work on')
        parser.add_argument('--perturbation_type', choices=['gaussian', 'spatial', 'sensor'], required=True, help='Train with spatial and temporal resolution control')
        parser.add_argument('--feedback', default=False, type=bool, help='enable feedback while test')
        parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

        # parser.add_argument('--custom_image_dir', type=str, default='data/run/gt')

        parser.add_argument('--warningnet_thrsh', type=float, default=0.5, help='number of total iterations for training D')
        # parser.add_argument('--p', type=float, default=0.5, help='number of total iterations for training D')
        parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
        parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_with_model'])

        # # Test configuration.
        # parser.add_argument('--gaussian_sigma', type=float, default=0.15, help='beta2 for Adam optimizer')
        # parser.add_argument('--spatial_subsample_factor', type=int, default=1, help='beta2 for Adam optimizer')
        # parser.add_argument('--hardware_voltage', type=str, default='1v', help='beta2 for Adam optimizer')

        # Step size.
        parser.add_argument('--log_step', type=int, default=20)
        parser.add_argument('--sample_step', type=int, default=100)
        parser.add_argument('--model_save_step', type=int, default=100)
        parser.add_argument('--lr_update_step', type=int, default=40)


        args = parser.parse_args()

        for arg in vars(args):
            print(arg, getattr(args, arg))

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        path_to_task_behavior_dir = args.task_behavior_dir + "/{:s}-{:s}-1-rgb-bs-1-resnet101-K-3".format(args.dataset, args.mode)
        path_to_task_model_path = args.task_model_path
        os.makedirs(path_to_checkpoint, exist_ok=True)

        _main(args, path_to_task_behavior_dir, path_to_checkpoint, path_to_data_dir, path_to_task_model_path)

    main()
