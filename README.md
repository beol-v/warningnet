# Warningnet - Deep Learning based Early Warning Generator
Repository for our work on WarningNet to be presented in DAC 2020

## Table of Contents
- Installation
- Usage
- Future Work
- Reference

## Installation
Compatible with Python 3.5 and Pytorch 0.4.0
1. Create a virtual environment and install requirements
'''
python3 -m venv venv1
source venv1/bin/activate
pip3 install -r requirements.txt
'''
2. Make checkpoints directory
'''
mkdir checkpoints
'''
3. Link the JHMDB dataset directory
'''
ln -s <JHMDB dataset_source>
'''
4. Store the task behavior results of a specific task network that a WarningNet is targetting to. 
'''
mkdir task_behavior
mkdir task_behavior/JHMDB-<train/test>-<split>-rgb-bs-<batch_size>-<architecture>-K-<K>
'''

## Usage
To train a WarningNet,
'''
python3 main.py --mode=train --task_behavior_dir=<task behavior dir> --checkpoint=<checkpoint dir> --task=activity --data_dir=<dataset dir> --K=3 --dataset=jhmdb --perturbation_type=<gaussian/sensor/spatial>
'''

