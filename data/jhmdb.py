"""UCF24 Dataset Classes

Author: Gurkirt Singh for ucf101-24 dataset

"""

import os
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np

CLASSES = (  # always index 0
    'brush_hair',
    'catch',
    'clap',
    'climb_stairs',
    'golf',
    'jump',
    'kick_ball',
    'pick',
    'pour',
    'pullup',
    'push',
    'run',
    'shoot_ball',
    'shoot_bow',
    'shoot_gun',
    'sit',
    'stand',
    'swing_baseball',
    'throw',
    'walk',
    'wave')

JHMDB_label_to_cat_dict = {  # always index 0
    0:'no_activity',
    1:'brush_hair',
    2:'catch',
    3:'clap',
    4:'climb_stairs',
    5:'golf',
    6:'jump',
    7:'kick_ball',
    8:'pick',
    9:'pour',
    10:'pullup',
    11:'push',
    12:'run',
    13:'shoot_ball',
    14:'shoot_bow',
    15:'shoot_gun',
    16:'sit',
    17:'stand',
    18:'swing_baseball',
    19:'throw',
    20:'walk',
    21:'wave'}

JHMDB_cat_to_label_dict = {v:k for k,v in JHMDB_label_to_cat_dict.items()}


class AnnotationTransform(object):
    """
    Same as original
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of UCF24's 24 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        self.ind_to_class = dict(zip(range(len(CLASSES)),CLASSES))

    def __call__(self, bboxs, labels, width, height):
        res = []
        for t in range(len(labels)):
            bbox = bboxs[t,:]
            label = labels[t]
            '''pts = ['xmin', 'ymin', 'xmax', 'ymax']'''
            bndbox = []
            for i in range(4):
                cur_pt = max(0,int(bbox[i]) - 1)
                scale =  width if i % 2 == 0 else height
                cur_pt = min(scale, int(bbox[i]))
                cur_pt = float(cur_pt) / scale
                bndbox.append(cur_pt)
            bndbox.append(label)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


def readsplitfile(splitfile):
    with open(splitfile, 'r') as f:
        temptrainvideos = f.readlines()
    trainvideos = []
    for vid in temptrainvideos:
        vid = vid.rstrip('\n')
        trainvideos.append(vid)
    return trainvideos


def make_lists(rootpath, imgtype, split=1, fulltest=False):
    with open(rootpath + 'splitfiles/JHMDB-GT.pkl','rb') as fff:
        database = pickle.load(fff, encoding='latin1')

    trainvideos = database['train_videos'][split]
    testvideos = database['test_videos'][split]

    imagesDir = rootpath + imgtype + '/'
    # splitfile = rootpath + 'splitfiles/trainlist{:02d}.txt'.format(split)
    # trainvideos = readsplitfile(splitfile)

    trainlist = []
    testlist = []

    all_videos = trainvideos + testvideos

    train_action_counts = np.zeros(len(CLASSES), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES), dtype=np.int32)

    ratios = np.asarray([0.28275, 0.29275, 0.27825, 0.254,   0.2925,  0.204,   0.1685,  0.22725, 0.295,
 0.38025, 0.27825, 0.22725, 0.1905,  0.33425, 0.3035,  0.223,   0.19425, 0.344,
 0.29125, 0.2555,  0.24225])

    # ratios = np.ones_like(ratios) #TODO:uncomment this line and line 155, 156 to compute new ratios might be useful for JHMDB21
    video_list = []
    for vid, videoname in enumerate(sorted(all_videos)):
        video_list.append(videoname)
        actidx = list(database['gttubes'][videoname].keys())[0] # Label of Video
        istrain = True
        step = ratios[actidx]
        numf = database['nframes'][videoname]
        lastf = numf-1
        if videoname not in trainvideos:
            istrain = False
            step = ratios[actidx]*2.0
        if fulltest:
            step = 1
            lastf = numf

        annotations = database['gttubes'][videoname][actidx]
        num_tubes = len(annotations)

        tube_labels = np.zeros((numf,num_tubes),dtype=np.int16) # check for each tube if present in
        tube_boxes = [[[] for _ in range(num_tubes)] for _ in range(numf)]

        for tubeid, tube in enumerate(annotations):
            # print('numf00', numf, tube['sf'], tube['ef'])
            for frame_id in range(tube.shape[0]): # start of the tube to end frame of the tube
                frame_num = int(tube[frame_id, 0] - 1)
                label = actidx
                assert actidx == label, 'Tube label and video label should be same'
                box = tube[frame_id, 1:]  # get the box as an array
                box = box.astype(np.float32)
                # Already in x1 y1 x2 y2 format
                # box[2] += box[0]  #convert width to xmax
                # box[3] += box[1]  #converst height to ymax
                tube_labels[frame_num, tubeid] = label+1  # change label in tube_labels matrix to 1 form 0
                tube_boxes[frame_num][tubeid] = box  # put the box in matrix of lists

        possible_frame_nums = np.arange(0, lastf, step)
        # print('numf',numf,possible_frame_nums[-1])
        for frame_num in possible_frame_nums: # loop from start to last possible frame which can make a legit sequence
            frame_num = np.int32(frame_num)
            check_tubes = tube_labels[frame_num,:]

            if np.sum(check_tubes>0)>0:  # check if there aren't any semi overlapping tubes
                all_boxes = []
                labels = []
                if imgtype == "rgb-images":
                    image_name = imagesDir + videoname+'/{:05d}.png'.format(frame_num+1)
                elif imgtype == "brox-images":
                    image_name = imagesDir + videoname + '/{:05d}.jpg'.format(frame_num + 1)
                else:
                    image_name = None
                # label_name = rootpath + 'labels/' + videoname + '/{:05d}.txt'.format(frame_num + 1)

                assert os.path.isfile(image_name), 'Image does not exist'+image_name
                for tubeid, tube in enumerate(annotations):
                    if tube_labels[frame_num, tubeid]>0:
                        box = np.asarray(tube_boxes[frame_num][tubeid])
                        all_boxes.append(box)
                        labels.append(tube_labels[frame_num, tubeid])

                if istrain: # if it is training video
                    trainlist.append([vid, frame_num+1, np.asarray(labels)-1, np.asarray(all_boxes)])
                    train_action_counts[actidx] += len(labels)
                else: # if test video and has micro-tubes with GT
                    testlist.append([vid, frame_num+1, np.asarray(labels)-1, np.asarray(all_boxes)])
                    test_action_counts[actidx] += len(labels)
            elif fulltest and not istrain: # if test video with no ground truth and fulltest is trues
                testlist.append([vid, frame_num+1, np.asarray([9999]), np.zeros((1,4))])

    for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
        print('train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx] , int(actidx), CLASSES[actidx]))

    newratios = train_action_counts/4000
    # print('new   ratios', newratios)
    # print('older ratios', ratios)
    print('Trainlistlen', len(trainlist), ' testlist ', len(testlist))
    return trainlist, testlist, video_list


class JHMDB21Detection(data.Dataset):
    """JHMDB21 Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='JHMDB', input_type='rgb', full_test=False, split=1):

        self.input_type = input_type
        input_type = input_type+'-images'
        self.root = root
        self.CLASSES = CLASSES
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(root, input_type)
        self.ids = list()
        self.split = split

        trainlist, testlist, video_list = make_lists(root, input_type, split=split, fulltest=full_test)
        self.video_list = video_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        im, gt, img_index = self.pull_item(index)

        return im, gt, img_index

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        print("HEREERERERERE!!!!")
        print(index)
        print(self.ids[index])
        exit()
        annot_info = self.ids[index]
        print('##############HERE###########################')
        print(annot_info)
        frame_num = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        if self.input_type == 'rgb':
            img_name = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num)
        elif self.input_type == 'brox':
            img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num)
        # print(img_name)
        img = cv2.imread(img_name)
        height, width, channels = img.shape

        target = self.target_transform(annot_info[3], annot_info[2], width, height)


        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # print(height, width,target)
        return torch.from_numpy(img).permute(2, 0, 1), target, index
        # return torch.from_numpy(img), target, height, width


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    targets = []
    imgs = []
    image_ids = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])
    return torch.stack(imgs, 0), targets, image_ids


class JHMDB21Detection_Tubelet(data.Dataset):
    """JHMDB21 Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='JHMDB', input_type='rgb', full_test=False, num_K=1, split=1):

        self.input_type = input_type
        input_type = input_type + '-images'
        self.root = root
        self.CLASSES = CLASSES
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(root, input_type)
        self.ids = list()

        self.K = num_K
        self.split = split

        trainlist, testlist, video_list = make_lists(root, input_type, split=split, fulltest=full_test)
        self.video_list = video_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        index, im, scale, bbox, label = self.pull_item(index)

        return index, im, scale, bbox, label

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        annot_info = self.ids[index]
        frame_num = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        if frame_num >= self.K:
            boxes = []
            labels = []
            imgs = []

            for i in range(self.K):
                # annot_frame = self.ids[index-(K-1)+i]
                # frame_id = annot_frame[1]
                if self.input_type == 'rgb':
                    img_name = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num - (self.K-1) + i)
                elif self.input_type == 'brox':
                    img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num - (self.K-1) + i)

                img = cv2.imread(img_name).astype(np.float32)
                height, width, channels = img.shape

                target_frame = self.target_transform(annot_info[3], annot_info[2], width, height)  # Normalizes boxes
                target_frame = np.array(target_frame)

                imgs.append(img)
                boxes.append(target_frame[:, :4])
                labels.append(target_frame[:, 4] + 1) # Transform labels to 1-indexed

            imgs, boxes, labels = self.transform(imgs, boxes, labels)


            new_height, new_width, _ = img.shape
            scale = new_height / height

            # Change to CHW + RGB
            imgs = [torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1) for img in imgs]
            # boxes = [torch.tensor(b, dtype=torch.float) for b in boxes]
            # labels = [torch.tensor(l, dtype=torch.long) for l in labels]

            boxes = [torch.from_numpy(b).float() for b in boxes]
            labels = [torch.from_numpy(l).long() for l in labels]

            return index, torch.stack(imgs,0), scale, boxes[-1], labels[-1]
        else:
            boxes = []
            labels = []
            imgs = []

            for i in range(self.K):
                # annot_frame = self.ids[index-(K-1)+i]
                # frame_id = annot_frame[1]
                if self.input_type == 'rgb':
                    img_name = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num)
                elif self.input_type == 'brox':
                    img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num)
                img = cv2.imread(img_name).astype(np.float32)
                height, width, channels = img.shape

                target_frame = self.target_transform(annot_info[3], annot_info[2], width, height)  # Normalizes boxes
                target_frame = np.array(target_frame)

                imgs.append(img)
                boxes.append(target_frame[:, :4])
                labels.append(target_frame[:, 4] + 1) # Transform labels to 1 indexed

            imgs, boxes, labels = self.transform(imgs, boxes, labels)

            new_height, new_width, _ = img.shape
            scale = new_height / height

            # Change to CHW + RGB
            imgs = [torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1) for img in imgs]
            # boxes = [torch.tensor(b, dtype=torch.float) for b in boxes]
            # labels = [torch.tensor(l, dtype=torch.long) for l in labels]

            boxes = [torch.from_numpy(b).float() for b in boxes]
            labels = [torch.from_numpy(l).long() for l in labels]
            return index, torch.stack(imgs,0), scale, boxes[-1], labels[-1]
            # return empty tensors


def detection_collate_tubelet(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    imgs = []
    image_ids = []
    scales = []
    bboxes = []
    labels = []

    for sample in batch:
        image_ids.append(sample[0])
        imgs.append(sample[1])
        scales.append(sample[2])
        bboxes.append(sample[3])
        labels.append(sample[4])


    # To deal with bigger sizes
    # (Batch Size x Images in Tubelet x Channel x Height x Width)
    # (Batch Size x Images in Tubelet x Num Tubes x 4)
    # (Batch Size x Images in Tubelet x Num Tubes)
    return image_ids, torch.stack(imgs, 0), scales, bboxes, labels
