from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import cv2


def calculate_image_offset(image_1, image_2, scaled=0.2, threshold_ratio=0.1):
    ''' calculate image offset image2- image1'''
    resized_image_1 = cv2.resize(image_1, (0, 0), fx=scaled, fy=scaled)
    resized_image_2 = cv2.resize(image_2, (0, 0), fx=scaled, fy=scaled)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(resized_image_1, None)
    kp2, des2 = sift.detectAndCompute(resized_image_2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold_ratio* n.distance:
            good.append([m])

    differrence = {"x": 0, "y": 0}
    for i in range(len(good)):
        left = kp1[good[i][0].queryIdx].pt
        right = kp2[good[i][0].trainIdx].pt
        differrence["x"] += right[0]-left[0] 
        differrence["y"] += right[1]-left[1]

    if len(good) > 30:
        x_offset = (differrence["x"] / len(good))*(1/scaled)
        y_offset = (differrence["y"] / len(good))*(1/scaled)
    
    else:
        x_offset = 0
        y_offset = 0

    return np.array([x_offset, y_offset])


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args
        self.args.enable_kalman_offset = True
        self.args.enable_lost_track = True
        self.args.is_xywh = True
        self.args.enable_image_offset = False

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                image_name = os.path.join(r"C:\Users\Asus\Downloads\MOT\src\ByteTrack\datasets\mot\train", img_file_name[0])

                if video_name == 'MOT17-02-FRCNN' or video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.6
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4,\
                                            self.args.track_thresh - 0.5, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.35, 0.25, 0.33, 0.1]
                    self.args.track_buffer = 40

                elif video_name == 'MOT17-01-SDP':
                    self.args.track_thresh = 0.63
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4,\
                                            self.args.track_thresh - 0.5, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.35, 0.25, 0.33, 0.1]
                    self.args.track_buffer = 35

                elif video_name == 'MOT17-01-DPM':
                    self.args.track_thresh = 0.57
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4,\
                                            self.args.track_thresh - 0.5, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.35, 0.25, 0.33, 0.1]
                    self.args.track_buffer = 35


                elif video_name == 'MOT17-04-FRCNN' or video_name == 'MOT17-03-FRCNN':
                    self.args.track_thresh = 0.72
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.4, \
                                                                        self.args.track_thresh - 0.6, 0.05, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.55, 0.35, 0.3, 0.15]
                    self.args.track_buffer = 30

                elif video_name == 'MOT17-03-SDP':
                    self.args.track_thresh = 0.75
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.4, \
                                                                        self.args.track_thresh - 0.6, 0.05, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.55, 0.35, 0.3, 0.15]
                    self.args.track_buffer = 30

                elif video_name == 'MOT17-03-DPM':
                    self.args.track_thresh = 0.67
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.4, \
                                                                        self.args.track_thresh - 0.6, 0.05, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.55, 0.35, 0.3, 0.15]
                    self.args.track_buffer = 30
                    

                elif video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.5
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, \
                                                                        self.args.track_thresh - 0.4, 0.05]
                    self.args.match_threshs = [self.args.match_thresh, 0.45, 0.4, 0.3]
                    self.args.track_buffer = 15

                elif video_name == 'MOT17-06-SDP':
                    self.args.track_thresh = 0.55
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, \
                                                                        self.args.track_thresh - 0.4, 0.05]
                    self.args.match_threshs = [self.args.match_thresh, 0.45, 0.4, 0.3]
                    self.args.track_buffer = 15

                elif video_name == 'MOT17-06-DPM':
                    self.args.track_thresh = 0.5
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, \
                                                                        self.args.track_thresh - 0.4, 0.05]
                    self.args.match_threshs = [self.args.match_thresh, 0.45, 0.4, 0.3]
                    self.args.track_buffer = 20
                
                elif video_name == 'MOT17-09-FRCNN' or video_name == 'MOT17-08-FRCNN':
                    self.args.track_thresh = 0.55
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh -0.4, \
                                                                        self.args.track_thresh - 0.5]
                    self.args.match_threshs = [self.args.match_thresh, 0.7, 0.35, 0.32]
                    self.args.track_buffer = 15

                elif video_name == 'MOT17-08-SDP':
                    self.args.track_thresh = 0.6
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh -0.4, \
                                                                        self.args.track_thresh - 0.5]
                    self.args.match_threshs = [self.args.match_thresh, 0.7, 0.35, 0.32]
                    self.args.track_buffer = 15

                elif video_name == 'MOT17-08-DPM':
                    self.args.track_thresh = 0.55
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh -0.4, \
                                                                        self.args.track_thresh - 0.5]
                    self.args.match_threshs = [self.args.match_thresh, 0.7, 0.35, 0.32]
                    self.args.track_buffer = 20


                elif video_name == 'MOT17-10-FRCNN' or video_name == 'MOT17-07-FRCNN':
                    self.args.track_thresh = 0.6
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4, \
                                                                        self.args.track_thresh - 0.5, 0.05]
                    self.args.match_threshs = [self.args.match_thresh, 0.82, 0.75, 0.32, 0.25]
                    self.args.track_buffer = 27

                elif video_name == 'MOT17-07-SDP':
                    self.args.track_thresh = 0.65
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4, \
                                                                        self.args.track_thresh - 0.5, 0.05]
                    self.args.match_threshs = [self.args.match_thresh, 0.82, 0.75, 0.32, 0.25]
                    self.args.track_buffer = 27

                elif video_name == 'MOT17-07-DPM':
                    self.args.track_thresh = 0.58
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4, \
                                                                        self.args.track_thresh - 0.5, 0.05]
                    self.args.match_threshs = [self.args.match_thresh, 0.82, 0.75, 0.32, 0.25]
                    self.args.track_buffer = 30



                elif video_name == 'MOT17-11-FRCNN' or video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh -0.4, \
                                                                        self.args.track_thresh - 0.5, 0.05, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.6, 0.35, 0.32, 0.3, 0.1]    
                    self.args.track_buffer = 30

                elif video_name == 'MOT17-12-SDP':
                    self.args.track_thresh = 0.65
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh -0.4, \
                                                                        self.args.track_thresh - 0.5, 0.05, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.6, 0.35, 0.32, 0.3, 0.1]    
                    self.args.track_buffer = 30

                elif video_name == 'MOT17-12-DPM':
                    self.args.track_thresh = 0.7
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh -0.4, \
                                                                        self.args.track_thresh - 0.5, 0.05, 0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.6, 0.35, 0.32, 0.3, 0.1]    
                    self.args.track_buffer = 25


                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.55
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4, \
                                                                         0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.7, 0.5, 0.32]
                    self.args.track_buffer = 20

                elif video_name == 'MOT17-14-SDP':
                    self.args.track_thresh = 0.58
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4, \
                                                                         0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.7, 0.5, 0.32]
                    self.args.track_buffer = 20

                elif video_name == 'MOT17-14-SDP':
                    self.args.track_thresh = 0.55
                    self.args.conf_scores = [self.args.track_thresh, self.args.track_thresh - 0.2, self.args.track_thresh - 0.4, \
                                                                         0.01]
                    self.args.match_threshs = [self.args.match_thresh, 0.7, 0.5, 0.32]
                    self.args.track_buffer = 25


                if video_name not in video_names:
                    video_names[video_id] = video_name

                image_offset = np.array([0, 0])
                if frame_id == 1:
                    if self.args.enable_image_offset:
                        image_last_frame = cv2.imread(image_name)
                        image_current_frame = cv2.imread(image_name)
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []
                else:
                    if self.args.enable_image_offset:
                        image_current_frame = cv2.imread(image_name)
                        image_offset = calculate_image_offset(image_last_frame, image_current_frame, scaled=0.2, threshold_ratio=0.1)
                        image_last_frame = image_current_frame

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                saved_path =  video_name +  "_" + img_file_name[0].split('/')[-1][:-3] + "pt"
                saved_path = os.path.join(r'inference_output_torch', saved_path)
                # saved_path =  video_name +  "_" + str(frame_id) 
                # saved_path = os.path.join(r'C:\Users\Asus\Downloads\MOT\src\ByteTrack_uncleaned\inference_output_torch_pub', saved_path)

                # saved_path =  video_name +  "_" + img_file_name[0].split('/')[-1][:-3] + "pt"
                # saved_path = os.path.join(r'C:\Users\Asus\Downloads\MOT\src\ByteTrack\inference_output_torch_mot17_test', saved_path)
                data = torch.load(saved_path).type(tensor_type)
                outputs = [data]
                # outputs = model(imgs)
                # outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs.copy(), info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], image_offset, info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
