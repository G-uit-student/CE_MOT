import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState


global_count = 0

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, from_det=False, is_xywh=False):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.old_mean, self.old_covariance = None, None

        self.score = score
        self.tracklet_len = 0
        self.time_by_tracking = 0
        self.from_det = from_det
        self.is_xywh = is_xywh

        self.second_track_id = None
        self.is_mixing = False
        self.old_velocity = None
        self.count = 0

    def roll_back(self):
        self.mean = self.old_mean
        self.covariance = self.old_covariance
        self.time_since_update -= 1

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, avg_velocities_gain= [0, 0, 0, 0, 0, 0, 0, 0], std_differences = [1, 1, 1, 1, 1, 1, 1, 1]):
        if len(stracks) > 0:
            multi_mean = []
            # for st in stracks:
            #     mean_error = np.abs(st.mean.copy() - avg_velocities_gain)
            #     # if std_differences[4] == 0:
            #     #     std_differences[4] = 1
            #     # if std_differences[5] == 0:
            #     #     std_differences[5] = 1
            #     # error_gain = np.divide(mean_error[4:6], std_differences[4:6])
            #     # if np.any(np.abs(error_gain) > 2):
            #     #     mean_differences = [0, 0, 0, 0, 0, 0, 0, 0]
            #     # else:
            #     if mean_error[4] < 3 * std_differences[4] or \
            #         std_differences[5] < 3 * std_differences[5]:
            #         mean_differences = st.mean.copy() + avg_velocities_gain
            #     else:
            #         mean_differences = st.mean.copy()
            #     multi_mean.append(mean_differences)
            # multi_mean = np.asarray(multi_mean)
            multi_mean = np.asarray([st.mean.copy() + avg_velocities_gain for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].old_mean = stracks[i].mean
                stracks[i].old_covariance = stracks[i].covariance
                stracks[i].mean = mean
                stracks[i].covariance = cov

                # if stracks[i].time_since_update > 0:
                #     stracks[i].tracklet_len = 0

                stracks[i].time_since_update += 1


    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        if self.is_xywh:
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        else:
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        if self.is_xywh:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
            )

        else:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
            )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.time_since_update = 0
        if new_track.from_det:
            self.time_by_tracking = 0
        else:
            self.time_by_tracking += 1
        self.tracklet_len += 1

        self.frame_id = frame_id
        new_tlwh = new_track.tlwh
        old_mean = self.mean.copy()

        if self.is_xywh:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))            
        else:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        return self.mean.copy(), old_mean

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        if self.is_xywh:
            ret[:2] -= ret[2:] / 2

        else:
            ret[2] *= ret[3]
            ret[:2] -= ret[2:] / 2

        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def tracklet_score(self):
        score = max(0, 1 - np.log(1 + 0.05 * self.time_by_tracking)) * (self.tracklet_len - self.time_by_tracking > 2)
        return score


    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, image_offset, img_info, img_size):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # strack_dists = matching.iou_distance(strack_pool, strack_pool)
        
        # for idx, strack in enumerate(strack_pool):
        #     if strack.is_mixing:
        #         if strack.count >= 0:
        #             strack.is_mixing = False
        #             second_track = strack.second_track
        #             second_track.count = 0
        #             second_track.is_mixing = False
        #             strack_direction =  np.dot(strack.old_velocity, strack.mean[4:6])
        #             second_strack_direction =  np.dot(second_track.old_velocity, second_track.mean[4:6])
        #             if strack_direction < 0 and  second_strack_direction < 0:
        #                 strack.track_id, second_track.track_id = second_track.track_id, strack.track_id
        #         else:
        #             strack.count -= 1
        #             strack.second_track.count -= 1



        # for idx, strack_dist in enumerate(strack_dists):
        #     second_idx = np.argsort(strack_dist)[1]
        #     if strack_dists[idx, second_idx] < 0.5:
        #         if not strack_pool[idx].is_mixing and strack_pool[idx].tracklet_len > 2 and strack_pool[second_idx].tracklet_len > 2:
        #             strack_pool[idx].second_track = strack_pool[second_idx]
        #             strack_pool[idx].is_mixing = True
        #             strack_pool[idx].old_velocity = strack_pool[idx].mean[4:6]
        #             strack_pool[idx].count = 1

        #             strack_pool[idx].second_track.second_track = strack_pool[idx]
        #             strack_pool[idx].second_track.is_mixing = True
        #             strack_pool[idx].second_track.old_velocity = strack_pool[second_idx].mean[4:6]
        #             strack_pool[idx].second_track.count = 1

        # path = r'C:\Users\Asus\Downloads\MOT\mixing_track\MOT17-05'
        # for idx, strack in enumerate(strack_pool):
        #     if strack.is_mixing:
        #         saved_path = os.path.join(path, str(self.frame_id) + "_" + str(idx) + ".npy")
        #         np.save(saved_path, [strack.mean, strack.track_id])





            
        # for i, track in enumerate(strack_pool):

        #     kalman_predict_data_save_path = os.path.join(r"C:\Users\Asus\Downloads\MOT\strack_pool_data_with_id_with_oldmean\MOT17-13", "strackpool_"\
        #                                     + str(self.frame_id) + "_" + str(i) + ".npy") 

        #     np.save(kalman_predict_data_save_path, [track.mean, track.track_id])

        conf_scores = self.args.conf_scores
        match_thresh = self.args.match_threshs
        # conf_scores = [self.args.track_thresh, 0.01]
        # match_thresh = [self.args.match_thresh, 0.5]


        if np.any(np.where(np.abs(image_offset) > 0)):
            for track in strack_pool:
                track.mean[0:2] += image_offset 
                enable_kalman_offset = False
        else:
            enable_kalman_offset = self.args.enable_kalman_offset

        activated_stracks, refind_stracks, u_stracks, u_detections = multi_conf_association(\
                                                                    strack_pool, bboxes, scores, conf_scores, self.args.track_thresh, \
                                                                        match_thresh, self.frame_id, not self.args.mot20, strack_det=False,\
                                                                        is_xywh=self.args.is_xywh, enable_kalman_offset=enable_kalman_offset)
       
        for track in u_stracks:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = u_detections
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        if self.args.enable_lost_track:
            output_lost_stracks = []
            for track in self.lost_stracks:
                time_lost = self.frame_id - track.end_frame
                if track.tracklet_len - time_lost > 0:
                    if (track.tracklet_len - time_lost) / track.tracklet_len > 0.9:
                        if track.score > 0.6:
                            if time_lost <= 10:
                                output_lost_stracks.append(track)
                        elif track.score > 0.1:
                            if time_lost <= 5:
                                output_lost_stracks.append(track)
                                
            output_stracks += output_lost_stracks

        return output_stracks 

def multi_conf_association(tracks, dets, scores, conf_scores, track_thresh, match_thresh, frame_id, is_fuse, strack_det, is_xywh, enable_kalman_offset):
    dets_array = []
    scores_array = []
    for i in range(len(conf_scores)):
        if i == 0:
            idx = scores > conf_scores[i]
        else:
            idx_high = scores > conf_scores[i-1]
            idx_low = scores > conf_scores[i]
            idx = np.logical_xor(idx_high, idx_low)

        dets_array.append(dets[idx])
        scores_array.append(scores[idx])

    u_stracks = tracks
    activated_stracks = []
    refind_stracks = []
    u_detections = []

    for idx, dets in enumerate(dets_array):
        if conf_scores[idx] < track_thresh:
            is_fuse = False                                                                 
            u_stracks = [track for track in u_stracks if track.state == TrackState.Tracked] ## remove lost track

        activated_strack, refind_strack, u_strack, u_detection = matching_detections_and_tracks(dets, scores_array[idx], u_stracks,\
                                                                        match_thresh[idx], frame_id, is_fuse, strack_det, is_xywh, enable_kalman_offset)
        activated_stracks += activated_strack
        refind_stracks += refind_strack
        u_stracks = u_strack

        if conf_scores[idx] >= track_thresh:
            u_detections += u_detection


    return activated_stracks, refind_stracks, u_stracks, u_detections

def matching_detections_and_tracks(dets, det_scores, stracks, match_thresh, frame_id, is_fuse=False, strack_det=False, is_xywh=False, enable_kalman_offset=False):
    global global_count
    activated_stracks = []
    refind_stracks = []

    if len(dets) > 0:
        if not strack_det:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, from_det=True, is_xywh=is_xywh) for
                        (tlbr, s) in zip(dets, det_scores)]
        else:
            detections = dets
    else:
        detections = []

    dists = matching.iou_distance(stracks, detections)
    if is_fuse:
        dists = matching.fuse_score(dists, detections)

    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=match_thresh)
    differences = []

    for itracked, idet in matches:
        track = stracks[itracked]
        det = detections[idet]

        if track.state == TrackState.Tracked:
            new_mean, old_mean = track.update(detections[idet], frame_id)

            # kalman_predict_data_save_path = os.path.join(r"C:\Users\Asus\Downloads\MOT\strack_pool_data_with_id_with_oldmean\MOT17-13", "strackpool_"\
            #                                 + str(frame_id) + "_" + str(global_count) + ".npy") 

            # np.save(kalman_predict_data_save_path, [new_mean, old_mean])
            # global_count += 1
            activated_stracks.append(track)
            if is_fuse and enable_kalman_offset:
                differences.append(new_mean - old_mean)

        else:
            track.re_activate(det, frame_id, new_id=False)
            refind_stracks.append(track)

    if is_fuse and enable_kalman_offset:
        differences = np.array(differences)
        if differences.shape[0] == 0:
            mean_differences = [0, 0, 0, 0, 0, 0, 0, 0]
            std_differences = [1, 1, 1, 1, 1, 1, 1, 1]
        else:
            mean_differences = np.mean(differences, axis = 0)
            mean_differences[0:4] = [0, 0, 0, 0]
            mean_differences[6:] = [0, 0]
            std_differences = np.std(differences, axis = 0)
            # mean_differences_path = os.path.join(r"C:\Users\Asus\Downloads\MOT\mean\mean_MOT17_05","mean_" + str(frame_id) + ".npy")
            # np.save(mean_differences_path, mean_differences)
            if np.any(std_differences[4:6] > 0.6):
                mean_differences = [0, 0, 0, 0, 0, 0, 0, 0]

        for i in u_track:
            # original_data = np.copy(stracks[i].old_mean)
            # kalman_predict_data = np.copy(stracks[i].mean)
            # original_data_save_path = os.path.join(r"C:\Users\Asus\Downloads\MOT\original_data\MOT17-13", "original_data_"\
            #      + str(frame_id) + "_" + str(global_count) + ".npy") 

            # kalman_predict_data_save_path = os.path.join(r"C:\Users\Asus\Downloads\MOT\kalmanpredict_data\MOT17-13", "kalman_predict_data_"\
            #      + str(frame_id) + "_" + str(global_count) + ".npy") 

            # np.save(kalman_predict_data_save_path, kalman_predict_data)
            # np.save(original_data_save_path, original_data)
            # global_count += 1
            stracks[i].roll_back()
        u_stracks = [stracks[i] for i in u_track]
        
        if u_stracks:
            STrack.multi_predict(u_stracks, mean_differences, std_differences)

            # for track in u_stracks:
            #     kalman_predict_data_save_path = os.path.join(r"C:\Users\Asus\Downloads\MOT\kalmancorrect_data\MOT17-13", "kalman_correct_data_"\
            #             + str(frame_id) + "_" + str(global_count) + ".npy") 

            #     np.save(kalman_predict_data_save_path, np.copy(track.mean))
            #     global_count += 1


    else:
        u_stracks = [stracks[i] for i in u_track]
    u_detections = [detections[i] for i in u_detection]
    return activated_stracks, refind_stracks, u_stracks, u_detections


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
