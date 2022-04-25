from ast import arg
from collections import deque
import sys
from venv import create

sys.path.insert(0, './YOLOX')
import torch
import numpy as np
import cv2
import time
from utils.couting import *
from YOLOX.yolox.data.data_augment import preproc
from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp.build import get_exp_by_name,get_exp_by_file
from YOLOX.yolox.utils import postprocess
from utils.visualize import vis
from YOLOX.yolox.utils.visualize import plot_tracking
from YOLOX.yolox.tracker.byte_tracker import BYTETracker
from torch2trt import TRTModule

COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)

class Detector():
    """ 图片检测器 """
    def __init__(self, model=None, ckpt=None):
        super(Detector, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        print("device = ",self.device)
        self.cls_names = COCO_CLASSES

        self.preproc = ValTransform(legacy=False)
        self.exp = get_exp_by_name(model)
        self.exp.test_size = (640,640)

        self.test_size = self.exp.test_size  # TODO: 改成图片自适应大小
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

        self.trt_file = "YOLOX/YOLOX_outputs/yolox_s/model_trt.pth"
        self.model.head.decode_in_inference = False
        self.decoder = self.model.head.decode_outputs

        self.load_modelTRT()

    def load_modelTRT(self):
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(self.trt_file))
        x = torch.ones(1, 3, self.exp.test_size[0], self.exp.test_size[1]).cuda()
        self.model(x)
        self.model = model_trt

    def detect(self, img):
        img_info = {"id": 0}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img, _ = self.preproc(img, None, self.test_size)

        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.exp.num_classes,  self.exp.test_conf, self.exp.nmsthre,
                class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        info = {}
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            info['boxes'], info['scores'], info['class_ids'],info['box_nums']=None,None,None,0
            return img,info

        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        info['boxes'] = bboxes
        info['scores'] = scores
        info['class_ids'] = cls
        info['box_nums'] = output.shape[0]

        return vis_res,info

class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.tsize = None
        self.name = 'yolox-s'
        self.ckpt = 'yolox_s.pth.tar'
        self.exp_file = None
        

if __name__=='__main__':
    args = Args()
    detector = Detector(model=args.name,ckpt=args.ckpt)
    tracker = BYTETracker(args, frame_rate=22)
    exp = get_exp_by_name(args.name)

    cap = cv2.VideoCapture('8.mp4')  # open one video

    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    frame_id = 0
    results = []
    fps = 0

    # create filter class
    filter_class = [2]

    # init variable for counting object
    memory = {}
    angle = -1
    in_count = 0
    out_count = 0
    already_counted = deque(maxlen=50)

    while True:
        _, im = cap.read() # read frame from video

        if im is None:
            break
        
        outputs, img_info = detector.detect(im)

        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, filter_class)

            # draw line for couting object
            line = [(0, int(0.1 * im.shape[0])), (int(im.shape[1]), int(0.1 * im.shape[0]))]
            # line = [(0, 0), (int(im.shape[1]), int(im.shape[0]))]
            cv2.line(im, line[0], line[1], (0, 255, 0), 2)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id                
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(f"{frame_id}, {tid}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f},{ t.score:.2f}, -1, -1, -1\n")

                # couting
                # get midpoint from bbox
                midpoint = tlbr_midpoint(tlwh)
                origin_midpoint = (midpoint[0], im.shape[0] - midpoint[1])  # get midpoint respective to bottom-left

                if tid not in memory:
                    memory[tid] = deque(maxlen=2)

                memory[tid].append(midpoint)
                previous_midpoint = memory[tid][0]

                origin_previous_midpoint = (previous_midpoint[0], im.shape[0] - previous_midpoint[1])

                if intersect(midpoint, previous_midpoint, line[0], line[1]) and tid not in already_counted:
                    # draw red line
                    cv2.line(im, line[0], line[1], (255, 0, 0), 2)
                    already_counted.append(tid)
                    angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                    if angle > 0:
                        out_count += 1
                    elif angle < 0:
                        in_count += 1

            if len(memory) > 50:
                del memory[list(memory)[0]]
                    
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,fps=fps, in_count=in_count, out_count=out_count)
        else:
            online_im = img_info['raw_img']

        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()
    
        # Calculating the fps
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
    
        online_im = cv2.resize(online_im,(1920,1080))
        cv2.imshow('demo', online_im)	# imshow

        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
