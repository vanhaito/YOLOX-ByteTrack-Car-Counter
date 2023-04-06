import time
import argparse
from collections import deque
import cv2
import torch
from torch2trt import TRTModule
from utils.couting import *
from yolox.data.data_augment import preproc
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp.build import get_exp_by_name
from yolox.utils import postprocess
from utils.visualize import vis
from yolox.utils.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from utils.torch_utils import select_device


class Detector():
    def __init__(self, device, model_name, trt_path, img_size):
        super(Detector, self).__init__()
        self.device = select_device(device)
        self.cls_names = COCO_CLASSES

        self.preproc = ValTransform(legacy=False)
        self.exp = get_exp_by_name(model_name)
        self.exp.test_size = (img_size, img_size)
        self.test_size = self.exp.test_size  

        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        self.model.head.decode_in_inference = False
        self.decoder = self.model.head.decode_outputs
        self.trt_path = trt_path
        self._load_modelTRT()

    def _load_modelTRT(self):
        model_trt = TRTModule()
        self.model = model_trt.load_state_dict(torch.load(self.trt_path))

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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='videos/input.mp4')
    parser.add_argument('--output-path', type=str, default='videos/output.mp4')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--trt-path', type=str, default='yolox-s_trt.pth')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--track-thresh', type=float, default=0.4)
    parser.add_argument('--track-buffer', type=float, default=30)
    parser.add_argument('--match-thresh', type=float, default=0.8) 
    parser.add_argument('--mot20', type=bool, default=False) 
    parser.add_argument('--frame-rate', type=int, default=22) 
    parser.add_argument('--name', type=str, default='yolox-s')
    parser.add_argument('--frame-width', type=int, default=1920)
    parser.add_argument('--frame-height', type=int, default=1080)
    parser.add_argument('--title', type=str, default='demo tracking')
    opt = parser.parse_args()
        
    detector = Detector(opt.device, opt.name, opt.trt_path, opt.img_size)
    tracker = BYTETracker(opt.track_thresh, opt.track_buffer, opt.match_thresh, opt.mot20, opt.frame_rate)

    cap = cv2.VideoCapture(opt.video_path)  # open one video

    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    frame_id = 0
    results = []
    fps = 0

    # create filter class
    filter_class = [2] # car

    # init variable for counting object
    memory = {}
    angle = -1
    in_count = 0
    out_count = 0
    already_counted = deque(maxlen=50)
    
    result = cv2.VideoWriter(opt.output_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (opt.frame_width, opt.frame_height))

    while True:
        _, im = cap.read() # read frame from video

        if im is None:
            break
        
        outputs, img_info = detector.detect(im)

        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], opt.img_size, filter_class)

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
    
        online_im = cv2.resize(online_im,(opt.frame_width, opt.frame_height))
        cv2.imshow(opt.title, online_im)	# imshow

        cv2.waitKey(1)
        if cv2.getWindowProperty(opt.title, cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()
