import argparse
import os
import time

import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight



class Predictor(object):
    def __init__(self,):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=r'H:\pycharmproject\capella_nanodet\nanodet-plus-m_416.yml',
                            help="model config file path")
        parser.add_argument("--model", default=r'H:\pycharmproject\capella_nanodet\nanodet-plus-m_416_checkpoint.ckpt',
                            help="model file path")
        parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
        self.args = parser.parse_args()

        local_rank = 0
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        load_config(cfg, self.args.config)
        self.logger = Logger(local_rank, use_tensorboard=False)
        self.device="cuda:0"

        self.cfg = cfg
        model = build_model(cfg.model)
        ckpt = torch.load(self.args.model, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, self.logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(self.device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img,all_box = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img,all_box


    def detect_person(self,):
        self.logger.log('Press "Esc", "q" or "Q" to exit.')
        cap = cv2.VideoCapture(self.args.camid)
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = self.inference(frame)
                person_res = {}
                person_res[0] = res[0][0]
                # all_boxs：label, x0, y0, x1, y1, score
                result_frame,all_box = self.visualize(person_res, meta, cfg.class_names, 0.35)

                cv2.imshow("det", result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break


if __name__ == "__main__":
    detect_person = Predictor()
    detect_person.detect_person()
