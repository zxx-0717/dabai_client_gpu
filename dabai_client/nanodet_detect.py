#-*- coding: UTF-8 -*-
import cv2
import time
import socket
from .utils import *
from struct import pack,unpack

import rclpy
import cv2
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo,PointCloud2
from rclpy.qos import qos_profile_sensor_data
from capella_ros_msg.msg import DetectResult
from capella_ros_msg.msg import SingleDetector
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


class DaBaiSubscriber(Node):

        def __init__(self):
                super().__init__('DaBai_Subscriber')
                self.K = None
                self.depth_data = []
                self.color_subscription = self.create_subscription(Image,'/camera/color/image_raw',
                self.color_subscription_callback,qos_profile_sensor_data)
                self.color_subscription  # prevent unused variable warning

                self.depth_subscription = self.create_subscription(Image,'/camera/depth/image_raw',
                self.depth_subscription_callback,
                qos_profile_sensor_data)
                self.depth_subscription  # prevent unused variable warning

                self.camera_info_subscription = self.create_subscription(CameraInfo, 'camera/depth/camera_info', self.camera_info_subscription_callback,qos_profile_sensor_data)

                self.pub_pose = self.create_publisher(DetectResult, '/person_detected', 2)

                parser = argparse.ArgumentParser()
                parser.add_argument("--config", default=r'/workspaces/capella_ros_docker/src/dabai_client/dabai_client/nanodet-plus-m_416.yml',
                                help="model config file path")
                parser.add_argument("--model", default=r'/workspaces/capella_ros_docker/src/dabai_client/dabai_client/nanodet-plus-m_416_checkpoint.ckpt',
                                help="model file path")
                parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
                # print('------------main entry')
                self.args = parser.parse_args(args=[])
                # print('*******************************')


                
                

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
                print('nanodet模型初始化完成')

                




        def color_subscription_callback(self, msg):
                self.color_width = msg.width
                self.color_height = msg.height
                self.color_data = np.array(msg.data).reshape((self.color_height,self.color_width,3))
                self.srcimg = cv2.cvtColor(self.color_data,cv2.COLOR_RGB2BGR)

                 # 计时
                start = time.perf_counter()
                # 读取图像
                # srcimg = cv2.imread(r'/workspaces/rknn-toolkit/nanodet-client/img3.jpg')
                # 预处理图片，得到图片预处理后的数据，只要数据，发送到服务端的还是原图
                meta, res = self.inference(self.srcimg)
                person_res = {}
                person_res[0] = res[0][0]
                # all_boxs：label, x0, y0, x1, y1, score
                result_frame,all_box = self.visualize(person_res, meta, cfg.class_names, 0.55)
                all_box = np.array(all_box)

                # result_ = np.fromstring(data).reshape((-1,6))
                print("延时：" + str(int((time.perf_counter() - start) * 1000)) + "ms")
                print(all_box)

                msg = DetectResult()
                
                single_msg = SingleDetector() 
                single_msg.part = False
                single_msg.x = 100.
                single_msg.y = 100.
                single_msg.z = 100.
                msg.result.append(single_msg) 
                
                if (len(all_box) > 0):
                        if(len(self.depth_data) > 0):
                                if(self.K != []):
                                        det_bboxes, det_conf, det_classid = all_box[:,1:5],all_box[:,5],all_box[:,0].astype(int)
                                        msg.result.clear()
                                        center_point_x = ((det_bboxes[:,2] + det_bboxes[:,0])/2).reshape((-1,1)).astype(int)
                                        center_point_y = ((det_bboxes[:,3]+det_bboxes[:,1])/2).reshape((-1,1)).astype(int)
                                        center_point = np.concatenate([center_point_x,center_point_y],axis=1)
                                        # print(center_point_x)
                                        # print(center_point_y)
                                        # center_point = np.array([[240,320]])
                                        # print(center_point)
                                        
                                        for x,y in center_point:
                                                # x = 320
                                                # y = 240
                                                single_msg = SingleDetector() 
                                                # y = int(480 / 416 * y) 
                                                # x = int(640 / 416 * x)
                                                z_ = (self.depth_data[y][x][1]*256 + self.depth_data[y][x][0])
                                                x_ = (x - self.K[2]) / self.K[0] * z_ 
                                                y_ = (y - self.K[5]) / self.K[4] * z_
                                                z_ /= 1000.
                                                x_ /= 1000.
                                                y_ /= 1000.
                                                # x_ = self.depth_data[y][x][:4]
                                                # y_ = self.depth_data[y][x][4:8]
                                                # z_ = self.depth_data[y][x][8:12]
                                                # depth_x = unpack("<f",pack('4B',*x_))[0]
                                                # depth_y = unpack("<f",pack('4B',*y_))[0]
                                                # depth_z = unpack("<f",pack('4B',*z_))[0]

                                                x_rot = z_
                                                y_rot = x_
                                                z_rot = y_
                                                x_unrot = x_rot / math.sqrt(2) + z_rot / math.sqrt(2) 
                                                y_unrot = y_rot
                                                z_unrot = x_rot / math.sqrt(2) - z_rot / math.sqrt(2) 
                                                
                                                print('******************', round(x_rot, 2), round(x_unrot,2))
                                                print('******************', round(y_rot, 2), round(y_unrot,2))
                                                print('******************', round(z_rot, 2), round(z_unrot,2))

                                                single_msg.part = False
                                                if z_== 0:
                                                        single_msg.part = False
                                                        single_msg.x = 100.
                                                        single_msg.y = 100.
                                                        single_msg.z = 100.
                                                        msg.result.append(single_msg)
                                                        
                                                else:
                                                        single_msg.x = x_unrot
                                                        single_msg.y = y_unrot
                                                        single_msg.z = z_unrot
                                                        msg.result.append(single_msg)
                                                        cv2.putText(self.srcimg,
                                                                    'x: ' + str(round(x_unrot ,2))+', y: ' + str(round(y_unrot, 2)),
                                                                    (x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255 ),thickness=1)
                
                self.pub_pose.publish(msg)
                cv2.imshow('img',self.srcimg)                
                cv2.waitKey(1)

        def depth_subscription_callback(self, msg):
                self.depth_width = msg.width
                self.depth_height = msg.height
                self.depth_data = np.array(msg.data).reshape((480,640,-1)).astype(int)
                # print(self.depth_data.shape)
                # self.color_data = np.array(msg.data).reshape((self.depth_height,self.depth_width,2))[:,:,0]

                # self.get_logger().info('depth height: "%s"' % self.depth_height)
                # self.get_logger().info('depth width: "%s"' % self.depth_width)
                # self.get_logger().info('depth data: "%s"' % self.depth_data[:16])
                # cv2.imshow('depth0',self.color_data)
                # cv2.waitKey(1)

                # center_point = np.array([[240,320]])
                # # print(center_point)
                # for x,y in center_point:
                #         x_ = self.depth_data[y][x][:4]
                #         y_ = self.depth_data[y][x][4:8]
                #         z_ = self.depth_data[y][x][8:12]
                #         depth_x = unpack("<f",pack('4B',*x_))[0]
                #         depth_y = unpack("<f",pack('4B',*y_))[0]
                #         depth_z = unpack("<f",pack('4B',*z_))[0]
                #         print(depth_x,depth_y,depth_z)

        def camera_info_subscription_callback(self, msg):
                self.K = msg.k

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
                meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
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

                                # cv2.imshow("det", result_frame)
                                ch = cv2.waitKey(1)
                                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                                        break
                        else:
                                break




def main(args=None):
        rclpy.init(args=args)

        minimal_subscriber = DaBaiSubscriber()

        rclpy.spin(minimal_subscriber)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
        main()









   