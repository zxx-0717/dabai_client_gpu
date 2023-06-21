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
                parser.add_argument("--config", default=r'/workspaces/capella_ros_docker/src/dabai_client_gpu/dabai_client/nanodet-plus-m_416.yml',
                                help="model config file path")
                parser.add_argument("--model", default=r'/workspaces/capella_ros_docker/src/dabai_client_gpu/dabai_client/nanodet-plus-m_416_checkpoint.ckpt',
                                help="model file path")
                parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
                # print('------------main entry')
                self.args = parser.parse_args(args=[])
                # print('*******************************')
                self.delta_x = 0.25

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
                                        
                                        index = 0
                                        for x,y in center_point:
                                                # single_msg = SingleDetector() 
                                                # z_ = (self.depth_data[y][x][1]*256 + self.depth_data[y][x][0])
                                                # x_ = (x - self.K[2]) / self.K[0] * z_ 
                                                # y_ = (y - self.K[5]) / self.K[4] * z_
                                                # z_ /= 1000.
                                                # x_ /= 1000.
                                                # y_ /= 1000.
                                                # x_rot = z_
                                                # y_rot = x_
                                                # z_rot = y_
                                                # x_unrot = x_rot / math.sqrt(2) + z_rot / math.sqrt(2) 
                                                # y_unrot = y_rot
                                                # z_unrot = x_rot / math.sqrt(2) - z_rot / math.sqrt(2)

                                                x_unrot, y_unrot, z_unrot = self.getDepth_XYZ(x, y)
                                                print('x_unrot: ', x_unrot, 'y_unrot: ', y_unrot)
                                                x_coordinate_select = 0
                                                y_abs_min = 5.0

                                                x1 = int(det_bboxes[index][0])
                                                x2 = int(det_bboxes[index][2])
                                                y1 = int(det_bboxes[index][1])
                                                y2 = int(det_bboxes[index][3])
                                                cv2.line(self.srcimg, (0,y),(639,y), (0,0,255), thickness=1, lineType=cv2.LINE_AA)
                                                print('x1: ', x1, 'x2: ', x2)
                                                print('y1: ', y1, 'y2: ', y2)
                                                range_x12 = x2 - x1
                                                range_y12 = y2 - y1
                                                y_up = 3
                                                y_start = y - y_up
                                                for yyy in range(y_up * 2 + 1):
                                                        for xxx in range(range_x12):
                                                                xx,yy,zz = self.getDepth_XYZ(x1 + xxx, y_start + yyy)
                                                                if abs(xx - x_unrot) < 0.3:
                                                                        self.srcimg[y_start + yyy][x1 + xxx] = (0,255,0)
                                                all_y_on_center_line = []
                                                all_x_on_center_line = []
                                                for j in range(y_up * 2 + 1):
                                                        for i in range(range_x12):
                                                                x_tmp, y_tmp, z_tmp = self.getDepth_XYZ(i + x1, y_start + j)
                                                                all_y_on_center_line.append(round(y_tmp,2))
                                                                all_x_on_center_line.append(round(x_tmp,2))
                                                                if abs(x_tmp - x_unrot) < self.delta_x:
                                                                        if abs(y_abs_min) > abs(y_tmp):
                                                                                y_abs_min = y_tmp
                                                                                x_coordinate_select = i + x1

                                                # print('****** x *******')
                                                # print(all_x_on_center_line)
                                                # print('****** y *******')
                                                # # print(all_y_on_center_line)
                                                # print('x_coordinate_select: ', x_coordinate_select)
                                                # print('====================================================')
                                                index += 1
                                                single_msg.part = False
                                                if x_unrot == 0:
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
                                                                    'x: ' + str(round(x_unrot ,2))+', y: ' + str(round(y_abs_min, 2)),
                                                                    (x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255 ),thickness=1)
                                                        cv2.circle(self.srcimg, (x_coordinate_select, y), 8, (0,0,255), thickness=1)
                
                self.pub_pose.publish(msg)
                cv2.imshow('img',self.srcimg)                
                cv2.waitKey(1)

        def getDepth_XYZ(self, x, y): # x => 深度， y => 水平偏移， z => 垂直偏移
                z_ = (self.depth_data[y][x][1]*256 + self.depth_data[y][x][0])
                x_ = (x - self.K[2]) / self.K[0] * z_ 
                y_ = (y - self.K[5]) / self.K[4] * z_
                z_ /= 1000.
                x_ /= 1000.
                y_ /= 1000.
                x_rot = z_
                y_rot = x_
                z_rot = y_
                x_unrot = x_rot / math.sqrt(2) + z_rot / math.sqrt(2) 
                y_unrot = y_rot
                z_unrot = x_rot / math.sqrt(2) - z_rot / math.sqrt(2) 

                return x_unrot, y_unrot, z_unrot



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









   