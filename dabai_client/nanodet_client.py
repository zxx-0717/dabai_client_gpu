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


                # 服务端ip地址
                HOST = '192.168.180.8'
                # 服务端端口号
                PORT = 8080
                ADDRESS = (HOST, PORT)

                # 创建一个套接字
                self.tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 连接远程ip
                print('------正在连接计算棒TCP服务------')
                while True:
                        try:
                                self.tcpClient.connect(ADDRESS)
                                break
                        except:
                                print('wait for TCP server!')
                                time.sleep(1)
                                continue
                print('------已连接计算棒TCP服务------')


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
                cv_image, newh, neww, top, left = pre_process(self.srcimg)
                # 压缩图像
                img_encode = cv2.imencode('.jpg', self.srcimg, [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
                # 转换为字节流
                bytedata = img_encode.tostring()
                # 标志数据，包括待发送的字节流长度等数据，用‘,’隔开
                flag_data = (str(len(bytedata))).encode() + ",".encode() + " ".encode()
                self.tcpClient.send(flag_data)
                # 接收服务端的应答
                data = self.tcpClient.recv(1024)
                if ("ok" == data.decode()):
                    # 服务端已经收到标志数据，开始发送图像字节流数据
                    self.tcpClient.send(bytedata)
                # 接收服务端的应答
                data = self.tcpClient.recv(1024)

                result_ = np.fromstring(data).reshape((-1,6))
                # print("延时：" + str(int((time.perf_counter() - start) * 1000)) + "ms")
                # print(result_)
                det_bboxes, det_conf, det_classid = result_[:,:4],result_[:,4],result_[:,5].astype(int)
                det_bboxes = det_bboxes[det_classid == 0]
                det_conf = det_conf[det_classid == 0]
                det_classid = det_classid[det_classid == 0]
                msg = DetectResult()
                
                single_msg = SingleDetector() 
                single_msg.part = False
                single_msg.x = 100.
                single_msg.y = 100.
                single_msg.z = 100.
                msg.result.append(single_msg) 
                
                if (len(det_classid) > 0):
                        if(len(self.depth_data) > 0):
                                if(self.K != []):
                                        msg.result.clear()
                                        self.srcimg = img_draw(self.srcimg,det_bboxes, det_conf, det_classid,newh, neww, top, left)
                                        center_point_x = ((det_bboxes[:,2] + det_bboxes[:,0])/2).reshape((-1,1)).astype(int)
                                        center_point_y = ((det_bboxes[:,3]+det_bboxes[:,1])/2).reshape((-1,1)).astype(int)
                                        center_point = np.concatenate([center_point_x,center_point_y],axis=1)
                                        # print(center_point_x)
                                        # print(center_point_y)
                                        # center_point = np.array([[240,320]])
                                        print(center_point)
                                        
                                        for x,y in center_point:
                                                # x = 320
                                                # y = 240
                                                single_msg = SingleDetector() 
                                                y = int(480 / 416 * y) 
                                                x = int(640 / 416 * x)
                                                z_ = (self.depth_data[y][x][1]*256 + self.depth_data[y][x][0])
                                                z_cos = z_ / math.sqrt(2)
                                                x_ = (x - self.K[2]) / self.K[0] * z_
                                                y_ = (y - self.K[5]) / self.K[4] * z_
                                                # x_ = self.depth_data[y][x][:4]
                                                # y_ = self.depth_data[y][x][4:8]
                                                # z_ = self.depth_data[y][x][8:12]
                                                # depth_x = unpack("<f",pack('4B',*x_))[0]
                                                # depth_y = unpack("<f",pack('4B',*y_))[0]
                                                # depth_z = unpack("<f",pack('4B',*z_))[0]
                                                print('------------------>',x_,y_,z_cos)
                                                single_msg.part = False
                                                if z_== 0:
                                                        single_msg.part = False
                                                        single_msg.x = 100.
                                                        single_msg.y = 100.
                                                        single_msg.z = 100.
                                                        msg.result.append(single_msg)
                                                else:
                                                        single_msg.x = z_cos / 1000.
                                                        single_msg.y = x_ / 1000.
                                                        single_msg.z = y_ / 1000.
                                                        msg.result.append(single_msg)
                
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









   