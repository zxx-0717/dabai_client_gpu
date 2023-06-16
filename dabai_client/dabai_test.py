import rclpy
import cv2
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from rclpy.qos import qos_profile_sensor_data

class DaBaiSubscriber(Node):

        def __init__(self):
                super().__init__('DaBai_Subscriber')
                self.color_subscription = self.create_subscription(Image,'/camera/color/image_raw',
                self.color_subscription_callback,qos_profile_sensor_data)
                self.color_subscription  # prevent unused variable warning

                self.depth_subscription = self.create_subscription(Image,'/camera/depth/image_raw',
                self.depth_subscription_callback,
                qos_profile_sensor_data)
                self.depth_subscription  # prevent unused variable warning

        def color_subscription_callback(self, msg):
                self.color_width = msg.width
                self.color_height = msg.height
                self.color_data = np.array(msg.data).reshape((self.color_height,self.color_width,3))
                self.color_data = cv2.cvtColor(self.color_data,cv2.COLOR_RGB2BGR)

                # self.get_logger().info('color_height: "%s"' % self.color_height)
                # self.get_logger().info('color_width: "%s"' % self.color_width)
                # self.get_logger().info('data: "%s"' % self.data)
                cv2.imshow('img',self.color_data)
                cv2.waitKey(1)

        def depth_subscription_callback(self, msg):
                self.depth_width = msg.width
                self.depth_height = msg.height
                self.depth_data = msg.data
                self.depth_data = np.array(msg.data).reshape((self.depth_height,self.depth_width,2))

                self.get_logger().info('depth height: "%s"' % self.depth_height)
                self.get_logger().info('depth width: "%s"' % self.depth_width)
                # print(self.depth_data.shape)
                depth_data16 = self.depth_data[:,:,0] + self.depth_data[:,:,1]*256
                # print(depth_data16)
                
                cv2.imshow('depth0',self.depth_data[:,:,0])
                depth2color = self.generate_false_map(self.depth_data[:,:,0])
                depth2color = np.array(depth2color).reshape(self.depth_height, self.depth_width, 3).astype(np.uint8)
                # print("depth2color.shape: ", depth2color.shape)
                cv2.imshow('depth00', depth2color)
                # cv2.imwrite('./1.jpg', depth2color)
                # print(depth2color)
                cv2.waitKey(1)

        def generate_false_map(self, src):
                dst = []
                max_val = 255.0
                map = [[0,0,0,114],[0,0,1,185],[1,0,0,114],[ 1,0,1,174],[0,1,0,114],[ 0,1,1,185],[1,1,0,114], [1,1,1,0]]
                sum = 0
                for i in range(8):
                        sum += map[i][3]
                weights = [0 for x in range(8)]
                cumsum = [0 for x in range(8)]
                for i in range(7):
                        weights[i] = sum / map[i][3]
                        cumsum[i+1] = cumsum[i] + map[i][3] / sum
                        # print('weight[', i,']: ', weights[i])
                        # print('cumsum[', i + 1,']: ', cumsum[i+1])
                height_ = src.shape[0]
                width_ = src.shape[1]

                for v in range(height_):
                        for u in range(width_):
                                val = min(max(src[v][u] / 255.0, 0.0), 1.0)
                                i = 0
                                for i in range(7):
                                        if val < cumsum[i + 1]:
                                                break
                                w = 1.0 - (val - cumsum[i]) * weights[i]
                                r = int((w * map[i][0] + (1.0 - w) * map[i + 1][0]) * 255.0)
                                g = int((w * map[i][1] + (1.0 - w) * map[i + 1][1]) * 255.0)
                                b = int((w * map[i][2] + (1.0 - w) * map[i + 1][2]) * 255.0)

                                r = min(max(r,0),255)
                                g = min(max(g,0),255)
                                b = min(max(b,0),255)

                                dst.append([b, g, r])

                return dst




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