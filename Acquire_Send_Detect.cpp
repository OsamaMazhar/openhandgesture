/* This file is part of openhandgesture program.
* Program description : This program acquires frames from Kinect V2,
* send them to skeleton extractor, receives skeleton coordinates from
* skeleton extractor, crops hand bounding box, pass it through our
* hand gestures detector and outputs the gesture class.

* For more description and citation:

@article{mazhar2019real,
  title={A real-time human-robot interaction framework with robust background invariant hand gesture detection},
  author={Mazhar, Osama and Navarro, Benjamin and Ramdani, Sofiane and Passama, Robin and Cherubini, Andrea},
  journal={Robotics and Computer-Integrated Manufacturing},
  volume={60},
  pages={34--48},
  year={2019},
  publisher={Elsevier}
}

@inproceedings{mazhar2018towards,
  title={Towards real-time physical human-robot interaction using skeleton information and hand gestures},
  author={Mazhar, Osama and Ramdani, Sofiane and Navarro, Benjamin and Passama, Robin and Cherubini, Andrea},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}

* Copyright (C) 2019 -  Osama Mazhar (osamazhar@yahoo.com). All Right reserved.
*
* openhandgesture is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* Foobar is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with openhandgesture.  If not, see <https://www.gnu.org/licenses/>.
*/

// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
// #include <glog/logging.h> // google::InitGoogleLogging //this clashes with tensorflow logging

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string>     // std::string, std::to_string
#include <vector>
#include <chrono>
#include <time.h>
#include <iterator>

#include <nnxx/message.h>
#include <nnxx/socket.h>
#include <nnxx/reqrep.h>
#include <nnxx/testing>
#include <nnxx/pubsub.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/stitching/detail/camera.hpp"

// For Kalman Filter
#include "opencv2/video/tracking.hpp"

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <signal.h>

// Tensorflow includes
#include <fstream>
#include <utility>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// #include "webcam_generated.h"
#include "./essentials/one_skeleton_generated.h"
#include "./essentials/gesture-recognition-request_generated.h"

#define drawCross( center, color, d )                                 \
cv::line( display_image, cv::Point( center.x - d, center.y - d ), cv::Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
cv::line( display_image, cv::Point( center.x + d, center.y - d ), cv::Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

constexpr size_t RX_MAX_PACKET_SIZE = 1000;     // 1 Kb in decimal
constexpr size_t TX_MAX_PACKET_SIZE = 10000000; // 10 Mb

//! [context]
libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *dev = 0;
libfreenect2::PacketPipeline *pipeline = 0;
//! [context]

bool protonect_shutdown = false; // Whether the running application should shut down.

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

void checkStatus(const tensorflow::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(1);
  }
}

tensorflow::Status ReadLabelsFile(const tensorflow::string& file_name, std::vector<tensorflow::string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }

  result->clear();
  tensorflow::string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return tensorflow::Status::OK();
}

void draw_rectangle(double angle, unsigned int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image, cv::Mat depthimage, cv::Mat* cropped_hand, cv::Mat* cropped_hand_depth)
{
	cv::Mat resized_rgb_crop, resized_depth_crop, depth_cropped;
  cv::Size size_of_initial_crop = cv::Size(rec_size*2, rec_size*2);
  cv::Mat aligned_bounding_rect;
  cv::Mat Rotation_Matrix, bounding_rect_rotated;
  cv::Mat rgb_cropped = cv::Mat(size_of_initial_crop, CV_32FC3);

	cv::RotatedRect rotatedRectangle = cv::RotatedRect(joint_pt, size_of_initial_crop, angle);
  cv::Rect diagonal_rectangle = rotatedRectangle.boundingRect();

  cv::Size rotatedRectangle_size = rotatedRectangle.size;
  cv::Size diagonal_rectangle_size = diagonal_rectangle.size();

  cv::Point2f points[4];
  rotatedRectangle.points(points);

  std::vector<std::vector<cv::Point>> pts = { { points[0], points[1], points[2], points[3] } };

	cv::cvtColor(source_image, source_image, CV_BGRA2BGR);

  cv::Mat depth_aligned_rect_pixels;
  cv::Mat depth_bounding_rect_rotated;

  cv::getRectSubPix(source_image, diagonal_rectangle_size, rotatedRectangle.center, aligned_bounding_rect);
  cv::getRectSubPix(depthimage, diagonal_rectangle_size, rotatedRectangle.center, depth_aligned_rect_pixels);

  cv::Point aligned_bounding_rect_center;

  aligned_bounding_rect_center.x = aligned_bounding_rect.cols / 2;
  aligned_bounding_rect_center.y = aligned_bounding_rect.rows / 2;

  Rotation_Matrix = getRotationMatrix2D(aligned_bounding_rect_center, angle + 90, 1.0);
  cv::warpAffine(aligned_bounding_rect, bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);
  cv::warpAffine(depth_aligned_rect_pixels, depth_bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);

  cv::getRectSubPix(bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, rgb_cropped);
  cv::getRectSubPix(depth_bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, depth_cropped);

  resized_rgb_crop = cv::Mat(cv::Size(224,224), CV_32FC3); // output resized is 244X244
	cv::resize(rgb_cropped, resized_rgb_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);
  cv::resize(depth_cropped, resized_depth_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);

  cv::line(input_image, points[0], points[1], cv::Scalar(0, 255, 0), 3);
  cv::line(input_image, points[1], points[2], cv::Scalar(0, 255, 0), 3);
  cv::line(input_image, points[2], points[3], cv::Scalar(0, 255, 0), 3);
  cv::line(input_image, points[3], points[0], cv::Scalar(0, 255, 0), 3);

  *cropped_hand_depth = resized_depth_crop;
	*cropped_hand = resized_rgb_crop;
}

void getmeanvalue(cv::Point current_point, int area_length, float* depth_val, cv::Mat src_d){
  int index_x, index_y;
  float mean_depth = 0;
  for (int i=0; i<area_length; i++){
    index_x = current_point.x - area_length/2 + i;
    for(int j=0; j<area_length; j++)
      {
        index_y = current_point.y - area_length/2 + j;
        mean_depth = mean_depth + src_d.at<float>(index_y, index_x);
      }
    }
  *depth_val = mean_depth / (float)(area_length * area_length) * 1000;
}

tensorflow::Tensor cvmat_to_tensor(cv::Mat input_image, cv::Size cv_image_size, tensorflow::int32 input_width, tensorflow::int32 input_height, float input_mean, float input_std)
{
  cv::Mat resized_image;
  cv::Mat floatimage;

  cv::resize(input_image,resized_image,cv_image_size,0,0,cv::INTER_AREA);

  // Initializing a Tensor for cv_image data transfer
  int number_of_channels = resized_image.channels();

  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,input_height,input_width,number_of_channels}));
  // get a tensor_map for data transfer
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  // Normalizing the image
  resized_image.convertTo(floatimage, CV_32FC1);
  resized_image = floatimage;

  floatimage = floatimage - input_mean;
  floatimage = floatimage / input_std;

  // Preparing to transfer data from cv read image (floatimage) to the tensor
  const float * source_data = (float*) floatimage.data;

  // Tensorflow detection starts here!
  // Transfering the data from the cv read image to the tensor
  for (int y = 0; y < input_height; ++y) {
    const float* source_row = source_data + (y * input_width * number_of_channels);
    for (int x = 0; x < input_width; ++x) {
      const float* source_pixel = source_row + (x * number_of_channels);
      const float* source_B = source_pixel + 0;
      const float* source_G = source_pixel + 1;
      const float* source_R = source_pixel + 2;

      input_tensor_mapped(0, y, x, 0) = *source_R;
      input_tensor_mapped(0, y, x, 1) = *source_G;
      input_tensor_mapped(0, y, x, 2) = *source_B;
    }
  }

return input_tensor;
}

// >>> Uncomment these functions to convert cameraparas into cv matrices
// cv::Mat Freenect2IrParams_to_cv_Mat(libfreenect2::Freenect2Device::IrCameraParams LocalIrCP)
//   {
//     cv::Mat IrCameraParams_Mat = (cv::Mat_<double>(3,3) <<  LocalIrCP.fx,
//                                                             LocalIrCP.fy,
//                                                             LocalIrCP.cx,
//                                                             LocalIrCP.cy,
//                                                             LocalIrCP.k1,
//                                                             LocalIrCP.k2,
//                                                             LocalIrCP.k3,
//                                                             LocalIrCP.p1,
//                                                             LocalIrCP.p2);
//     return IrCameraParams_Mat;
//   }
// cv::Mat Freenect2ColorParams_to_cv_Mat(libfreenect2::Freenect2Device::ColorCameraParams LocalCCP)
//   {
//     cv::Mat Kinect2ColorParams_Mat = (cv::Mat_<double>(6,5) <<  LocalCCP.fx,
//                                                                 LocalCCP.fy,
//                                                                 LocalCCP.cx,
//                                                                 LocalCCP.cy,
//                                                                 LocalCCP.shift_d,
//                                                                 LocalCCP.shift_m,
//                                                                 LocalCCP.mx_x3y0,
//                                                                 LocalCCP.mx_x0y3,
//                                                                 LocalCCP.mx_x2y1,
//                                                                 LocalCCP.mx_x1y2,
//                                                                 LocalCCP.mx_x2y0,
//                                                                 LocalCCP.mx_x0y2,
//                                                                 LocalCCP.mx_x1y1,
//                                                                 LocalCCP.mx_x1y0,
//                                                                 LocalCCP.mx_x0y1,
//                                                                 LocalCCP.mx_x0y0,
//                                                                 LocalCCP.my_x3y0,
//                                                                 LocalCCP.my_x0y3,
//                                                                 LocalCCP.my_x2y1,
//                                                                 LocalCCP.my_x1y2,
//                                                                 LocalCCP.my_x2y0,
//                                                                 LocalCCP.my_x0y2,
//                                                                 LocalCCP.my_x1y1,
//                                                                 LocalCCP.my_x1y0,
//                                                                 LocalCCP.my_x0y1,
//                                                                 LocalCCP.my_x0y0,
//                                                                 0,
//                                                                 0,
//                                                                 0,
//                                                                 0);
//     return Kinect2ColorParams_Mat;
//   }
// <<< Uncomment these functions to convert cameraparas into cv matrices

int main(int argc, char *argv[])
  {
    // Lets create a map for Skeletal Joint Names
    std::map<int, std::string> Joint_name;
    Joint_name[0] = "Nose";
    Joint_name[1] = "Neck";
    Joint_name[2] = "RShoulder";
    Joint_name[3] = "RElbow";
    Joint_name[4] = "RWrist";
    Joint_name[5] = "LShoulder";
    Joint_name[6] = "LElbow";
    Joint_name[7] = "LWrist";
    Joint_name[8] = "RHip";
    Joint_name[9] = "RKnee";
    Joint_name[10] = "RAnkle";
    Joint_name[11] = "LHip";
    Joint_name[12] = "LKnee";
    Joint_name[13] = "LAnkle";

    // Initializing google logging (Caffe uses it for logging)
    // google::InitGoogleLogging("openPoseKinectCustom");
    auto start_time = std::chrono::high_resolution_clock::now();
    namespace tf = tensorflow;

    tf::string graph = "./essentials/Sep_data_augment_10_hand_g.pb";
    tf::string labels_file_name = "./essentials/indices_10.txt";

    tf::int32 input_width = 224;
    tf::int32 input_height = 224;
    float input_mean = 0;
    float input_std = 255;
    // tf::string input_layer = "firstConv2D_input";
    // tf::string output_layer = "k2tfout_0";
    tf::string input_layer = "input_1";
    tf::string output_layer = "dense_2_0";

    std::vector<tensorflow::Flag> flag_list = {
        tensorflow::Flag("graph", &graph, "graph to be executed"),
        tensorflow::Flag("labels", &labels_file_name, "name of file containing labels"),
        tensorflow::Flag("input_width", &input_width, "resize image to this width in pixels"),
        tensorflow::Flag("input_height", &input_height,
             "resize image to this height in pixels"),
        tensorflow::Flag("input_mean", &input_mean, "scale pixel values to this mean"),
        tensorflow::Flag("input_std", &input_std, "scale pixel values to this std deviation"),
        tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
        tensorflow::Flag("output_layer", &output_layer, "name of output layer"),
    };

    tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
      LOG(ERROR) << usage;
      return -1;
    }

    // We need to call this to set up global state for TensorFlow.
    tensorflow::port::InitMain(argv[0], &argc, &argv);
    if (argc > 1) {
      LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
      return -1;
    }

    std::cout << "Loaded graph location: " << graph << std::endl;

    // >>> nanomsgxx sockets and addresses initialization
    nnxx::socket Coordinate_rep_socket { nnxx::SP, nnxx::REP };
    nnxx::socket Frame_req_socket { nnxx::SP, nnxx::REP };
    nnxx::socket Frame_data_socket { nnxx::SP, nnxx::REQ };

    const char *Coordinate_rep_addr = "ipc://./essentials/Coordinate_req.ipc";
    const char *Frame_req_addr = "ipc://./essentials/kinect.ipc";
    const char *Frame_data_addr = "ipc://./essentials/frame_data.ipc";

    auto Coordinate_rep_socket_id = Coordinate_rep_socket.bind(Coordinate_rep_addr);
    auto Frame_req_socket_id = Frame_req_socket.bind(Frame_req_addr);
    auto Frame_data_socket_id = Frame_data_socket.connect(Frame_data_addr);

    // >>> Nanomsg for Robot initializtion
    auto socket = nnxx::socket { nnxx::SP, nnxx::PUB };
    std::string address = "tcp://*:4751";
  	std::cout << address << std::flush;
  	auto socket_id = socket.bind(address);
  	std::cout << " (" << socket_id << ")\n";
    // <<< Nanomsg for Robot initializtion
    // <<< nanomsgxx sockets and addresses initialization

    namespace tf = tensorflow;

    // Tensorflow Initialization
    auto options = tensorflow::SessionOptions();
    tf::Status status;
    tf::GraphDef graph_def;
    status = ReadBinaryProto(tf::Env::Default(), graph, &graph_def);
    checkStatus(status);

    // options.config.mutable_gpu_options()->set_visible_device_list("0");
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.3);
    options.config.mutable_gpu_options()->set_allow_growth(true);

    tf::Session* session;
    status = tf::NewSession(options, &session);
    checkStatus(status);
    status = session->Create(graph_def);
    checkStatus(status);

    cv::Mat resized_image;
    cv::Mat floatimage;

    // Resize cv_image to the desired height and width
    cv::Size cv_image_size(input_height,input_width);

    // Initializing a Tensor for cv_image data transfer
    int number_of_channels = resized_image.channels();
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,input_height,input_width,number_of_channels}));
    std::vector<tensorflow::string> labels;
    size_t label_count;

    // Read labels file // From the provided Function
    tensorflow::Status read_labels_status = ReadLabelsFile(labels_file_name, &labels, &label_count);

    std::vector<tensorflow::Tensor> outputs;

    // We need to Run this network (even with empty input_tensor) once here to avoid CUDA memory errors while Running the network in the loop
    session->Run({{input_layer, input_tensor}}, {output_layer}, {}, &outputs);

    // >>> Kinect2 (libfreenect2) Initialization
    std::cout << "Streaming from Kinect One sensor!" << std::endl;

    //! [discovery]
    if(freenect2.enumerateDevices() == 0) {
        std::cout << "no device connected!" << std::endl;
        return -1;
      }
    std::string serial = freenect2.getDefaultDeviceSerialNumber();
    std::cout << "SERIAL: " << serial << std::endl;
    if(pipeline) {
        //! [open]
        dev = freenect2.openDevice(serial, pipeline);
        //! [open]
      } else {
        dev = freenect2.openDevice(serial);
    }

    if(dev == 0){
      std::cout << "failure opening device!" << std::endl;
      return -1;
    }

    signal(SIGINT, sigint_handler);
    protonect_shutdown = false;

    //! [listeners]
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color |
                                                  libfreenect2::Frame::Depth |
                                                  libfreenect2::Frame::Ir);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    //! [listeners]

    //! [start]
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
    //! [start]

    libfreenect2::Freenect2Device::IrCameraParams LocalIrCP = dev->getIrCameraParams();
    libfreenect2::Freenect2Device::ColorCameraParams LocalCCP = dev->getColorCameraParams();

    // >>> Lines to save IR and Color Camera Parameters
    // >>> Uncomment the functions also to use the following:
    // cv::Mat IrParams_Mat = Freenect2IrParams_to_cv_Mat(LocalIrCP);
    // cv::Mat ColorParams_Mat = Freenect2ColorParams_to_cv_Mat(LocalCCP);
    // cv::FileStorage fs("Kinect2_CameraParams.yml", cv::FileStorage::WRITE);
    // time_t rawtime; time(&rawtime);
    // fs << "CalibrationDate" << asctime(localtime(&rawtime));
    // fs << "Kinect2IrCameraParams" << IrParams_Mat ;
    // fs << "Kinect2ColorCameraParams" << ColorParams_Mat ;
    // fs.release();
    // std::cout <<"Camera Ir Parameters: " << IrParams_Mat << std::endl;
    // std::cout <<"Camera Color Parameters: " << ColorParams_Mat << std::endl;
    // <<< Lines to save IR and Color Camera Parameters

    //! [registration setup]
    libfreenect2::Registration* registration = new libfreenect2::Registration(LocalIrCP, LocalCCP);
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4); // check here (https://github.com/OpenKinect/libfreenect2/issues/337) and here (https://github.com/OpenKinect/libfreenect2/issues/464) why depth2rgb image should be bigger
    // 2 extra lines in the bigdepth image are basically a 1-pixel border on the bottom and the top, so the actual data is from (0,1) to (1920,1081).
    //! [registration setup]

    cv::Mat rgb_actual, depth_registered_with_rgb, depth_actual;

    // <<< Kinect2 (libfreenect2) Initialization

    cv::namedWindow("Transmitted Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Transmitted Image", 960, 540);
    auto buffer = new unsigned char[RX_MAX_PACKET_SIZE];

    std::vector<cv::Point3i> temp_coordinate_vector;
    std::vector<cv::Point3i> skeletal_coordinates, previous_skeletal_coordinates;
    cv::Point3i temp_coordinate;

    cv::Point fps_display_position;
    fps_display_position.x = 1400;
    fps_display_position.y = 100;

    cv::Point Distance_display_position;
    Distance_display_position.x = 1400;
    Distance_display_position.y = 200;

    cv::Point TopLabel_display_position;
    TopLabel_display_position.x = 1000;
    TopLabel_display_position.y = 300;

    // >>> For averaging FPS on display
    const int numReadings = 33;
    int readings[numReadings];      // the readings from the analog input
    int readIndex = 0;
    int total = 0;                  // the running total
    int average_fps = 0;            // the average

    int distance_readings[numReadings];      // the readings from the analog input
    int distance_readIndex = 0;
    int distance_total = 0;                  // the running total
    int average_distance = 0;            // the average



    for (int thisReading = 0; thisReading < numReadings; thisReading++) {
        readings[thisReading] = 0;
      }
    // <<< For averaging FPS on display

    for (int thisReading = 0; thisReading < numReadings; thisReading++) {
        distance_readings[thisReading] = 0;
      }

    cv::Point current_point_2d;
    float depth_val;
    int mean_area_length = 6;
    unsigned int depth_round;
    cv::Mat cropped_Left, cropped_Right;  // Their addresses will be sent to the draw_rectangle function
    cv::Mat cropped_R_float;
    // int bbox_tuning = 65; // tune the size of bounding box on the hand
    int bbox_tuning = 75; // tune the size of bounding box on the hand
    cv::Point right_elbow_point, left_elbow_point;
    double forearmlength;
    cv::Point hand_center;
    float rec_scale_f;
    unsigned int rec_scale;
    cv::Mat cropped_Left_depth, cropped_Right_depth;
    int bytes_sent;

    // >>> Kalman Filter Initialization
    int stateSize = 4;  // [x, y, v_x, v_y]
    int measSize = 2;   // [z_x, z_y] // we will only measure mouse cursor x and y
    int contrSize = 0;  // no control input

    unsigned int F_type = CV_32F;

    // initiation of OpenCV Kalman Filter
    cv::KalmanFilter KF(stateSize, measSize, contrSize, F_type);

    // creating state vector
    cv::Mat state(stateSize, 1, F_type);  // [x, y, v_x, v_y] // column Matrix

    // creating measurement vector
    cv::Mat meas(measSize, 1, F_type);    // [z_x, z_y] // column matrix

    cv::setIdentity(KF.transitionMatrix);

    KF.measurementMatrix = cv::Mat::zeros(measSize, stateSize, F_type);
    KF.measurementMatrix.at<float>(0) = 1.0f;
    KF.measurementMatrix.at<float>(5) = 1.0f;

    KF.processNoiseCov.at<float>(0) = 1e-2; // Try changing this
    KF.processNoiseCov.at<float>(5) = 1e-2;
    KF.processNoiseCov.at<float>(10) = 5.0f;
    KF.processNoiseCov.at<float>(15) = 5.0f;

    // Measure Noise Covariance Matrix
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar(1e-1));
    double ticks = 0;
    // <<< Kalman Filter Initialization
    int wait_timeout = 0; // For Clearing Skeletal Coordinates if no frames are received

    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    char log_file_name [120];
    int Year = now->tm_year - 100 + 2000;
    int Month = now->tm_mon + 1;
    int Day = now->tm_mday;
    int Hour = now->tm_hour;
    int Minutes = now->tm_min;
    // strftime (date_buffer,80,"gestures_log_%Y-%m-%d-%s",now);
    std::string log_file_buffer = "./gestures_logs/gestures_log_%d_%d_%d_%d_%d.txt";
    sprintf(log_file_name, log_file_buffer.c_str(), Year, Month, Day, Hour, Minutes);

    std::ofstream gesture_log_file;
    gesture_log_file.open (log_file_name);
    gesture_log_file << "Time Elapsed; Gesture Detected; Score; Time Taken\n";

    int minimum_distance_temp = 5000; // initialize with a large values to prevent 0 detection
    int min_distance_index, minimum_distance;

    std::string TopLabel, MinimumDistanceStr;

    while(!protonect_shutdown)
    {
      // >>> Kalman Filter Predict Part
      double precTick = ticks;
      ticks = (double) cv::getTickCount();
      double dT = (ticks - precTick) / cv::getTickFrequency(); // seconds

      // >>> Kalman Prediction
      // >>> Matrix A
      KF.transitionMatrix.at<float>(2) = dT;
      KF.transitionMatrix.at<float>(7) = dT;
      // <<< Matrix A

      // std::cout << "dt: " << dT << std::endl;

      state = KF.predict(); // First predict, to update the internal statePre variable
      // std::cout << "State post: " << state << std::endl;

      cv::Point predictPt(state.at<float>(0), state.at<float>(1));
      // <<< Kalman Prediction
      // <<< Kalman Filter Predict Part

      auto t0 = std::chrono::high_resolution_clock::now();

      listener.waitForNewFrame(frames);
      libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
      libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

      cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgb_actual);

      //! [registration]
      registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);
      //! [registration]

      cv::Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(depth_registered_with_rgb);
      cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, (depth2rgb.data)).copyTo(depth_actual);

      cv::Mat current_frame = rgb_actual;
      cv::Mat src_d = depth_actual.clone() / 4096.0f;
      cv::Mat inputImage;
      cv::Mat source_image_original = current_frame.clone();
      cv::cvtColor(current_frame, inputImage, cv::COLOR_RGB2BGR); //openpose requires BGR image format

      // >>> FlatBuffer Initialization

      flatbuffers::FlatBufferBuilder skeleton_builder(RX_MAX_PACKET_SIZE);
      flatbuffers::FlatBufferBuilder Kinect_Frames_builder(TX_MAX_PACKET_SIZE);

      auto serialize_frame = [&Kinect_Frames_builder](libfreenect2::Frame* frame)
			{
				auto data = Kinect_Frames_builder.CreateVector(frame->data, frame->width*frame->height*frame->bytes_per_pixel);
				return gesture_recognition::Createkinect2_frame(
					Kinect_Frames_builder,
					frame->width,
					frame->height,
					frame->bytes_per_pixel,
					data,
					frame->timestamp,
					frame->sequence,
					frame->exposure,
					frame->gain,
					frame->gamma,
					frame->status,
					frame->format
          );
				};

      auto rgb_serialized = serialize_frame(rgb);
      auto depth_serialized = serialize_frame(depth);
  		auto request = gesture_recognition::Createrequest_identification(Kinect_Frames_builder, rgb_serialized, depth_serialized);
      Kinect_Frames_builder.Finish(request);
      // std::cout << "request builder size: " << Kinect_Frames_builder.GetSize() << std::endl;

      cv::Mat display_image = current_frame.clone();
      nnxx::message frame_request {Frame_req_socket.recv(nnxx::DONTWAIT)};
      std::string string_message(frame_request.begin(), frame_request.end());
      // std::cout << "Message is: " << string_message.length() << std::endl;
      if(string_message == "Frame_req")
        {
          std::cout << "........................" << std::endl;
          std::cout << ".Frame Request Received." << std::endl;
          std::cout << "........................" << std::endl;
          bytes_sent = Frame_data_socket.send((const void *)Kinect_Frames_builder.GetBufferPointer(), Kinect_Frames_builder.GetSize());
          nnxx_check(bytes_sent == Kinect_Frames_builder.GetSize());
          std::cout << ".......Frame Sent......." << std::endl;
          std::cout << "........................" << std::endl;
          // std::cout << bytes_sent << " bytes sent!" << std::endl;
        }
      auto co_bytes_received = Coordinate_rep_socket.recv(buffer, RX_MAX_PACKET_SIZE, nnxx::DONTWAIT);
      // std::cout << "Coordinates bytes received: " << co_bytes_received << std::endl;
      previous_skeletal_coordinates = skeletal_coordinates;
      if(co_bytes_received > 0)
        {
          auto coordinate_request = op_skeleton::Getone_skeleton(buffer);
          auto name = coordinate_request->name()->c_str();
          auto number_of_points = coordinate_request->number_of_points();
          for(int i = 0; i < (int)number_of_points; i++)
            {
              auto point = coordinate_request->coordinates()->Get(i);

              temp_coordinate.x = point->x();
              temp_coordinate.y = point->y();
              temp_coordinate.z = point->z();
              temp_coordinate_vector.push_back(temp_coordinate);
            }
          skeletal_coordinates = temp_coordinate_vector;
          // std::cout << "Size of point vector: " << skeletal_coordinates.size() << std::endl;
        }

      // When the skeleton extractor do not request frames (has stopped its execution)
      // the last skeleton coordinates retrieved are kept. So we want to wipe them
      // If after 20 different cycles the skeletal coordinates are not updated.
      if(skeletal_coordinates == previous_skeletal_coordinates)
        {
          wait_timeout++;
          if(wait_timeout > 20)
            {
              wait_timeout = 0;
              skeletal_coordinates.clear();
            }
        }

      for(int i = 0; i < skeletal_coordinates.size(); i++)
        {
          // std::cout << "Distance Coordinate " << i << ": " << skeletal_coordinates[i].z << std::endl;
          if(skeletal_coordinates[i].z < minimum_distance_temp && skeletal_coordinates[i].z != 0)
            {
              min_distance_index = i;
              minimum_distance_temp = skeletal_coordinates[i].z;
            }
        }
      minimum_distance = minimum_distance_temp;
      MinimumDistanceStr = std::to_string(minimum_distance);
      std::cout << "Minimum Distance is: " << minimum_distance << std::endl;
      std::cout << "Minimum Distance Joint is: " << Joint_name[min_distance_index] << std::endl;

      // auto min_index = std::min_element(joint_distances.begin(), joint_distances.end());
      // std::cout << "Minimum Value is: " << std::distance(joint_distances.begin(), min_index) << std::endl;

      // The skeletal coordinates are received here, so now we proceed with
      // printing them on the image, drawing bounding box, and cropping the
      // hand images to pass them through the CNN.

      for( int joint_point_loop = 0; joint_point_loop < skeletal_coordinates.size(); joint_point_loop++) // -5 as we don't want last five points to show
        {
          // In this loop, each joint will be printed, bouding box will be drawn
          // and hand crops will be forwarded to tensorflow CNN.
          if(joint_point_loop == 9 || joint_point_loop == 10 || joint_point_loop == 12 || joint_point_loop == 13) // Skippinh lower body joints
            continue;     // we don't want these joints of lower extremity to be printed/taken care of.
          current_point_2d.x = skeletal_coordinates[joint_point_loop].x;
          current_point_2d.y = skeletal_coordinates[joint_point_loop].y;
          depth_val = skeletal_coordinates[joint_point_loop].z;
          // if(joint_point_loop == 3){
          //   right_elbow_point.x = skeletal_coordinates[joint_point_loop].x; // 3 is for right-elbow;
          //   right_elbow_point.y = skeletal_coordinates[joint_point_loop].y;
          //   }
          // if(joint_point_loop == 4 && current_point_2d.x != 0 && current_point_2d.y != 0)
          // {
          //   cv::line(display_image, right_elbow_point, current_point_2d, cv::Scalar(0,0,255), 2);
          //   double Angle = atan2(current_point_2d.y - right_elbow_point.y,current_point_2d.x - right_elbow_point.x) * 180.0 / CV_PI;
          //
          //   forearmlength = sqrt(pow(right_elbow_point.x - current_point_2d.x, 2.0) + pow(right_elbow_point.y - current_point_2d.y, 2.0));
          //   hand_center.x = current_point_2d.x + (current_point_2d.x - right_elbow_point.x) / forearmlength * (forearmlength / 2);
          //   hand_center.y = current_point_2d.y + (current_point_2d.y - right_elbow_point.y) / forearmlength * (forearmlength / 2);
          //   rec_scale_f = bbox_tuning / depth_val * 500;
          //   rec_scale = rec_scale_f;
          //   if(current_point_2d.x != 0 && current_point_2d.y !=0 && rec_scale != 0)
          //     {
          //       draw_rectangle(Angle, rec_scale, hand_center, display_image, source_image_original, src_d, &cropped_Right, &cropped_Right_depth);
          //       cv::imshow("Cropped Right", cropped_Right);
          //
          //       input_tensor = cvmat_to_tensor(cropped_Right, cv_image_size, input_width, input_height, input_mean, input_std);
          //       session->Run({{input_layer, input_tensor}}, {output_layer}, {}, &outputs);
          //       std::vector<float> result;
          //       for(int i=0; i<4; i++)
          //         result.push_back(outputs[0].flat<float>()(i));
          //       auto largest = std::max_element(std::begin(result), std::end(result));
          //       std::cout << "Max score is " << *largest << " and Gesture detected is " << labels[std::distance(std::begin(result), largest)] << std::endl;
          //     }
          //   }
            if(joint_point_loop == 6){
              left_elbow_point.x = skeletal_coordinates[joint_point_loop].x; // 3 is for right-elbow;
              left_elbow_point.y = skeletal_coordinates[joint_point_loop].y;
              }
            if(joint_point_loop == 7 && current_point_2d.x != 0 && current_point_2d.y != 0)
            {
              cv::line(display_image, left_elbow_point, current_point_2d, cv::Scalar(0,0,255), 2);
              double Angle = atan2(current_point_2d.y - left_elbow_point.y,current_point_2d.x - left_elbow_point.x) * 180.0 / CV_PI;
              std::cout << "Angle is: " << Angle << std::endl;
              forearmlength = sqrt(pow(left_elbow_point.x - current_point_2d.x, 2.0) + pow(left_elbow_point.y - current_point_2d.y, 2.0));
              hand_center.x = current_point_2d.x + (current_point_2d.x - left_elbow_point.x) / forearmlength * (forearmlength / 2);
              hand_center.y = current_point_2d.y + (current_point_2d.y - left_elbow_point.y) / forearmlength * (forearmlength / 2);
              rec_scale_f = bbox_tuning / depth_val * 500;
              rec_scale = rec_scale_f;
              if(current_point_2d.x != 0 && current_point_2d.y !=0 && rec_scale != 0)
                {
                  // >>> Passing the measured values to the measurement vector
                  meas.at<float>(0) = current_point_2d.x;
                  meas.at<float>(1) = current_point_2d.y;
                  // <<< Passing the measured values to the measurement vector

                  // >>> Kalman Update Phase
                  cv::Mat estimated = KF.correct(meas);

                  cv::Point statePt(estimated.at<float>(0),estimated.at<float>(1));
                  cv::Point measPt(meas.at<float>(0),meas.at<float>(1));
                  // <<< Kalman Update Phase

                  drawCross( statePt, cv::Scalar(255,255,255), 5 );

                  draw_rectangle(Angle, rec_scale, hand_center, display_image, source_image_original, src_d, &cropped_Left, &cropped_Left_depth);
                  cv::imshow("Cropped Right", cropped_Left);
                  if(Angle < -40)
                    {
                      input_tensor = cvmat_to_tensor(cropped_Left, cv_image_size, input_width, input_height, input_mean, input_std);
                      session->Run({{input_layer, input_tensor}}, {output_layer}, {}, &outputs);
                      std::vector<float> result;
                      for(int i=0; i<10; i++)
                        result.push_back(outputs[0].flat<float>()(i));
                      auto largest = std::max_element(std::begin(result), std::end(result));
                      TopLabel = labels[std::distance(std::begin(result), largest)];
                      std::cout << "Max score is " << *largest << " and Gesture detected is " << TopLabel << std::endl;
                      // Calculating Times
                      auto time_elap = std::chrono::high_resolution_clock::now();
                      std::chrono::duration<double, std::milli> Time_Elapsed = time_elap - start_time;
                      std::chrono::duration<double, std::milli> gesture_time = time_elap - t0;
                      std::cout << "Time Elapsed: " << (int)Time_Elapsed.count() << " ms" << std::endl;
                      // Writing in the log files
                      gesture_log_file << (int)Time_Elapsed.count() << "; " << labels[std::distance(std::begin(result), largest)] << "; " << *largest << "; " << (int)gesture_time.count() << " ms\n";
                      socket.send("gesture/"+TopLabel+";"+MinimumDistanceStr);
                    }
                  else
                    {
                      TopLabel = "None";
                      socket.send("gesture/"+TopLabel+";"+MinimumDistanceStr);
                      std::cout << "My top label is: None" << std::endl;
                    }
                }
              else
                socket.send("gesture/None;"+MinimumDistanceStr);
              }
        }
      for (int i = 0; i < skeletal_coordinates.size(); i++)
        cv::circle(display_image, cv::Point(skeletal_coordinates[i].x, skeletal_coordinates[i].y), 8, (0,0,255), -1);

      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> fp_ms = t1 - t0;
      double dt = 1 / (1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
      int time_taken = (int)dt;
      std::cout << "Whole Process Takes: " << (int)fp_ms.count() << " ms" << std::endl;
      // std::cout << "Clock Time: " << (int)millis << " ms" << std::endl;
      total = total - readings[readIndex];  //subtract one element from readings array from position readIndex
      readings[readIndex] = time_taken;
      total = total + readings[readIndex];
      // advance to the next position in the array:
      readIndex ++;
      if(readIndex >= numReadings)
        readIndex = 0;
      average_fps = total / numReadings;


      distance_total = distance_total - distance_readings[readIndex];  //subtract one element from readings array from position readIndex
      distance_readings[readIndex] = minimum_distance;
      distance_total = distance_total + distance_readings[readIndex];
      // advance to the next position in the array:
      readIndex ++;
      if(readIndex >= numReadings)
        readIndex = 0;
      average_distance = distance_total / numReadings;


      cv::putText(display_image, "FPS: " + std::to_string(average_fps),
                  fps_display_position, cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  3.0, // Scale. 2.0 = 2x bigger
                  cv::Scalar(0, 0, 255), // Color
                  2, // Thickness
                  CV_AA); // Anti-alias

      cv::putText(display_image, "TopLabel: " + TopLabel,
                  TopLabel_display_position, cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  3.0, // Scale. 2.0 = 2x bigger
                  cv::Scalar(0, 0, 255), // Color
                  2, // Thickness
                  CV_AA); // Anti-alias

      cv::putText(display_image, "Distance: " + std::to_string(minimum_distance),
                  Distance_display_position, cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  3.0, // Scale. 2.0 = 2x bigger
                  cv::Scalar(0, 0, 255), // Color
                  2, // Thickness
                  CV_AA); // Anti-alias
      cv::imshow("Transmitted Image", display_image);


      temp_coordinate_vector.clear();
      int key = cv::waitKey(1);
      protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27));
      listener.release(frames);
      minimum_distance_temp = 5000; // reset this value to a large value so we can detect the minimum with new distance vector values
    }
    gesture_log_file.close();
    dev->stop();
    dev->close();
    delete registration;
    std::cout << "Streaming Ends!" << std::endl;
    return 0;
  }
