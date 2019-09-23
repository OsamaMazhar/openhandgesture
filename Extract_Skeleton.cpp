/* This file is part of openhandgesture program.
* Program description : This program acquires Kinect V2 frames from
* Acquire_Send_Detect program through nanomsg, run openpose through
* the received RGB frame, get the depth values corresponding to the
* obtained 2D skeleton and pass the pseudo-3D skeleton back to
* Acquire_Send_Detect program.
*
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

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string>     // std::string, std::to_string
#include <vector>
#include <chrono>
#include <algorithm>

#include <nnxx/message.h>
#include <nnxx/pubsub.h>
#include <nnxx/socket.h>
#include <nnxx/pair.h>
#include <nnxx/reqrep.h>
#include <nnxx/testing>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <signal.h>

// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
//#include <glog/logging.h> // google::InitGoogleLogging //this clashes with tensorflow logging

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif

// #include "webcam_generated.h"
#include "./essentials/one_skeleton_generated.h"
#include "./essentials/gesture-recognition-request_generated.h"

constexpr size_t TX_MAX_PACKET_SIZE = 100000;     // 1 Kb in decimal
constexpr size_t RX_MAX_PACKET_SIZE = 30000000; // 10 Mb

// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "media/COCO_val2014_000000000328.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              19,             "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF");
DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                                        " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                                        " `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
                                                        // Reads a model graph definition from disk, and creates a session object we
                                                        // can use to run it.

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

libfreenect2::Freenect2Device::IrCameraParams Mat_to_Freenect2IrParams(cv::Mat IrParams)
  {
    libfreenect2::Freenect2Device::IrCameraParams LocalIrCP;
    LocalIrCP.fx = IrParams.at<double>(0);
    LocalIrCP.fy = IrParams.at<double>(1);
    LocalIrCP.cx = IrParams.at<double>(2);
    LocalIrCP.cy = IrParams.at<double>(3);
    LocalIrCP.k1 = IrParams.at<double>(4);
    LocalIrCP.k2 = IrParams.at<double>(5);
    LocalIrCP.k3 = IrParams.at<double>(6);
    LocalIrCP.p1 = IrParams.at<double>(7);
    LocalIrCP.p2 = IrParams.at<double>(8);

    return LocalIrCP;
  }

libfreenect2::Freenect2Device::ColorCameraParams Mat_to_Freenect2ColorParams(cv::Mat ColorParams)
  {
    libfreenect2::Freenect2Device::ColorCameraParams LocalCCP;
    LocalCCP.fx = ColorParams.at<double>(0);
    LocalCCP.fy = ColorParams.at<double>(1);
    LocalCCP.cx = ColorParams.at<double>(2);
    LocalCCP.cy = ColorParams.at<double>(3);
    LocalCCP.shift_d = ColorParams.at<double>(4);
    LocalCCP.shift_m = ColorParams.at<double>(5);
    LocalCCP.mx_x3y0 = ColorParams.at<double>(6);
    LocalCCP.mx_x0y3 = ColorParams.at<double>(7);
    LocalCCP.mx_x2y1 = ColorParams.at<double>(8);
    LocalCCP.mx_x1y2 = ColorParams.at<double>(9);
    LocalCCP.mx_x2y0 = ColorParams.at<double>(10);
    LocalCCP.mx_x0y2 = ColorParams.at<double>(11);
    LocalCCP.mx_x1y1 = ColorParams.at<double>(12);
    LocalCCP.mx_x1y0 = ColorParams.at<double>(13);
    LocalCCP.mx_x0y1 = ColorParams.at<double>(14);
    LocalCCP.mx_x0y0 = ColorParams.at<double>(15);
    LocalCCP.my_x3y0 = ColorParams.at<double>(16);
    LocalCCP.my_x0y3 = ColorParams.at<double>(17);
    LocalCCP.my_x2y1 = ColorParams.at<double>(18);
    LocalCCP.my_x1y2 = ColorParams.at<double>(19);
    LocalCCP.my_x2y0 = ColorParams.at<double>(20);
    LocalCCP.my_x0y2 = ColorParams.at<double>(21);
    LocalCCP.my_x1y1 = ColorParams.at<double>(22);
    LocalCCP.my_x1y0 = ColorParams.at<double>(23);
    LocalCCP.my_x0y1 = ColorParams.at<double>(24);
    LocalCCP.my_x0y0 = ColorParams.at<double>(25);

    return LocalCCP;
  }

  cv::Mat Freenect2IrParams_to_cv_Mat(libfreenect2::Freenect2Device::IrCameraParams LocalIrCP)
    {
      cv::Mat IrCameraParams_Mat = (cv::Mat_<double>(3,3) <<  LocalIrCP.fx,
                                                              LocalIrCP.fy,
                                                              LocalIrCP.cx,
                                                              LocalIrCP.cy,
                                                              LocalIrCP.k1,
                                                              LocalIrCP.k2,
                                                              LocalIrCP.k3,
                                                              LocalIrCP.p1,
                                                              LocalIrCP.p2);
      return IrCameraParams_Mat;
    }
  cv::Mat Freenect2ColorParams_to_cv_Mat(libfreenect2::Freenect2Device::ColorCameraParams LocalCCP)
    {
      cv::Mat Kinect2ColorParams_Mat = (cv::Mat_<double>(6,5) <<  LocalCCP.fx,
                                                                  LocalCCP.fy,
                                                                  LocalCCP.cx,
                                                                  LocalCCP.cy,
                                                                  LocalCCP.shift_d,
                                                                  LocalCCP.shift_m,
                                                                  LocalCCP.mx_x3y0,
                                                                  LocalCCP.mx_x0y3,
                                                                  LocalCCP.mx_x2y1,
                                                                  LocalCCP.mx_x1y2,
                                                                  LocalCCP.mx_x2y0,
                                                                  LocalCCP.mx_x0y2,
                                                                  LocalCCP.mx_x1y1,
                                                                  LocalCCP.mx_x1y0,
                                                                  LocalCCP.mx_x0y1,
                                                                  LocalCCP.mx_x0y0,
                                                                  LocalCCP.my_x3y0,
                                                                  LocalCCP.my_x0y3,
                                                                  LocalCCP.my_x2y1,
                                                                  LocalCCP.my_x1y2,
                                                                  LocalCCP.my_x2y0,
                                                                  LocalCCP.my_x0y2,
                                                                  LocalCCP.my_x1y1,
                                                                  LocalCCP.my_x1y0,
                                                                  LocalCCP.my_x0y1,
                                                                  LocalCCP.my_x0y0,
                                                                  0,
                                                                  0,
                                                                  0,
                                                                  0);
      return Kinect2ColorParams_Mat;
    }

int main(int argc, char *argv[])
{
  // Initializing google logging (Caffe uses it for logging)
  // google::InitGoogleLogging("openPoseKinectCustom");

  nnxx::socket Coordinate_req_socket { nnxx::SP, nnxx::REQ };
	nnxx::socket Frame_req_socket { nnxx::SP, nnxx::REQ };
  nnxx::socket Frame_data_socket { nnxx::SP, nnxx::REP };

	const char *Coordinate_req_addr = "ipc://./essentials/Coordinate_req.ipc";
	const char *Frame_req_addr = "ipc://./essentials/kinect.ipc";
  const char *Frame_data_addr = "ipc://./essentials/frame_data.ipc";

	auto Coordinate_rep_socket_id = Coordinate_req_socket.connect(Coordinate_req_addr);
	auto Frame_req_socket_id = Frame_req_socket.connect(Frame_req_addr);
	auto Frame_data_socket_id = Frame_data_socket.bind(Frame_data_addr);
	Frame_data_socket.setopt(NN_SOL_SOCKET, NN_RCVMAXSIZE, -1);

	// std::cout << "Coordinates will be sent at: (" << Coordinate_rep_socket_id << ")\n";
	// std::cout << "Frame request will be sent at: (" << Frame_req_socket_id << ")\n";
	// std::cout << "Frame data will be received at: (" << Frame_data_socket_id << ")\n";

  // >>> Loading Kinect2 Camera Calibration Files:
  cv::FileStorage fs("./essentials/Kinect2_CameraParams.yml", cv::FileStorage::READ);
  cv::Mat LocalIrCP, LocalCCP;
  fs["Kinect2IrCameraParams"] >> LocalIrCP;
  fs["Kinect2ColorCameraParams"] >> LocalCCP;
  fs.release();

  libfreenect2::Freenect2Device::IrCameraParams LocalIrCParams;
  libfreenect2::Freenect2Device::ColorCameraParams LocalCCParams;

  LocalIrCParams = Mat_to_Freenect2IrParams(LocalIrCP);
  LocalCCParams = Mat_to_Freenect2ColorParams(LocalCCP);

  // cv::Mat IrTest = Freenect2IrParams_to_cv_Mat(LocalIrCParams);
  // cv::Mat ColorTest = Freenect2ColorParams_to_cv_Mat(LocalCCParams);
  //
  // std::cout <<"Camera Ir Parameters Test: " << IrTest << std::endl;
  // std::cout <<"Camera Color Parameters Test: " << ColorTest << std::endl;

  // <<< Loading Kinect2 Camera Calibration Files:
  libfreenect2::Registration* registration = new libfreenect2::Registration(LocalIrCParams, LocalCCParams);
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4);

	// >>> Openpose Initialization

	// ------------------------- INITIALIZATION -------------------------
	// Step 1 - Set logging level
	// - 0 will output all the logging messages
	// - 255 will output nothing

  // Kinect libfreenect2 Initialization ends here

  // ----------------  OPEN POSE INITIALIZATION ------------------

  // Step 1 - Set logging level
      // - 0 will output all the logging messages
      // - 255 will output nothing
  op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
  op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
  op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
  // Step 2 - Read Google flags (user defined configuration)
  // outputSize
  const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
  // netInputSize
  const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
  // netOutputSize
  const auto netOutputSize = netInputSize;
  // poseModel
  const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
  // Check no contradictory flags enabled
  if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
      op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
  if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
      op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.", __LINE__, __FUNCTION__, __FILE__);
  // Logging
  op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
  // Step 3 - Initialize all required classes
  op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
  op::CvMatToOpInput cvMatToOpInput{poseModel};
  op::CvMatToOpOutput cvMatToOpOutput;
  op::PoseExtractorCaffe poseExtractorCaffe{poseModel, FLAGS_model_folder, FLAGS_num_gpu_start};
  op::PoseCpuRenderer poseRenderer{poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                                   (float)FLAGS_alpha_pose};
  op::OpOutputToCvMat opOutputToCvMat;
  op::FrameDisplayer frameDisplayer{"OpenPose Kinect - Osama", outputSize};
  // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
  poseExtractorCaffe.initializationOnThread();
  poseRenderer.initializationOnThread();

  std::string message = "Frame_req";
  int count = 0;
  int bytes_sent;

  auto buffer = new unsigned char[RX_MAX_PACKET_SIZE];

  // cv::Mat received_image;
  cv::namedWindow("Received RGB", cv::WINDOW_NORMAL);

  cv::Mat inputImage;
  cv::Point current_point_2d;
  int mean_area_length = 6;
  float depth_val;

  cv::Point fps_display_position;
  fps_display_position.x = 1400;
  fps_display_position.y = 100;

  // >>> For averaging FPS on display
  const int numReadings = 10;
  int readings[numReadings];      // the readings from the analog input
  int readIndex = 0;
  int total = 0;                  // the running total
  int average_fps = 0;            // the average

  for (int thisReading = 0; thisReading < numReadings; thisReading++) {
      readings[thisReading] = 0;
    }
  // <<< For averaging FPS on display
  cv::Mat depth_reg;
  std::vector<float> joint_scores;

	op_skeleton::Point points[13]; // this number -> number of joints selected
  int totalbodyParts;
  for(;;)
    {

      // >>> FlatBuffer Initialization
		  flatbuffers::FlatBufferBuilder result_builder(RX_MAX_PACKET_SIZE);
			flatbuffers::FlatBufferBuilder skeleton_builder(TX_MAX_PACKET_SIZE);


      bytes_sent = Frame_req_socket.send(message);
      nnxx_check(bytes_sent == message.length());
      std::cout << "........................." << std::endl;
      std::cout << "...Frame Request Sent...." << std::endl;
      std::cout << "........................." << std::endl;
      std::cout << "..Waiting for the Frame.." << std::endl;
      std::cout << "........................." << std::endl;
      auto bytes_received = Frame_data_socket.recv(buffer, RX_MAX_PACKET_SIZE);
			// std::cout << bytes_received << " bytes received!" << std::endl;
			std::cout << "...New Frame Received...." << std::endl;
      std::cout << "........................." << std::endl;
      auto t0 = std::chrono::high_resolution_clock::now();
      auto request = gesture_recognition::Getrequest_identification(buffer);

			auto fill_frame = [](const ::gesture_recognition::kinect2_frame* req, libfreenect2::Frame* frame)
				{
					frame->width = req->width();
					frame->height = req->height();
					frame->bytes_per_pixel = req->bytes_per_pixel();
					frame->data = const_cast<unsigned char*>(req->data()->Data());
					frame->timestamp = req->timestamp();
					frame->sequence = req->sequence();
					frame->exposure = req->exposure();
					frame->gain = req->gain();
					frame->gamma = req->gamma();
					frame->status = req->status();
					frame->format = static_cast<libfreenect2::Frame::Format>(req->format());
				};

    	libfreenect2::Frame color_frame(request->color()->width(), request->color()->height(), request->color()->bytes_per_pixel());
			libfreenect2::Frame depth_frame(request->depth()->width(), request->depth()->height(), request->depth()->bytes_per_pixel());

			fill_frame(request->color(), &color_frame);
			fill_frame(request->depth(), &depth_frame);

      //! [registration]
      registration->apply(&color_frame, &depth_frame, &undistorted, &registered, true, &depth2rgb);
      //! [registration]

      auto rgb = cv::Mat(request->color()->height(), request->color()->width(), CV_8UC4, const_cast<unsigned char*>(request->color()->data()->Data()));
			auto depth = cv::Mat(request->depth()->height(), request->depth()->width(), CV_32FC1, const_cast<unsigned char*>(request->depth()->data()->Data()));
      cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, (depth2rgb.data)).copyTo(depth_reg);

      cv::cvtColor(rgb, inputImage, cv::COLOR_RGB2BGR); //openpose requires BGR image format
      cv::Mat source_image_original = rgb.clone();
      if(inputImage.empty())
          op::error("Empty Image matrix received and Image not found at the Image path! " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
      const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
      // Step 2 - Format input image to OpenPose input and output formats
      std::vector<double> scaleInputToNetInputs;
      std::vector<op::Point<int>> netInputSizes;
      double scaleInputToOutput;
      op::Point<int> outputResolution;
      std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
          = scaleAndSizeExtractor.extract(imageSize);
      // Step 3 - Format input image to OpenPose input and output formats
      const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
      auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
      // Step 4 - Estimate poseKeypoints
      poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
      const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();

      /* NEW Lines added 7th Nov */
      cv::Point joints[poseKeypoints.getSize(1)];
      int number_of_persons = poseKeypoints.getSize(0);

      // for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
      // {
          //op::log("Person " + std::to_string(person) + " (x, y, score):");
      // >>> If person is found, extract skeleton points and send it to main streamer
      if(number_of_persons != 0)
        {
          {
            totalbodyParts = poseKeypoints.getSize(1);
            for (auto bodyPart = 0 ; bodyPart < totalbodyParts ; bodyPart++)
              {
                for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
                  {
                    if (xyscore == 0)
                      joints[bodyPart].x = poseKeypoints[{0, bodyPart, xyscore}];
                    if (xyscore == 1)
                      joints[bodyPart].y = poseKeypoints[{0, bodyPart, xyscore}];
                    if (xyscore == 2){
                      joint_scores.push_back(poseKeypoints[{0, bodyPart, xyscore}]);
                    }
                  }
                }
            }
      // }

        // POSE_COCO_BODY_PARTS {
        //     {0,  "Nose"},
        //     {1,  "Neck"},
        //     {2,  "RShoulder"},
        //     {3,  "RElbow"},
        //     {4,  "RWrist"},
        //     {5,  "LShoulder"},
        //     {6,  "LElbow"},
        //     {7,  "LWrist"},
        //     {8,  "RHip"},
        //     {9,  "RKnee"},
        //     {10, "RAnkle"},
        //     {11, "LHip"},
        //     {12, "LKnee"},
        //     {13, "LAnkle"},
        //     {14, "REye"},
        //     {15, "LEye"},
        //     {16, "REar"},
        //     {17, "LEar"},
        //     {18, "Bkg"},
        // }
        auto skeleton_name = skeleton_builder.CreateString("Person1");
        // for(int i = 0; i < joint_scores.size(); i++)
        //   std::cout << "Joint " << i << " score is: " << joint_scores[i] << std::endl;
        for( int joint_point_loop = 0; joint_point_loop < totalbodyParts - 5; joint_point_loop++) // -5 as we don't want last five points to show
          {
            current_point_2d = joints[joint_point_loop];
            if(current_point_2d.x != 0 && current_point_2d.y != 0)
              getmeanvalue(current_point_2d, mean_area_length, &depth_val, depth_reg / 4096.0f);
            else
              depth_val = 0;
            if((int)depth_val < 0)
              depth_val = 0;
            if(joint_scores[joint_point_loop] < 0.6)
              depth_val = 0;
            points[joint_point_loop] = op_skeleton::Point(current_point_2d.x, current_point_2d.y, (int)depth_val);
          }
    	    auto point_vector = skeleton_builder.CreateVectorOfStructs(points, totalbodyParts-5);
			    auto num_of_points = totalbodyParts-5;
			    auto unique_skeleton = Createone_skeleton(skeleton_builder, skeleton_name, point_vector, num_of_points);
          skeleton_builder.Finish(unique_skeleton);
			    bytes_sent = Coordinate_req_socket.send((const void *)skeleton_builder.GetBufferPointer(), skeleton_builder.GetSize());
			    // std::cout << "Coordinates bytes sent: " << bytes_sent << std::endl;
        } // <<< If person is found, otherwise the above part will be ignored!

      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> fp_ms = t1 - t0;
      double dt = 1 / (1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
      int time_taken = (int)dt;
      std::cout << "Frame, Skeleton Extraction and Coordinate Sending took: " << (int)fp_ms.count() << " ms" << std::endl;
      total = total - readings[readIndex];  //subtract one element from readings array from position readIndex
      readings[readIndex] = time_taken;
      total = total + readings[readIndex];
      // advance to the next position in the array:
      readIndex ++;
      if(readIndex >= numReadings)
        readIndex = 0;
      average_fps = total / numReadings;

      cv::putText(rgb, "FPS: " + std::to_string(average_fps),
                  fps_display_position, cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  3.0, // Scale. 2.0 = 2x bigger
                  cv::Scalar(0, 0, 255), // Color
                  2, // Thickness
                  CV_AA); // Anti-alias


      cv::imshow("Received RGB", rgb);

      int key = cv::waitKey(1);
      if(key > 0 && ((key & 0xFF) == 27)) // Escape Key to exit
        break;
      joint_scores.clear();
    }
}
