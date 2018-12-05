#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <time.h>

#define PI 3.14159265

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  if (argc != 9) {
    cerr << "Usage: " << argv[0] << " deploy.prototxt network.caffemodel gpu input thr dks eks output" << endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string net_file = argv[1];
  string weight_file = argv[2];

  Caffe::SetDevice(atoi(argv[3]));
  Caffe::set_mode(Caffe::GPU);

  shared_ptr<Net<float> > net_;
  net_.reset(new Net<float>(net_file, TEST));
  net_->CopyTrainedLayersFrom(weight_file);

  string input_dir = argv[4];
  string input_file = input_dir+"*.jpg";
  vector<String> fn;
  glob(input_file, fn, false);
  int fn_count = fn.size();
  clock_t start_time = clock();
  for (int idx = 0; idx < fn_count; idx++) {
    cout << fn[idx] << endl;
//    cout << fn[idx].substr(fn[i].rfind("/")+1) << endl;
    Mat img = imread(fn[idx], 1);
    int height = img.size().height;
    int width = img.size().width;
    int channel = img.channels();
  
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, channel, height, width);
    net_->Reshape();
//    cout << input_layer->height() << input_layer->width() << input_layer->channels() << endl;
  
    float* input_data = input_layer->mutable_cpu_data();
    vector<Mat> input_channels;
    for (int i = 0; i < input_layer->channels(); ++i) {
      Mat channel(height, width, CV_32FC1, input_data);
      input_channels.push_back(channel);
      input_data += height * width;
    }
  
    Mat mean_ = Mat(height, width, CV_32FC3);
    mean_ = Scalar(103.939, 116.779, 123.68);
    img.convertTo(img, CV_32FC3);
    subtract(img, mean_, img);
    split(img, input_channels);
  
    net_->Forward();
  
    Blob<float>* output_layer = net_->output_blobs()[0];
    float* output_data = output_layer->mutable_cpu_data();
//    cout << output_layer->height() << output_layer->width() << output_layer->channels() << endl;
    Mat predx(height, width, CV_32FC1, output_data);
    output_data += height * width;
    Mat predy(height, width, CV_32FC1, output_data);
  
    float thr = atof(argv[5]);
    int dks = atoi(argv[6]);
    int eks = atoi(argv[7]);
    Mat magnitude(height, width, CV_32FC1);
    Mat angle(height, width, CV_32FC1);
    Mat mask(height, width, CV_32FC1);
    Mat ending(height, width, CV_32FC1, Scalar(0));
    Mat prob(height, width, CV_32FC1, Scalar(0));
    cartToPolar(predx, predy, magnitude, angle);
  
    double v_min, v_max;
  //  double a_min, a_max;
    minMaxIdx(magnitude, &v_min, &v_max);
  //  minMaxIdx(angle, &a_min, &a_max);
  //  cout << v_min << endl;
  //  cout << v_max << endl;
  //  cout << a_min << endl;
  //  cout << a_max << endl;
    compare(magnitude, Scalar(thr), mask, CMP_GT);
    mask.convertTo(mask, CV_32FC1);
  
    for (int i = 0; i < height; i++) {
      float* mask_p = mask.ptr<float>(i);
      float* angle_p = angle.ptr<float>(i);
      float* ending_p = ending.ptr<float>(i);
      for (int j = 0; j < width; j++) {
        if (mask_p[j] == 255) {
          if (angle_p[j] >= PI/8 && angle_p[j] < 3*PI/8 && i+1 <= height-1 && j+1 <= width-1) {
            float* parent_p = mask.ptr<float>(i+1);
            if (parent_p[j+1] == 0) {
              ending_p[j] = 1;
            }
          }
          else if (angle_p[j] >= 3*PI/8 && angle_p[j] < 5*PI/8 && j+1 <= width-1) {
            float* parent_p = mask.ptr<float>(i);
            if (parent_p[j+1] == 0) {
              ending_p[j] = 1;
            }
          }
          else if (angle_p[j] >= 5*PI/8 && angle_p[j] < 7*PI/8 && i-1 >= 0 && j+1 <= width-1) {
            float* parent_p = mask.ptr<float>(i-1);
            if (parent_p[j+1] == 0) {
              ending_p[j] = 1;
            }
          }
          else if (angle_p[j] >= 7*PI/8 && angle_p[j] < 9*PI/8 && i-1 >= 0) {
            float* parent_p = mask.ptr<float>(i-1);
            if (parent_p[j] == 0) {
              ending_p[j] = 1;
            }
          }
          else if (angle_p[j] >= 9*PI/8 && angle_p[j] < 11*PI/8 && i-1 >= 0 && j-1 >= 0) {
            float* parent_p = mask.ptr<float>(i-1);
            if (parent_p[j-1] == 0) {
              ending_p[j] = 1;
            }
          }
          else if (angle_p[j] >= 11*PI/8 && angle_p[j] < 13*PI/8 && j-1 >= 0) {
            float* parent_p = mask.ptr<float>(i);
            if (parent_p[j-1] == 0) {
              ending_p[j] = 1;
            }
          }
          else if (angle_p[j] >= 13*PI/8 && angle_p[j] < 15*PI/8 && i+1 <= height-1 && j-1 >= 0) {
            float* parent_p = mask.ptr<float>(i+1);
            if (parent_p[j-1] == 0) {
              ending_p[j] = 1;
            }
          }
          else if ((angle_p[j] < PI/8 || angle_p[j] >= 15*PI/8) && i+1 <= height-1 && j+1 <= width-1) {
            float* parent_p = mask.ptr<float>(i+1);
            if (parent_p[j] == 0) {
              ending_p[j] = 1;
            }
          }
        }
      }
    }
  
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(dks, dks));
    dilate(ending, ending, element);
    element = getStructuringElement(MORPH_ELLIPSE, Size(eks, eks));
    erode(ending, ending, element);
  
  //  double max, min;
  //  minMaxIdx(ending, &min, &max);
  //  cout << min << endl;
  //  cout << max << endl;
  
    for (int i = 0; i < height; i++) {
      float* prob_p = prob.ptr<float>(i);
      float* magnitude_p = magnitude.ptr<float>(i);
      float* ending_p = ending.ptr<float>(i);
        for (int j = 0; j < width; j++) {
            prob_p[j] = (v_max - magnitude_p[j])/v_max;
            if (ending_p[j] != 1){
                prob_p[j] = 0;
            }
        }
    }
  
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(2);
    string output_dir = argv[8];
    string output_fn = output_dir+fn[idx].substr(0,fn[idx].length()-4).substr(fn[idx].rfind("/")+1)+".png";
//    cout << output_fn << endl;
    imwrite(output_fn, 255*prob, compression_params);
  }
  clock_t end_time = clock();
  cout<<"-----Runtime: "<<static_cast<float>(end_time - start_time)/CLOCKS_PER_SEC/fn_count<<" s/image-----"<<endl;
}

#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
