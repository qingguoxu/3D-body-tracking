#ifndef _CONNECT_FUNCTIONS
#define _CONNECT_FUNCTIONS

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <utility> //std::pair

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog/logging.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdio.h>  // snprintf

// #include <unistd.h>  // in windows use io.h
#include <io.h>

#include <stdlib.h>


#include <time.h>
//#include <windows.h> 
#include <winsock2.h> //#include <netinet/in.h>
//#include <sys/socket.h>
//#include <sys/time.h>
#include <sys/types.h>

#include <boost/thread/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
// #include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/cpm/frame.h"
#include "caffe/cpm/layers/imresize_layer.hpp"
#include "caffe/cpm/layers/nms_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/blocking_queue.hpp"
// #include "caffe/util/render_functions.hpp"
// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/db.hpp"
// #include "caffe/util/io.hpp"
// #include "caffe/util/benchmark.hpp"

#include "rtpose/modelDescriptor.h"
#include "rtpose/modelDescriptorFactory.h"
#include "rtpose/renderFunctions.h"


#endif