#include <opencv2/opencv.hpp>
#include <zbar.h>
#include <thread>
#include "yolo.hpp"
#include "argparse.hpp"
cv::dnn::Net net;
int main(int argc, const char **argv)
{
    argparse::ArgumentParser program("detect");
    program.add_argument("-i", "--index")
           .help("The camera index")
           .scan<'i', int>()
           .default_value(0);
    
    program.add_argument("-t", "--frame_time")
           .help("The target duration of a frame in ms")
           .scan<'u', unsigned>()
           .default_value(100);
    
    program.add_argument("--GPU")
        .help("Whether to use the GPU")
        .implicit_value(true)
        .default_value(false);
    
    try{
        program.parse_args(argc, argv);
    }
    catch (std::runtime_error& e){
        std::cerr << e.what() << std::endl;
        std::cerr << program;
        return -1;
    }

    int frame_time = program.get<int>("t");
    yolo::load_net(net, program.get<bool>("GPU") ? cv::dnn::DNN_TARGET_OPENCL_FP16 : cv::dnn::DNN_TARGET_CPU);
    int index = program.get<int>("i");
    cv::VideoCapture cap(index);

    if (!cap.isOpened())
    {
        std::cerr << "Can't access primary camera (" << index << ')' << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    zbar::ImageScanner scanner;
    scanner.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
    cv::Mat frame;
    std::vector<yolo::Detection> output;
    int64_t inference_time = 0;
    int64_t frame_count = 0;
    int64_t cumulative_time = 0;
    // Rate rate(83); // aim for 12 fps
    yolo::Rate rate(frame_time); // aim for 10 fps
    while (true)
    {
        using namespace std::chrono;
        cap >> frame;

        auto t0 = high_resolution_clock::now();
        cv::Mat grey;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        zbar::Image image(grey.cols, grey.rows, "Y800", (uchar *)grey.data, grey.cols * grey.rows);

        auto detect_yolo = [&]()
        {
            output = yolo::detect(frame, net);
        };
        auto detect_qr = [&]()
        {
            scanner.scan(image);
        };

        std::thread yolo_thread(detect_yolo);
        std::thread qr_thread(detect_qr);

        qr_thread.join();
        yolo_thread.join();
        auto t1 = high_resolution_clock::now();
        yolo::draw_qrs(image, frame);
        yolo::draw_boxes(output, frame);

        cumulative_time += duration_cast<milliseconds>(t1 - t0).count();
        frame_count++;
        if (cumulative_time > 250 && frame_count > 10)
        {
            inference_time = cumulative_time / frame_count; // 1000 * 1000 * 1000 * frame_count / cumulative_time;
            frame_count = 0;
            cumulative_time = 0;
            // std::cout << fps << std::endl;
        }
        cv::putText(frame, std::to_string(inference_time) + "ms inference", {25, 460}, cv::FONT_HERSHEY_DUPLEX, 1.0, {0, 0, 0}, 2);
        cv::imshow("Detections", frame);
        char c = (char)cv::waitKey(1);
        if (c == 'q')
            break;
        rate.pause();
    }
}