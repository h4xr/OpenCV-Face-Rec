#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char seperator=';') {
    std::ifstream ifile(filename.c_str(), ifstream::in);
    if(!ifile) {
        string error_msg = "Input file invalid! Unable to open file";
        CV_Error(CV_StsBadArg, error_msg);
    }
    string line, path, classlabel;
    while(getline(ifile, line)) {
        stringstream liness(line);
        getline(liness, path, seperator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[]) {
    //Check commandline arguments and print menu if invalid
    if(argc!=4) {
        cout<<"Usage: "<< argv[0] << "<path_to_haarcascade> <path_to_csv> <device_id>" << endl;
        exit(1);
    }
    //Get path to csv
    string fn_haar = argv[1];
    string fn_csv = argv[2];
    int device_id = atoi(argv[3]);

    //Hold the images and their corresponding labels
    vector<Mat> images;
    vector<int> labels;

    //Get started reading the data
    try {
        read_csv(fn_csv, images, labels);
    } catch(cv::Exception& e) {
        cerr<< "Error loading image " << fn_csv << ": " << e.msg << "\n";
        exit(1); //Can't continue without data
    }

    //Get image width and height for later processing
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    //Let's create a face recognizer and train that
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);

    //Let's build the classifier
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    //Get video device
    VideoCapture cap(device_id);

    //Check if the device can be used or not
    if(!cap.isOpened()) {
        cerr<< "Unable to read from device "<< device_id <<'\n';
        return -1; //Nothing to do now, bye!
    }
    //Hold the current frame from video device
    Mat frame;
    for(;;) {
        cap >> frame;
        //Clone the frame
        Mat original = frame.clone();
        //Convert frame to grayscale
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        //Let's find some faces in the frame
        vector < Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        //Make a prediction and annotate it in video
        for(int i=0;i<faces.size();i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);

            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            //Now let's predict
            int prediction = model->predict(face_resized);

            //Show our prediction on the original image
            rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
            string box_text = format("Prediction = %d", prediction);

            //Calculate where to put our text
            int pos_x = std::max(face_i.tl().x-10 , 0);
            int pos_y = std::max(face_i.tl().y-10, 0);

            //Put the text into image
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
        }
        //Show the image
        imshow("Face recognizer", original);
        //Display it
        char key = (char) waitKey(20);
        if(key == 27)
            break;
    }
    return 0;
}
