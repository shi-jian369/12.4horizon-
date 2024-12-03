#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
using namespace cv;
using namespace std;
int main(){
    VideoCapture cap("/home/shijian/桌面/vscode/build/resources/20241026_225305.mp4");
    if(!cap.isOpened()){
        cout<<"Error opening video stream or file"<<endl;
        return -1;
    }
    QRCodeDetector qrcode;
    while(1){
        Mat frame;
        cap>>frame;
        if(frame.empty()){
            break;
        }
        vector<Point> points;
        string data = qrcode.detectAndDecode(frame, points);
        if(!data.empty()){
            cout<<"QR code data: " << data << endl;
            if(points.size() == 4){
                for(int i = 0; i < 4; i++){
                    line(frame, points[i], points[(i+1)%4], Scalar(0, 0, 255), 2);
                }
            }
            putText(frame, data, Point(points[0].x, points[0].y-10), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 2);

    }
    imshow("QR code detection", frame);
    if(waitKey(10) == 27){
        break;
    }
    }
    return 0;
}
