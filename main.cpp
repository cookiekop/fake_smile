
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

using namespace dlib;

#define SMILE_EXTENT 10

std::vector<cv::Point2f> keyPoints;

void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = getAffineTransform( srcTri, dstTri );

    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}

void warpTriangle(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2)
{

    cv::Rect r1 = boundingRect(t1);
    cv::Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    std::vector<cv::Point2f> t1Rect, t2Rect;
    std::vector<cv::Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {

        t1Rect.push_back( cv::Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( cv::Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        t2RectInt.push_back( cv::Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly

    }

    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warpImage to small rectangular patches
    cv::Mat img1Rect;
    img1(r1).copyTo(img1Rect);

    cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());

    applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

    multiply(img2Rect,mask, img2Rect);
    multiply(img2(r2), cv::Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Rect;


}

void calculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &delaunayTri){

    // Create an instance of Subdiv2D
    cv::Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for( std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);

    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    std::vector<cv::Point2f> pt(3);
    std::vector<int> ind(3);

    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        cv::Vec6f t = triangleList[i];
        pt[0] = cv::Point2f(t[0], t[1]);
        pt[1] = cv::Point2f(t[2], t[3]);
        pt[2] = cv::Point2f(t[4], t[5 ]);

        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
            for(int j = 0; j < 3; j++)
                for(size_t k = 0; k < points.size(); k++)
                    if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
                        ind[j] = k;

            delaunayTri.push_back(ind);
        }
    }

}

void getKeyPoints(const cv::Mat &src) {
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("./model/shape_predictor_68_face_landmarks.dat") >> pose_model;

    cv_image<bgr_pixel> cimg(src);

    std::vector<rectangle> faces = detector(cimg);

    std::vector<full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));


    cv::Mat tmp = src.clone();
    if (!shapes.empty()) {
        keyPoints.clear();
        for (int i = 0; i < 68; i++) {
            cv::Point2f t(shapes[0].part(i).x(), shapes[0].part(i).y());
            keyPoints.push_back(t);
            circle(tmp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
            //	shapes[0].part(i).x();//68个
        }
    }

    cv::imshow("key points", tmp);

}

int main() {
    cv::Mat img = cv::imread("./input.jpg");

    getKeyPoints(img);

    GaussianBlur( img, img, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    if (keyPoints.size() == 0) {
        std::cout << "no key point" << std::endl;
        return -1;
    }

    img.convertTo(img, CV_32F);

    std::vector<cv::Point2f> keyPoints2;

    for(int i=0; i<68; ++i) {
        cv::Point2f t;
        if (i == 48) {
            t = cv::Point2f(keyPoints[i].x - 2*SMILE_EXTENT, keyPoints[i].y - SMILE_EXTENT);
            keyPoints2.push_back(t);
        }
        else if (i == 54){
            t = cv::Point2f(keyPoints[i].x + 2*SMILE_EXTENT, keyPoints[i].y - SMILE_EXTENT);
            keyPoints2.push_back(t);
        }
        else keyPoints2.push_back(keyPoints[i]);
    }

    /*std::vector<int> hullIndex;
    convexHull(keyPoints, hullIndex, false, false);

    std::vector<cv::Point2f> hullPoint, hullPoint2;

    for(int i = 0; i < hullIndex.size(); i++)
    {
        hullPoint.push_back(keyPoints[hullIndex[i]]);
        hullPoint2.push_back(keyPoints2[hullIndex[i]]);
    }*/

    std::vector< std::vector<int> > dt;
    cv::Rect rect(0, 0, img.cols, img.rows);
    calculateDelaunayTriangles(rect, keyPoints, dt);

    cv::Mat out = img.clone();
    for(size_t i = 0; i < dt.size(); i++)
    {
        std::vector<cv::Point2f> t1, t2;
        // Get points for img1, img2 corresponding to the triangles
        for(size_t j = 0; j < 3; j++)
        {
            t1.push_back(keyPoints[dt[i][j]]);
            t2.push_back(keyPoints2[dt[i][j]]);
        }

        warpTriangle(img, out, t1, t2);
    }

    cv::imshow("output", out/255.0);
    cv::waitKey(0);
    return 0;
}