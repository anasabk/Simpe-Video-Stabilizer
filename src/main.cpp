#include <iostream>

#include "stabcv.hpp"

int main(int argc, char* argv[]) {
    // cv::Mat im1 = cv::imread(argv[1], cv::IMREAD_REDUCED_GRAYSCALE_2);
    // cv::Mat im2 = cv::imread(argv[2], cv::IMREAD_REDUCED_GRAYSCALE_2);
    // cv::Mat fixed;

    // stabcv::Stabilizer stabalizer;
    // cv::Mat affine = stabalizer.compute_affine(im1, im2, true);
    // printf("Orientation: %f degrees\n", -atan2(affine.at<double>(1, 0), affine.at<double>(0, 0))*180/M_PI);
    // cv::warpAffine(im2, fixed, affine, im1.size());
    // imshow("original", im1);
    // imshow("training", im2);
    // imshow("transfromed", fixed);
    // imwrite("original.jpg", im1);
    // imwrite("training.jpg", im2);
    // imwrite("transfromed.jpg", fixed);
    // cv::waitKey(0);

    stabcv::Stabilizer stabilizer;
    // stabilizer.compute_vid_avg(argv[1], argv[2], true, atoi(argv[3]), strcmp(argv[4], "fixed") == 0);    // fixed test
    // stabilizer.compute_vid_avg(argv[1], argv[2], true, 3, false);
    stabilizer.compute_vid_staged(argv[1], argv[2], false, atoi(argv[3]), strcmp(argv[4], "fixed") == 0); //Staged fix
    // stabilizer.compute_vid_staged(argv[1], argv[2], false, atoi(argv[3]));
}