// #define CUDA_STAB 1

#ifdef CUDA_STAB
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudafeatures2d.hpp>
#else
#include <opencv4/opencv2/features2d.hpp>
#endif

namespace stabcv
{
struct Trajectory
{
    Trajectory() {
        this->x = 0;
        this->y = 0;
        this->a = 0;
    }

    Trajectory(double x, double y, double a) {
        this->x = x;
        this->y = y;
        this->a = a;
    }

    Trajectory operator+(Trajectory &t) {
        return Trajectory(this->x + t.x, this->y + t.y, this->a + t.a);
    }

    void operator+=(Trajectory &t) {
        this->x += t.x;
        this->y += t.y;
        this->a += t.a;
    }

    void operator+=(Trajectory &&t) {
        this->x += t.x;
        this->y += t.y;
        this->a += t.a;
    }

    Trajectory operator-(Trajectory &t) {
        return Trajectory(this->x - t.x, this->y - t.y, this->a - t.a);
    }

    Trajectory operator/(int div) {
        return Trajectory(this->x/div, this->y/div, this->a/div);
    }

    double x;
    double y;
    double a;
};


class Stabilizer
{
private:

#ifdef CUDA_STAB
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
    cv::Ptr<cv::cuda::ORB> orb;
    cv::Ptr<cv::cuda::Filter> gauss_filter;
#else
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::ORB> orb;
#endif
    cv::Size process_size;

public:
    Stabilizer();
    Stabilizer(cv::Size size);
    ~Stabilizer();

    cv::Mat compute_affine(cv::Mat &base, cv::Mat &train, bool show_matches = false);
    void compute_vid_avg(char* source, char* dest, bool liveview = true, int window_size = 5, bool fixed = false);
    void compute_vid_staged(char* source, char* dest, bool liveview = true, int stage = 5, bool fixed = false);
};
    
} // namespace stabcv
