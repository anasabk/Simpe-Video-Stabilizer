#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudafeatures2d.hpp>


namespace stabcv
{

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
    void compute_vid_avg(char* source, char* dest, bool liveview = true, int stage = 5, bool fixed = false);
    void compute_vid_staged(char* source, char* dest, bool liveview = true, int stage = 5, bool fixed = false);
};
    
} // namespace stabcv
