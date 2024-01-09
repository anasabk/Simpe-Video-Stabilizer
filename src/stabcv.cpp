#include <opencv4/opencv2/opencv.hpp>
#include <deque>
#include "stabcv.hpp"

#ifdef CUDA_STAB
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#endif


stabcv::Stabilizer::Stabilizer()
    :   process_size(0, 0),
#ifdef CUDA_STAB
        matcher(cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING)),
        orb(cv::cuda::ORB::create()),
        gauss_filter(cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 2, 2))
#else
        matcher(cv::BFMatcher::create(cv::NORM_HAMMING)),
        orb(cv::ORB::create())
#endif
{
}

stabcv::Stabilizer::Stabilizer(cv::Size size)
    :   process_size(size),
#ifdef CUDA_STAB
        matcher(cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING)),
        orb(cv::cuda::ORB::create()),
        gauss_filter(cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 2, 2))
#else
        matcher(cv::BFMatcher::create(cv::NORM_HAMMING)),
        orb(cv::ORB::create())
#endif
{
}

stabcv::Stabilizer::~Stabilizer() {

}

cv::Mat stabcv::Stabilizer::compute_affine(cv::Mat &base, cv::Mat &train, bool show_matches) {
    bool resize_flag = true;
    if(process_size.width == 0 || process_size.height == 0)
        resize_flag = false;

#ifdef CUDA_STAB
    cv::cuda::GpuMat base_buf, train_buf;

    base_buf.upload(base);
    train_buf.upload(train);

    if(resize_flag) {
        cv::cuda::resize(base_buf, base_buf, process_size);
        cv::cuda::resize(train_buf, train_buf, process_size);
    }

    if(base.channels() == 3)
        cv::cuda::cvtColor(base_buf, base_buf, cv::COLOR_BGR2GRAY);
    if(train.channels() == 3)
        cv::cuda::cvtColor(train_buf, train_buf, cv::COLOR_BGR2GRAY);

    gauss_filter->apply(base_buf, base_buf);
    gauss_filter->apply(train_buf, train_buf);
#else
    cv::Mat base_buf = base.clone(), train_buf = train.clone();

    if(resize_flag) {
        cv::resize(base_buf, base_buf, process_size);
        cv::resize(base_buf, train_buf, process_size);
    }

    if(base.channels() == 3)
        cv::cvtColor(base_buf, base_buf, cv::COLOR_BGR2GRAY);
    if(train.channels() == 3)
        cv::cvtColor(train_buf, train_buf, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(base_buf, base_buf, cv::Size(5, 5), 0);
    cv::GaussianBlur(train_buf, train_buf, cv::Size(5, 5), 0);
#endif

    // Detect the features
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    orb->detect(base_buf ,keypoints1);
    orb->detect(train_buf ,keypoints2);

    // Compute BRIEF descriptors.
    cv::Mat descriptors1, descriptors2;
    orb->compute(base_buf, keypoints1, descriptors1);
    orb->compute(train_buf, keypoints2, descriptors2);

    // Match the features.
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Find min and max distances in the match features.
    double min_dist = 10000;
    for(int i = 0; i < matches.size(); i++)
        if(matches[i].distance < min_dist) min_dist = matches[i].distance;

    // Choose matches whose distances are less than 2*min_distance or 30
    std::vector< cv::DMatch > good_matches;
    for (int i = 0; i < matches.size(); i++)
        if(matches[i].distance <= std::max(2*min_dist, 30.0))
            good_matches.push_back(matches[i]);

    // Create lists of matched features.
    std::vector<cv::Point2f> obj(good_matches.size()), scene(good_matches.size());
    for(size_t i = 0; i < good_matches.size(); i++) {
        scene[i] = keypoints1[good_matches[i].queryIdx].pt;
        obj[i] = keypoints2[good_matches[i].trainIdx].pt;
    }

    if(show_matches) {
        cv::Mat out, base_feat, train_feat;
        cv::drawKeypoints(base, keypoints1, base_feat);
        cv::drawKeypoints(train, keypoints2, train_feat);
        cv::drawMatches(
            base, 
            keypoints1, 
            train, 
            keypoints2, 
            good_matches, 
            out
        );
        imshow("base features", base_feat);
        imshow("train features", train_feat);
        imshow("matched", out);
        imwrite("base features.jpg", base_feat);
        imwrite("train features.jpg", train_feat);
        imwrite("matched.jpg", out);
        cv::waitKey(0);
    }

    return estimateAffine2D(obj, scene);
}

void stabcv::Stabilizer::compute_vid_staged(char* source, char* dest, bool liveview, int stage, bool fixed) {
    char dir_template[] = "temp/stabilzer_XXXXXX";
    mkdtemp(dir_template);

    cv::VideoWriter temp_buf;
    cv::VideoCapture temp_read;

    double min_dist=10000, dist;
    char buf_in[512], buf_out[512];
    strcpy(buf_in, source);

    for(int cur_stage = 0; cur_stage < stage; cur_stage++) {
        if(cur_stage == stage -1) {
            stpcpy(buf_out, dest);
            compute_vid_avg(buf_in, buf_out, liveview, cur_stage, fixed);

        } else {
            sprintf(buf_out, "%s/temp_stable%d.avi", dir_template, cur_stage);
            compute_vid_avg(buf_in, buf_out, liveview, cur_stage, false);
        }
        stpcpy(buf_in, buf_out);
    }

    sprintf(buf_in, "rm -rf %s", dir_template);
    system(buf_in);
}

void stabcv::Stabilizer::compute_vid_avg(char* source, char* dest, bool liveview, int avg_window, bool fixed) {
    bool resize_flag = true;
    if(process_size.width == 0 || process_size.height == 0)
        resize_flag = false;

    cv::VideoCapture input(source);
    cv::VideoWriter output(
        dest, 
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
        30, 
        cv::Size(input.get(cv::CAP_PROP_FRAME_WIDTH), input.get(cv::CAP_PROP_FRAME_HEIGHT))
    );

#ifdef CUDA_STAB
    cv::cuda::GpuMat cur_buf, gen_buf, prev_desc, cur_desc;
#else
    cv::Mat cur_buf, gen_buf, prev_desc, cur_desc;
#endif
    std::vector<cv::KeyPoint> prev_feat, cur_feat;
    std::vector<cv::DMatch> matches, good_matches;
    std::vector<cv::Point2f> obj, scene;
    std::queue<cv::Mat> temp_affine_queue;
    cv::Mat affine_buf, affine_avg, cur, gen;

    double min_dist;
    char buffer[512];
    strcpy(buffer, source);
    while(true) {
        input >> cur;
        if(cur.data == NULL)
            break;

        // cv::Rect crop(0, 15, 640, 465);
        // cur = cur(crop);

#ifdef CUDA_STAB
        cur_buf.upload(cur);

        if(resize_flag)
            cv::cuda::resize(cur_buf, cur_buf, process_size);

        cv::cuda::cvtColor(cur_buf, cur_buf, cv::COLOR_BGR2GRAY);
        gauss_filter->apply(cur_buf, cur_buf);
#else
        if(resize_flag)
            cv::resize(cur_buf, cur_buf, process_size);
        else
            cur_buf = cur.clone();

        cv::cvtColor(cur_buf, cur_buf, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(cur_buf, cur_buf, cv::Size(5, 5), 0);
#endif
        orb->detectAndCompute(cur_buf, cv::noArray(), cur_feat, cur_desc);

        if(prev_desc.empty()) {
            cur_feat.swap(prev_feat);
            cv::swap(cur_desc, prev_desc);
            continue;
        } else if(cur_desc.empty()) {
            continue;
        }

        matcher->match(prev_desc, cur_desc, matches);

        // Find min and max distances in the match features.
        min_dist = 10000;
        for(int i = 0; i < matches.size(); i++)
            if(matches[i].distance < min_dist) min_dist = matches[i].distance;

        // Choose matches whose distances are between 30 and min_distance^2
        good_matches.clear();
        for (int i = 0; i < matches.size(); i++)
            if(matches[i].distance <= std::max(2*min_dist, 30.0))
                good_matches.push_back(matches[i]);

        // Create lists of matched features.
        scene.clear();
        obj.clear();
        for(size_t i = 0; i < good_matches.size(); i++) {
            scene.push_back(prev_feat[good_matches[i].queryIdx].pt);
            obj.push_back(cur_feat[good_matches[i].trainIdx].pt);
        }

        if(obj.empty() || scene.empty() || (affine_buf = estimateAffinePartial2D(obj, scene)).empty())
            affine_buf = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
        // std::cout << affine_buf << std::endl;

        // Create the averaged affine matrix
        if(avg_window == 0) {
            affine_avg = affine_buf;

        } else if(temp_affine_queue.size() >= avg_window) {
            affine_avg += (affine_buf - temp_affine_queue.front())/(avg_window);
            temp_affine_queue.pop();
            temp_affine_queue.push(affine_buf);

        } else {
            if(affine_avg.empty())
                affine_avg = affine_buf/(avg_window);
            else
                affine_avg += affine_buf/(avg_window);
            temp_affine_queue.push(affine_buf);

            if(!fixed) {
                cur_feat.swap(prev_feat);
                cur_desc.copyTo(prev_desc);
            }
            continue;
        }

        // Apply the affine matrix
#ifdef CUDA_STAB
        cur_buf.upload(cur);
        cv::cuda::warpAffine(cur_buf, gen_buf, affine_avg, cur_buf.size(), cv::INTER_CUBIC);
        gen_buf.download(gen);
#else
        cv::warpAffine(cur, gen, affine_avg, cur.size(), cv::INTER_CUBIC);
#endif
        output << gen;

        if(liveview) {
            imshow("current", cur);
            imshow("stabalized", gen);
            cv::waitKey((int)(1000/input.get(cv::CAP_PROP_FPS)));
        }

        if(!fixed) {
            cur_feat.swap(prev_feat);
            cur_desc.copyTo(prev_desc);
        }
    }
}
