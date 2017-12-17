#ifndef OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
#define OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>
#include <unordered_map>
namespace op
{

    class OP_API PersonIdExtractor
    {

    public:
        PersonIdExtractor();

        virtual ~PersonIdExtractor();

        Array<long long> extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput);

    private:
        struct person_entry
        {
            long long last_frame;
            std::vector<cv::Point2f> keypoints;
            std::vector<char> active;
            /*
            person_entry(long long _last_frame, 
                         std::vector<cv::Point2f> _keypoints,
                         std::vector<char> _active):
                         last_frame(_last_frame), keypoints(_keypoints),
                         active(_active)
                         {}
            */
        };
        long long mNextPersonId;
        bool init;
        cv::Mat previous_frame;
        std::vector<cv::Point2f> I;
        std::vector<cv::Point2f> J;
        std::vector<char> status;
        long long max_person;
        std::unordered_map<int,person_entry> track_map;
        const float thres_conf = 0.1;

        DELETE_COPY(PersonIdExtractor);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
