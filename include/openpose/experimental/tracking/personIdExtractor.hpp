#ifndef OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
#define OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>
#include <unordered_map>
namespace op
{

    struct person_entry
    {
        long long counter;
        std::vector<cv::Point2f> keypoints;
        std::vector<char> status;
        /*
        person_entry(long long _last_frame, 
                     std::vector<cv::Point2f> _keypoints,
                     std::vector<char> _active):
                     last_frame(_last_frame), keypoints(_keypoints),
                     active(_active)
                     {}
        */
    };
    class OP_API PersonIdExtractor
    {

    public:
        PersonIdExtractor();

        virtual ~PersonIdExtractor();

        Array<long long> extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput);

    private:
        long long mNextPersonId;
        bool init;
        cv::Mat previous_frame;
        std::vector<cv::Point2f> I;
        std::vector<cv::Point2f> J;
        std::vector<char> status;
        long long max_person;
        std::vector<person_entry> openpose_points;
        std::unordered_map<int,person_entry> lkanade_points;
        DELETE_COPY(PersonIdExtractor);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
