#include <openpose/experimental/tracking/lkpyramidal.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>
#include <iostream>
namespace op
{

    PersonIdExtractor::PersonIdExtractor() :
        mNextPersonId{0ll},
        init{false}
    {
        try
        {
            // error("PersonIdExtractor (`identification` flag) not available yet, but we are working on it! Coming"
            //       " soon!", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PersonIdExtractor::~PersonIdExtractor()
    {
    }
    

    Array<long long> PersonIdExtractor::extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput)
    {
        try
        {
            Array<long long> poseIds{poseKeypoints.getSize(0), -1};

            if (!init)
            {
                /* First frame initialization values */
                init = true;
                max_person = 0;

                /* Add first persons to the tracking map */
                int npersons = poseKeypoints.getSize(0);

                for (int p = 0; p < npersons; p++)
                {
                    long long framen = 0;
                    int actual_person = max_person++;
                    poseIds[p] = actual_person;

                    std::vector<char> active;
                    std::vector<cv::Point2f> keypoints;

                    for (int kp = 0; kp < poseKeypoints.getSize(1); kp++)
                    {
                        //std::cout<< poseKeypoints[{p,kp,0}]<<std::endl;
                        //std::cout<< poseKeypoints[{p,kp,1}]<<std::endl;
                        //std::cout<< poseKeypoints[{p,kp,2}]<<std::endl;
                        cv::Point2f cp;
                        cp.x = poseKeypoints[{p,kp,0}];
                        cp.y = poseKeypoints[{p,kp,1}];
                        keypoints.push_back(cp);

                        if (poseKeypoints[{p,kp,0}] < thres_conf)
                          active.push_back(1);
                        else
                           active.push_back(0); 
                            
                    }

                    /* Create and add person entry the trackinng map */
                    person_entry pe;
                    pe.keypoints = keypoints;
                    pe.last_frame = framen;
                    pe.active = active;
                    track_map[actual_person] = pe;
                }

                cvMatInput.convertTo(previous_frame, CV_32F);
                /*cv::Point2f d;
                d.x = 20.0;
                d.y = 22.0;
                I.push_back(d);
                status.push_back(0);
                */
            }
            else
            {
                cv::Mat current_frame;
                cvMatInput.convertTo(current_frame, CV_32F);
                runLKPyramidal(I, J, previous_frame, current_frame, status, 3, 21);
                previous_frame = current_frame.clone();
            }
            // Dummy: giving a new id to each element
            //Array<long long> poseIds{poseKeypoints.getSize(0), -1};
            //poseIds[343] = 34;
            // for (auto i = 0u ; i < poseIds.getVolume() ; i++)
            //     poseIds[i] = mNextPersonId++;
            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }
}
