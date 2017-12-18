#include <openpose/experimental/tracking/lkpyramidal.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>
#include <iostream>
namespace op
{
    const float thres_conf = 0.1;
    const int idle_frame = 10;
    const float inlier_thr = 0.5;
    const float dist_thr = 30.0;


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

    float get_euclidean_distance(cv::Point2f a, cv::Point2f b)
    {

        return std::sqrt((a.x - b.x)*(a.x - b.x)*
                         + (a.y - b.y)*(a.y - b.y));

    }
    
    void caputre_keypoints(const Array<float>& poseKeypoints, std::vector<person_entry> &result)
    {
        int npersons = poseKeypoints.getSize(0);
        result.clear();

        for (int p = 0; p < npersons; p++)
        {
            std::vector<char> status;
            std::vector<cv::Point2f> keypoints;

            for (int kp = 0; kp < poseKeypoints.getSize(1); kp++)
            {
                cv::Point2f cp;
                cp.x = poseKeypoints[{p,kp,0}];
                cp.y = poseKeypoints[{p,kp,1}];
                keypoints.push_back(cp);

                if (poseKeypoints[{p,kp,0}] < thres_conf)
                  status.push_back(0);
                else
                   status.push_back(1); 
                    
            }

            /* Create and add person entry the trackinng map */
            person_entry pe;
            pe.keypoints = keypoints;
            pe.counter = 0;
            pe.status = status;
            result.push_back(pe);
        }        
    }

    void update_lkanade(std::unordered_map<int,person_entry> &lkanade_points, cv::Mat& prev, cv::Mat& next)
    {
        for (auto &entry: lkanade_points)
        {
            int idx = entry.first;

            if (lkanade_points[idx].counter++ > idle_frame)
            {
                lkanade_points.erase(idx);
                continue;
            }

            /* Update all keypoints for that entry */
            std::vector<cv::Point2f> new_points;

            runLKPyramidal(lkanade_points[idx].keypoints, new_points, prev, next, 
                             lkanade_points[idx].status, 3, 21);

            person_entry pe;
            pe.keypoints = new_points;
            pe.counter = 0;
            pe.status = lkanade_points[idx].status;
            lkanade_points[idx] = pe;
        }
    }

    void init_lkanade(std::unordered_map<int,person_entry>& lkanade_points,
                     const Array<float>& poseKeypoints,
                     long long &max_person)
    {
        int npersons = poseKeypoints.getSize(0);

        for (int p = 0; p < npersons; p++)
        {
            int actual_person = max_person++;

            std::vector<char> status;
            std::vector<cv::Point2f> keypoints;

            for (int kp = 0; kp < poseKeypoints.getSize(1); kp++)
            {
                cv::Point2f cp;
                cp.x = poseKeypoints[{p,kp,0}];
                cp.y = poseKeypoints[{p,kp,1}];
                keypoints.push_back(cp);

                if (poseKeypoints[{p,kp,0}] < thres_conf)
                  status.push_back(1);
                else
                   status.push_back(0);         
            }

            /* Create and add person entry the tracking map */
            person_entry pe;
            pe.keypoints = keypoints;
            pe.counter = 0;
            pe.status = status;
            lkanade_points[actual_person] = pe;
        }        
    }

    Array<long long> match_lk_vs_op(std::unordered_map<int,person_entry>& lkanade_points,
                                    std::vector<person_entry> &openpose_points,
                                    long long &max_person)
    {

        Array<long long> poseIds{openpose_points.size(), -1};
        std::unordered_map<int,person_entry> pending_queue;


        for (int i = 0; i < openpose_points.size(); i++)
        {
            long long best_match = -1;
            float best_score = 0.0;

            /* Find best correspondance in the LK set */
            for (auto &entry_lk: lkanade_points)
            {
                int inliers = 0, active = 0;
                int idx = entry_lk.first;

                /* Iterate through all keypoints */
                for (int kp = 0; kp < openpose_points[0].keypoints.size(); kp++)
                {
                    /* Not enough threshold */
                    if (lkanade_points[idx].status[kp] || openpose_points[i].status[kp])
                        continue;

                    active ++;
                    float dist = get_euclidean_distance(lkanade_points[idx].keypoints[kp],
                                                        openpose_points[i].keypoints[kp]);
                    if (dist < dist_thr)
                        inliers ++;
                }

                float score = 0.0;

                if (inliers) score = (float) inliers / (float) active;;

                if (score >= inlier_thr && score > best_score)
                {
                    best_score = score;
                    best_match = entry_lk.first;
                }
            }
            /* Found a best match, update LK table and poseIds */
            if (best_match != -1)
                poseIds[i] = best_match;
            else
                poseIds[i] = max_person++;
            
            pending_queue[poseIds[i]] = openpose_points[i];

        }

        /* Update LK table with pending queue */
        for (auto &entry_q: pending_queue)
            lkanade_points[entry_q.first] = entry_q.second;
        
        return poseIds;
    }
    Array<long long> PersonIdExtractor::extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput)
    {
        try
        {
            Array<long long> poseIds;
            caputre_keypoints(poseKeypoints, openpose_points);

            if (!init)
            {
                /* First frame initialization values */
                init = true;
                max_person = 0;

                /* Add first persons to the lknade set */
                init_lkanade(lkanade_points, poseKeypoints, max_person);

                /* Capture current frame as floating point */
                cvMatInput.convertTo(previous_frame, CV_32F);
            }
            else
            {
                cv::Mat current_frame;
                cvMatInput.convertTo(current_frame, CV_32F);
                update_lkanade(lkanade_points, previous_frame, current_frame);
                previous_frame = current_frame.clone();
            }

            /* Get poseIds and update LKset according to OpenPose set */
            poseIds = match_lk_vs_op(lkanade_points,openpose_points,max_person);

            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }
}
