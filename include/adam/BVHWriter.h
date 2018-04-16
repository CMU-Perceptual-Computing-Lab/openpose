#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include <totalmodel.h>

struct BVHData
{
	BVHData(std::string _name, double* _offset): name(_name), num_children(0)
	{
		offset[0] = _offset[0];
		offset[1] = _offset[1];
		offset[2] = _offset[2];
	}
	std::string name;
	std::array<double, 3> offset;
	std::vector<std::shared_ptr<BVHData>> children;
	std::vector<std::array<double, 3>> euler;
	int num_children;
};

class BVHWriter
{
public:
	BVHWriter(int* _m_parent, const bool unityCompatible = false) :
		mUnityCompatible{unityCompatible}
	{
		std::copy(_m_parent, _m_parent + TotalModel::NUM_JOINTS, m_parent);
	}
	void parseInput(const Eigen::Matrix<double, 3 * TotalModel::NUM_JOINTS, 1>& J0, std::vector<Eigen::Matrix<double, 3, 1>>& t, std::vector<Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>>& pose);
	void writeBVH(std::string output_file, double frame_time);
private:
	void getHierarchy(const Eigen::Matrix<double, 3 * TotalModel::NUM_JOINTS, 1>& J0);
	void getDynamic(Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>& pose);
	void writeData(std::shared_ptr<BVHData> node, int depth);
	const bool mUnityCompatible;
	int m_parent[TotalModel::NUM_JOINTS];
	int num_frame;
	std::vector<Eigen::Matrix<double, 3, 1>> trans;
	std::string outStr;
	std::vector<std::string> dynamicStr;
	std::shared_ptr<BVHData> root;
	std::array<std::shared_ptr<BVHData>, TotalModel::NUM_JOINTS> data;
};
