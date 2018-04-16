#include <BVHWriter.h>
#include <cassert>
#include <ceres/rotation.h>
#include <cmath>
#include <fstream>

#define PI 3.1415926

const char* joint_name[] = {"center", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "neck", "left_feet", "right_feet", "head1",
    "left_armpit", "right_armpit", "head2", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4", "left_index1", "left_index2", "left_index3", "left_index4",
    "left_middle1", "left_middle2", "left_middle3", "left_middle4", "left_ring1", "left_ring2", "left_ring3", "left_ring4",
    "left_little1", "left_little2", "left_little3", "left_little4",
    "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4", "right_index1", "right_index2", "right_index3", "right_index4",
    "right_middle1", "right_middle2", "right_middle3", "right_middle4", "right_ring1", "right_ring2", "right_ring3", "right_ring4",
    "right_little1", "right_little2", "right_little3", "right_little4"
};

void RotationMatrixToEulerAngle(const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>& R, std::array<double, 3>& angle)
{
	if (R(2, 0) < 1.0)
	{
		if (R(2, 0) > -1.0)
		{
			angle.at(0) = atan2(R(2, 1), R(2, 2));
			angle.at(1) = -asin(R(2, 0));
			angle.at(2) = atan2(R(1, 0), R(0, 0));
		}
		else
		{
			angle.at(0) = 0.0;
			angle.at(1) = PI / 2;
			angle.at(2) = -atan2(R(1, 2), R(1, 1));
		}
	}
	else
	{
		angle.at(0) = 0.0;
		angle.at(1) = -PI / 2;
		angle.at(2) = -atan2(R(1, 2), R(1, 1));
	}

	for (int i = 0; i < 3; i++) angle.at(i) = angle.at(i) * 180 / PI;
}

void BVHWriter::parseInput(const Eigen::Matrix<double, 3 * TotalModel::NUM_JOINTS, 1>& J0, std::vector<Eigen::Matrix<double, 3, 1>>& t, std::vector<Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>>& pose)
{
	assert(t.size() == pose.size());
	this->num_frame = t.size();
	this->trans = t;

	getHierarchy(J0);
	for (int time = 0; time < this->num_frame; time++)
	{
		Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>& pose_frame = pose[time];
		getDynamic(pose_frame);
	}
	if (mUnityCompatible)
	{
		// add additional joints
		// left_hand
		std::shared_ptr<BVHData> left_hand_node = std::make_shared<BVHData>(std::string("left_hand"), this->data[20]->offset.data());
		for (int time = 0; time < this->num_frame; time++)
		{
			std::array<double, 3> left_hand_angle = {0.0, 0.0, 0.0};
			left_hand_node->euler.push_back(left_hand_angle);
		}
		left_hand_node->children = this->data[20]->children;
		this->data[20]->children.clear();
		this->data[20]->children.push_back(left_hand_node);
		for(int i = 0; i < 3; i++) this->data[20]->offset[i] = 0.0;

		// right_hand
		std::shared_ptr<BVHData> right_hand_node = std::make_shared<BVHData>(std::string("right_hand"), this->data[21]->offset.data());
		for (int time = 0; time < this->num_frame; time++)
		{
			std::array<double, 3> right_hand_angle = {0.0, 0.0, 0.0};
			right_hand_node->euler.push_back(right_hand_angle);
		}
		right_hand_node->children = this->data[21]->children;
		this->data[21]->children.clear();
		this->data[21]->children.push_back(right_hand_node);
		for(int i = 0; i < 3; i++) this->data[21]->offset[i] = 0.0;
	}
}

void BVHWriter::writeBVH(std::string output_file, double frame_time)
{
	outStr.clear();
	for (int i = 0; i < this->num_frame; i++) dynamicStr.push_back(std::string());

	outStr += "HIERARCHY\n";
	outStr += (std::string("ROOT ") + this->root->name);
	outStr += "\n{\n";
	outStr += "\tOFFSET 0.0 0.0 0.0\n";
	outStr += "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n";

	for (int i = 0; i < this->num_frame; i++)
		dynamicStr[i] += std::to_string(this->trans[i](0) + this->root->offset[0]) + " " + std::to_string(this->trans[i](1) + this->root->offset[1]) + " " + std::to_string(this->trans[i](2) + this->root->offset[2]);
	if (mUnityCompatible)
		for (int i = 0; i < this->num_frame; i++)
			dynamicStr[i] += " 0.0 0.0 180.0";
	else
		for (int i = 0; i < this->num_frame; i++)
			dynamicStr[i] += " 0.0 0.0 0.0";

	writeData(this->root, 0);

	outStr += "}\n";
	outStr += "MOTION\n";
	outStr += ("Frames: " + std::to_string(this->num_frame) + "\n");
	outStr += ("Frame Time: " + std::to_string(frame_time) + "\n");

	for (int i = 0; i < this->num_frame; i++)
		outStr += (dynamicStr[i] + "\n");

	std::ofstream out(output_file.c_str(), std::ios::out);
	out << outStr;
	out.close();
}

void BVHWriter::getHierarchy(const Eigen::Matrix<double, 3 * TotalModel::NUM_JOINTS, 1>& J0)
{
	int idj = 0;
	double offset[3] = {J0(3 * idj), J0(3 * idj + 1), J0(3 * idj + 2)};
	this->root = std::make_shared<BVHData>(std::string(joint_name[idj]), offset);
	this->data[idj] = this->root;

	for (idj = 1; idj < TotalModel::NUM_JOINTS; idj++)
	{
		int idp = this->m_parent[idj];
		offset[0] = J0(3 * idj) - J0(3 * idp);
		offset[1] = J0(3 * idj + 1) - J0(3 * idp + 1);
		offset[2] = J0(3 * idj + 2) - J0(3 * idp + 2);
		std::shared_ptr<BVHData> node = std::make_shared<BVHData>(std::string(joint_name[idj]), offset);
		this->data.at(idp)->children.push_back(node);
		this->data.at(idj) = node;
	}
}

void BVHWriter::getDynamic(Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>& pose)
{
	int idj = 0;
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R;
	std::array<double, 3> angle;
	ceres::AngleAxisToRotationMatrix(pose.data() + 3 * idj, R.data());
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> RT;
	RT << R.transpose();
	RotationMatrixToEulerAngle(RT, angle);
	this->root->euler.push_back(angle);

	for (idj = 1; idj < TotalModel::NUM_JOINTS; idj++)
	{
		ceres::EulerAnglesToRotationMatrix(pose.data() + 3 * idj, 3, R.data());
		RT << R.transpose();
		std::array<double, 3> angle;
		RotationMatrixToEulerAngle(RT, angle);
		this->data[idj]->euler.push_back(angle);
	}
}

void BVHWriter::writeData(std::shared_ptr<BVHData> node, int depth)
{
	if (node->children.size() > 0)
	{
		for (auto& child: node->children)
		{
			for (int i = 0; i < depth + 1; i++) outStr += "\t";
			outStr +=  "JOINT ";
			outStr += (child->name + "\n");
			for (int i = 0; i < depth + 1; i++) outStr += "\t";
			outStr += "{\n";
			for (int i = 0; i < depth + 2; i++) outStr += "\t";
			outStr += "OFFSET ";
			if (depth == 0)
			{ // root node
				outStr += "0.0 0.0 0.0\n";
			}
			else
			{
				outStr += std::to_string(node->offset[0]) + " " + std::to_string(node->offset[1]) + " " + std::to_string(node->offset[2]) + "\n";
			}
			for (int i = 0; i < depth + 2; i++) outStr += "\t";
			outStr += "CHANNELS 3 Zrotation Yrotation Xrotation\n";
			for (int j = 0; j < this->num_frame; j++)
				dynamicStr[j] += (" " + std::to_string(node->euler[j][2]) + " " + std::to_string(node->euler[j][1]) + " " + std::to_string(node->euler[j][0]));
			writeData(child, depth + 1);
			for (int i = 0; i < depth + 1; i++) outStr += "\t";
			outStr += "}\n";
		}
	}
	else
	{
		for (int i = 0; i < depth + 1; i++) outStr += "\t";
		outStr += "End Site\n";
		for (int i = 0; i < depth + 1; i++) outStr += "\t";
		outStr +=  "{\n";
		for (int i = 0; i < depth + 2; i++) outStr += "\t";
		outStr += "OFFSET ";
		outStr += std::to_string(node->offset[0]) + " " + std::to_string(node->offset[1]) + " " + std::to_string(node->offset[2]) + "\n";
		for (int i = 0; i < depth + 1; i++) outStr += "\t";
		outStr +=  "}\n";
	}
}