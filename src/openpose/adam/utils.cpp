#include "utils.h"
#include <iostream>
#include <cmath>
#include <vector>

void model_size(double* joint, std::vector<int> connMat)
{
	double lhand_size = 0, body_size = 0, rhand_size = 0;
	for (int i = 0; i < connMat.size(); i += 2)
	{
		double length2 = (joint[3*connMat[i]] - joint[3*connMat[i+1]]) * (joint[3*connMat[i]] - joint[3*connMat[i+1]])
						+ (joint[3*connMat[i] + 1] - joint[3*connMat[i+1] + 1]) * (joint[3*connMat[i] + 1] - joint[3*connMat[i+1] + 1])
						+ (joint[3*connMat[i] + 2] - joint[3*connMat[i+1] + 2]) * (joint[3*connMat[i] + 2] - joint[3*connMat[i+1] + 2]);
		double length = sqrt(length2);
		if ((i >= 4 && i < 8) || (i >= 10 && i < 14) || (i >= 18 && i < 22) || (i >= 24 && i < 28))
			body_size += length;
		else if (i >= 36 && i < 76)
			lhand_size += length;
		else if (i >= 76)
			rhand_size += length;
	}
	std::cout << "body size: " << body_size << " lhand size: " << lhand_size << " rhand size: " << rhand_size << std::endl;
}