#include <opencv2/contrib/contrib.hpp>

#ifndef CMESHMODELINSTANCE
#define CMESHMODELINSTANCE

struct TotalModel;

class CMeshModelInstance
{
public:

	enum EnumMeshType
	{
		MESH_TYPE_UNKNOWN = -1,
		MESH_TYPE_SMPL = 0,
		MESH_TYPE_FACE,
		MESH_TYPE_RHAND,
		MESH_TYPE_LHAND,
		MESH_TYPE_TOTAL,
		MESH_TYPE_ADAM
	};

	CMeshModelInstance()
	{
		m_meshType = MESH_TYPE_UNKNOWN;
		m_humanIdx = -1;
		//m_pSourceParam = NULL;
		m_sourceParamIdx = std::make_pair(-1, -1);

		m_visType = 0;	

		m_frameIdx = -1;
	}

	void RecomputeNormal(const TotalModel& model);

	std::vector<unsigned int> m_face_vertexIndices;
	std::vector<cv::Point3d> m_vertices;
	std::vector<cv::Point3d> m_colors;
	std::vector<cv::Point3d> m_normals;
	std::vector<cv::Point2d> m_uvs;
	std::vector<cv::Point3d> m_joints;
	std::vector<cv::Point3d> m_joints_regress;		//newly regressed joint
	int m_humanIdx;
	int m_visType;

	int m_frameIdx;

	std::pair<int, int> m_sourceParamIdx;		//Where this mesh was generated. g_kinModelManager.m_smpl_fitParms[m_sourceParamIdx.first].m_meshes.m_params[m_sourceParamIdx.secod]

	EnumMeshType m_meshType;
};

#endif
