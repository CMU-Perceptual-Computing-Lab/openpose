#ifndef VISUALIZEDATA
#define VISUALIZEDATA

#include <opencv2/contrib/contrib.hpp>
#include <vector>
#include <utility>
#include <GL/glut.h>

using namespace std;

enum EnumRenderType
{
	RENDER_None=-1,
	RENDER_Play_Frames=0,
	RENDER_Play_CamMove=1,			//3DPS
};

class VisualizedData
{
public:
	VisualizedData()
	{
		//Init color Map
		cv::Mat colorMapSource = cv::Mat::zeros(256,1,CV_8U);
		for(unsigned int i=0;i<=255;i++)
			colorMapSource.at<uchar>(i,0) = i;
		cv::Mat colorMap;
		cv::applyColorMap(colorMapSource, colorMap, cv::COLORMAP_JET);
		for(unsigned int i=0;i<=255;i++)
		{
			cv::Point3f tempColor;
			tempColor.z = colorMap.at<cv::Vec3b>(i,0)[0]/255.0; //blue
			tempColor.y = colorMap.at<cv::Vec3b>(i,0)[1]/255.0; //green
			tempColor.x = colorMap.at<cv::Vec3b>(i,0)[2]/255.0;	//red
			m_colorMapGeneral.push_back(tempColor);
		}

		m_backgroundColor = cv::Point3f(1,1,1);
		m_pPickedBoneGroup = NULL;

		//Set frame as local part's frame
		for(int i=0;i<16;++i)
			m_AnchorMatrixGL[i]=0;
		m_AnchorMatrixGL[0] = 1;
		m_AnchorMatrixGL[5] = 1;
		m_AnchorMatrixGL[10] = 1;
		m_AnchorMatrixGL[15] = 1;
		
		bShowBackgroundTexture = false;
		m_renderType=RENDER_None;
		m_selectedHandIdx = make_pair(0,0);
		//g_shaderProgramID =0;
		//m_reloadShader = false;
		read_buffer = NULL;
		targetJoint = NULL;
		resultJoint = NULL;
		vis_type = 0;

		int hand[] = {0, 1, 1, 2, 2, 3, 3, 4,
			0, 5, 5, 6, 6, 7, 7, 8,
			0, 9, 9, 10, 10, 11, 11, 12,
			0, 13, 13, 14, 14, 15, 15, 16,
			0, 17, 17, 18, 18, 19, 19, 20
		};
		std::vector<int> connMat_hand(hand, hand + sizeof(hand) / sizeof(int));
		connMat.push_back(connMat_hand);

		int body[] = {
			0, 1,
			0, 3, 3, 4, 4, 5,
			0, 9, 9, 10, 10, 11,
			0, 2,
			2, 6, 6, 7, 7, 8,
			2, 12, 12, 13, 13, 14,
			1, 15, 15, 16,
			1, 17, 17, 18,
			0, 19
		};
		std::vector<int> connMat_body(body, body + sizeof(body) / sizeof(int));
		connMat.push_back(connMat_body);

		std::vector<int> connMat_total(body, body + sizeof(body) / sizeof(int));
		std::vector<int> connMat_lhand(0);
		std::vector<int> connMat_rhand(0);
		for (auto i = 0u; i < connMat_hand.size(); i++)
		{
			connMat_total.push_back(connMat_hand[i] + 20); // left hand
			connMat_lhand.push_back(connMat_hand[i] + 20);
		}
		for (auto i = 0u; i < connMat_hand.size(); i++)
		{
			connMat_total.push_back(connMat_hand[i] + 41); // right hand
			connMat_rhand.push_back(connMat_hand[i] + 41);
		}
		connMat.push_back(connMat_total);
		connMat.push_back(connMat_lhand);
		connMat.push_back(connMat_rhand);
	}

	~VisualizedData() {}
	
	// vector<CamVisInfo> m_camVisVector;
	// vector<textureCoord> m_camTexCoord;
	// bool m_loadNewCamTextureTrigger;
	// vector<cv::Mat> m_camTextureImages;	//Num should be the same as m_camVisVector
	// vector<CamVisInfo> m_newlyRegisteredCamVisVector;
	// vector<cv::Point3f> m_cameraColorVectorByMotionCost;
	// vector<cv::Point3f> m_cameraColorVectorByNormalCost;
	// vector<cv::Point3f> m_cameraColorVectorByAppearanceCost;
	// vector<cv::Point3f> m_cameraColorVectorByTotalDataCost;

	//Patch Clound & Trajectory Stream
	// vector<PatchCloudUnit> m_patchCloud;
	// vector< pair<cv::Point3d,cv::Point3d> > m_trajectoryTotal;  //line
	// vector< float> m_trajectoryTotal_alpha;  //line

	//Mesh Structure;
	vector<cv::Point3d> m_meshVertices;		
	vector<cv::Point3d> m_meshVerticesColor;		
	vector<cv::Point3d> m_meshVerticesNormal;
	vector<cv::Point2d> m_meshVerticesUV;
	vector<unsigned int> m_meshIndices;
	vector< pair<cv::Point3f, cv::Point3f> > m_meshJoints;		//point, color

	// vector<CSkeletonVisUnit> m_skeletonVisVector;
	// vector<CSkeletonVisUnit> m_skeletonVisVectorCompare;
	void* m_pPickedBoneGroup;		//type is CPartTrajProposal*
	GLfloat m_AnchorMatrixGL[16]; //To make normalize coordinate w.r.t the selected bone

	//Face
	vector< cv::Point3d > m_faceCenters; //point,color
	vector< pair<cv::Point3d,cv::Point3d> > m_faceLandmarks; //point,color
	vector< pair<cv::Point3d,cv::Point3d> > m_faceNormal;//point,color
	vector< pair<cv::Point3d,cv::Point3d> > m_faceLandmarksGT; //point,color
	vector< pair<cv::Point3d,cv::Point3d> > m_faceNormalGT;//point,color
	vector< pair<cv::Point3d,cv::Point3d> > m_faceLandmarks_pm; //point,color
	vector<pair<cv::Point3d, string > > m_faceNames_pm;
	GLfloat m_face_pm_modelViewMatGL[16];
	// vector< CFaceVisUnit > m_faceAssociated;

	//SSP 
	vector< pair<cv::Point3d,cv::Point3d> > m_gazePrediction; //point,color


	//Hands
	vector< pair<cv::Point3d,cv::Point3d> > m_handLandmarks; //point,color
	vector< pair<cv::Point3d,cv::Point3d> > m_handNormal;//point,color
	GLfloat m_hand_modelViewMatGL[16];
	pair<int,int> m_selectedHandIdx;


	//Foot
	vector< pair<cv::Point3d, cv::Point3d> > m_footLandmarks; //point,color
	vector< pair<cv::Point3d, cv::Point3d> > m_footLandmarks_ankleBone; //point,color

	//Gaze Engagement
	vector< pair<cv::Point3d, cv::Point3d> > m_gazeEngInfo;

	//Visual Hull Generation
	// CVolumeVisUnit m_visualHullVisualizeUnit;
	// CVolumeVisUnit m_faceNodeProposalVisUnit;
	// CVolumeVisUnit m_nodeProposalVisUnit;
	
	// //General purpose component
	// vector< VisStrUnit> m_debugStr; //point,color
	// vector< pair<cv::Point3d, pair< cv::Point3d, float> > > m_debugLinesWithAlpha; //point,color
	// vector< pair<cv::Point3d,cv::Point3d> > m_superTrajLines; //point,color
	// vector< pair<cv::Point3d,cv::Point3d> > m_debugLines; //point,color
	// vector< pair<cv::Point3d,cv::Point3d> > m_debugPt; //point,color
	// vector< pair<cv::Point3d,cv::Point3d> > m_debugSphere; //point,color
	// vector< pair<Bbox3D,cv::Point3d> > m_debugCubes;
	// //I made an additional layer to use "debug display tools"
	// //Instead of addition directly to (for example) m_debugPt,
	// //Add data to m_debugPtData, which draw it through m_debugPt, in SFMManager.visualizeEvertyhing() function
	// //I do this 1) m_debugPt is cleared in every visualization, so need to add there in every step again
	// //			2) m_debugPt can be used in  other data structure for simple visualization 
	// vector< pair<cv::Point3d,cv::Point3d> > m_debugLineData; //point,color
	// vector< pair<cv::Point3d,cv::Point3d> > m_debugPtData; //point,color
	// vector< pair<cv::Point3d,cv::Point3d> > m_debugSphereData; //point,color
	// vector< VisStrUnit> m_debugStrData; //point,color

	//Rendering and Save to Images
	// bool m_saveToFileTrigger;
	EnumRenderType m_renderType;		
	// void SaveToImage(bool bCamView=false,bool bFrameIdxAsName=false);

	cv::Point3f m_backgroundColor;	//opengl background color

	// //Texture of Patch Cloud. Not used anymore
	// vector<textureCoord> g_trajTextCoord;
	// vector<cv::Mat*> m_PatchVector;

	vector< cv::Point3f > m_colorMapGeneral;    //0~255, 0:bluish,    255:reddish   0<colorValueRGB <1 in order to be used in OpenGL
	// cv::Point3f GetColorByCost(float cost,float minCost,float maxCost);

	//background texture
	//bool bReLoadBackgroundTexture;
	bool bShowBackgroundTexture;

	double* targetJoint;
	double* resultJoint;
	uint vis_type; // 0 for hand, 1 for body, 2 for body with hands
	std::vector<std::vector<int>> connMat;

	//Shader
	//bool m_reloadShader;
	//GLuint m_shaderProgramID;
	GLubyte* read_buffer;
};

struct VisualizationOptions
{
	VisualizationOptions(): K(NULL), xrot(0.0f), yrot(0.0f), view_dist(300.0f), nRange(40.0f), CameraMode(0u), show_joint(true),
		ortho_scale(1.0f), width(600), height(600), zmin(0.01f), zmax(1000.0f), meshSolid(false) {}
	double* K;
	GLfloat	xrot, yrot;
	GLfloat view_dist, nRange;
	uint CameraMode;
	float ortho_scale;
	GLint width, height;
	GLfloat zmin, zmax; // used only in camera mode (to determine the range of objects in z direction)
	bool meshSolid;
	bool show_joint;
};

#endif