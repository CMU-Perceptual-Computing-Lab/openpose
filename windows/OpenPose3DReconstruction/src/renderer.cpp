#include <stdio.h>
#include <openpose3d/renderer.hpp>
#include <GL/glut.h>

struct SDispUnit
{
	op::Array<float> mPoseKeypoints;
	op::Array<float> mFaceKeypoints;
	op::Array<float> mLeftHandKeypoints;
	op::Array<float> mRightHandKeypoints;
	bool mKeypointsValid;
};

SDispUnit gDispUnit3d;

static auto gRatio = 16.f / 9.f;
static auto gWidth = 600;                         // Initial window width
static auto gHeight = 600;                        // Initial window height
static GLfloat gNearPlaneForDefaultRender = -100; //0.01;
static GLfloat gFarPlaneForDefaultRender = 1000;

const GLfloat LIGHT_DIFFUSE[] = {1.f, 1.f, 1.f, 1.f};  /* Red diffuse light. */
const GLfloat LIGHT_POSITION[] = {1.f, 1.f, 1.f, 0.f};  /* Infinite light location. */
const GLfloat COLOR_DIFFUSE[] = { 0.f, 1.f, 1.f, 1.f};
const GLfloat COLOR_AMBIENT[] = { 0.f, 0.7f, 0.7f, 1.f};

//View Change by Mouse
static bool gBButton1Down = false;
static auto gXClick = 0.f;
static auto gYClick = 0.f;
static auto gGViewDistance = -82.3994f; //-45;
static auto gMouseXRotate = -63.2f; //0;
static auto gMouseYRotate = 7.f; //60;
static auto gMouseXPan = -69.2f; // 0;
static auto gMouseYPan = -29.9501f; // 0;
static auto gMouseZPan = 0.f;
static auto gScaleForMouseMotion = 0.1f;

enum class CameraMode {
	CAM_DEFAULT,
	CAM_ROTATE,
	CAM_ZOOM,
	CAM_PAN,
	CAM_PAN_Z
};
static CameraMode gCameraMode = CameraMode::CAM_DEFAULT;

const auto RADPERDEG = 0.0174532925199433;

void DrawConeByTwoPts(cv::Point3f& pt1,cv::Point3f& pt2,float ptSize)
{
	const GLdouble x1 = pt1.x;
	const GLdouble y1 = pt1.y;
	const GLdouble z1 = pt1.z;
	const GLdouble x2 = pt2.x;
	const GLdouble y2 = pt2.y;
	const GLdouble z2 = pt2.z;

    const double x = x2-x1;
	const double y = y2-y1;
	const double z = z2-z1;
    //const double L = std::sqrt(x*x+y*y+z*z);

    glPushMatrix ();

    glTranslated(x1,y1,z1);

	if ((x != 0.) || (y != 0.))
	{
        glRotated(std::atan2(y,x)/RADPERDEG,0.,0.,1.);
        glRotated(std::atan2(std::sqrt(x*x+y*y),z)/RADPERDEG,0.,1.,0.);
    }
	else if (z<0)
        glRotated(180,1.,0.,0.);

	const auto height = sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) +  (pt1.y - pt2.y)*(pt1.y - pt2.y)   + (pt1.z - pt2.z)*(pt1.z - pt2.z) );
    glutSolidCone(ptSize, height, 5, 5);

    glPopMatrix();
}


void RenderHumanBody()
{
    //const auto numberPeople = gDispUnit3d.mPoseKeypoints.getSize(0);
    const auto numberBodyParts = gDispUnit3d.mPoseKeypoints.getSize(1);
	const int person = 0;
    //for(int person=0;person<numberPeople;++person)
    {
		for (auto part = 0 ; part < numberBodyParts ; part++)
        {
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,COLOR_AMBIENT);
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,COLOR_DIFFUSE);
            if(gDispUnit3d.mPoseKeypoints[4*part+2 + person*numberBodyParts] > 0)
            {
                glPushMatrix();
                //glTranslatef(skeleton.m_jointPos[i].first.x,skeleton.m_jointPos[i].first.y,skeleton.m_jointPos[i].first.z);
                glTranslatef(-(gDispUnit3d.mPoseKeypoints[4*part + person*numberBodyParts] -640 )/1280*30,
                             -(gDispUnit3d.mPoseKeypoints[4*part+1+ person*numberBodyParts]-360)/720*30,
                             -(gDispUnit3d.mPoseKeypoints[4*part+2+ person*numberBodyParts]-360)/720*30
                );

				if (part == 4 || part == 7)
                    glutSolidSphere(0.5*0.7,20,20);
                else
                    glutSolidSphere(0.5,20,20);
                //glutSolidSphere(20/WORLD_TO_CM_RATIO,20,20);
                glPopMatrix();
            }
        }

		const int parentIdx[] = { 1, -1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15 };
        //Bone
        //0-1-2 1-3-4-5 0-6-7 6-8-9-10
        //int sizeOfJoints = skeleton.m_boneEndsPos.size();
        //sizeOfJoints /=2;
        for(int part=0;part<numberBodyParts;++part)
        {
			if (parentIdx[part] >= 0)
			{
				if (gDispUnit3d.mPoseKeypoints[4 * part + 3 + person*numberBodyParts]>0)
				{
					cv::Point3f child(
						-(gDispUnit3d.mPoseKeypoints[4 * part + person*numberBodyParts] - 640) / 1280 * 30,
						-(gDispUnit3d.mPoseKeypoints[4 * part + 1 + person*numberBodyParts] - 360) / 720 * 30,
						-(gDispUnit3d.mPoseKeypoints[4 * part + 2 + person*numberBodyParts] - 360) / 720 * 30
					);
					cv::Point3f parent(
						-(gDispUnit3d.mPoseKeypoints[4 * parentIdx[part] + person*numberBodyParts] - 640) / 1280 * 30,
						-(gDispUnit3d.mPoseKeypoints[4 * parentIdx[part] + 1 + person*numberBodyParts] - 360) / 720 * 30,
						-(gDispUnit3d.mPoseKeypoints[4 * parentIdx[part] + 2 + person*numberBodyParts] - 360) / 720 * 30
					);
					DrawConeByTwoPts(parent, child, 0.5);
				}
			}
        }
    }
}


void InitGraphics(void)
{
    // Enable a single OpenGL light
    glLightfv(GL_LIGHT0, GL_AMBIENT, LIGHT_DIFFUSE);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, LIGHT_DIFFUSE);
    glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);

    // Use depth buffering for hidden surface elimination
    glEnable(GL_DEPTH_TEST);

    // Setup the view of the cube
    glMatrixMode(GL_PROJECTION);
    gluPerspective( /* field of view in degree */ 40.0,
        /* aspect ratio */ 1.0,
        /* Z near */ 1.0, /* Z far */ 1000.0);
    glMatrixMode(GL_MODELVIEW);
    gluLookAt(
        0.0, 0.0, 5.0,  // eye is at (0,0,5)
        0.0, 0.0, 0.0,  // center is at (0,0,0)
        0.0, 1.0, 0.  // up is in positive Y direction
    );

    // Adjust cube position to be asthetic angle
    glTranslatef(0.0, 0.0, -1.0);
    glRotatef(60, 1.0, 0.0, 0.0);
    glRotatef(-20, 0.0, 0.0, 1.0);

    glColorMaterial(GL_FRONT, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
}


// this is the actual idle function
void IdleFunc()
{
    glutPostRedisplay();
    glutSwapBuffers();
}


void reshape(GLint width, GLint height)
{
     gWidth = width;
     gHeight = height;
     //printf("Window Reshape: %d, %d\n",width,height);
     glViewport(0, 0, gWidth, gHeight);
     glMatrixMode(GL_PROJECTION);
     glLoadIdentity();
     gRatio = (float)gWidth / gHeight;
     gluPerspective(65.0, gRatio, gNearPlaneForDefaultRender, gFarPlaneForDefaultRender);
     glMatrixMode(GL_MODELVIEW);
}

void RenderDomeFloor()
{
    glDisable(GL_LIGHTING);

	const cv::Point3f gGloorCenter{0,0,0};   //ankle
	const cv::Point3f Noise{0,1,0};

	cv::Point3f upright = Noise - gGloorCenter;
    upright = 1.0/sqrt(upright.x *upright.x +  upright.y *upright.y + upright.z *upright.z )*upright;
	const cv::Point3f gGloorAxis2 = cv::Point3f{1,0,0}.cross(upright);
	const cv::Point3f gGloorAxis1 = gGloorAxis2.cross(upright);

    const auto gridNum = 10;
	const auto width = 50.;//sqrt(Distance(gGloorPts.front(),gGloorCenter)*2 /gridNum) * 1.2;
	const cv::Point3f origin =  gGloorCenter - gGloorAxis1*(width*gridNum/2 ) - gGloorAxis2*(width*gridNum/2);
	const cv::Point3f axis1 =  gGloorAxis1 * width;
	const cv::Point3f axis2 =  gGloorAxis2 * width;
	for (auto y = 0; y <= gridNum; ++y)
	{
		for (auto x = 0; x <= gridNum; ++x)
        {
			if ((x + y) % 2 == 0)
			{
                //continue;
                //if(g_visData.m_backgroundColor.x ==0)
                    glColor4f(0.2f, 0.2f, 0.2f, 1.f); //black
                //else
                //  glColor4f(1.f,1.f,1.f,1.f); //white
			}
            else
            {
                //if(g_visData.m_backgroundColor.x ==0) //black background
                    glColor4f(0.5f, 0.5f, 0.5f, 1.f); //grey
                //else
                //  glColor4f(0.9.f, 0.9.f, 0.9.f, 1); //grey
            }

            const cv::Point3f p1 = origin + axis1*x + axis2*y;
			const cv::Point3f p2 = p1 + axis1;
			const cv::Point3f p3 = p1 + axis2;
			const cv::Point3f p4 = p1 + axis1 + axis2;


            glBegin(GL_QUADS);

//            glNormal3f(0.f, -1.f, 0.f);
            glVertex3f(   p1.x, p1.y,p1.z);
            //glNormal3f(0.f, -1.f, 0.f);
            glVertex3f(   p2.x, p2.y,p2.z);
//            glNormal3f(0.f, -1.f, 0.f);
            glVertex3f(   p4.x, p4.y,p4.z);
//            glNormal3f(0.f, -1.f, 0.f);
            glVertex3f(   p3.x, p3.y,p3.z);
            glEnd();
        }
	}
    glEnable(GL_LIGHTING);
}

void RenderMain(void)
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();
    //gluLookAt(0,0,0, 0, 0, 1, 0, -1, 0);
    gluLookAt(
        0.0, 0.0, 5.0,  // eye is at (0,0,5)
        0.0, 0.0, 0.0,  // center is at (0,0,0)
        0.0, 1.0, 0.  // up is in positive Y direction
    );


    glTranslatef(0,0,gGViewDistance);
    glRotatef(-gMouseYRotate, 1.f, 0.f, 0.f);
    glRotatef(-gMouseXRotate, 0.f, 1.f, 0.f);
    //glRotatef(g_rotateAngle,0,0,1);

    glTranslatef(-gMouseXPan,gMouseYPan,-gMouseZPan);
    //std::cout << gGViewDistance << " "  << gMouseYRotate<< " "  << gMouseXRotate << "\t\t\t";
    //std::cout << gMouseXPan << " "  << gMouseYPan<< " "  << gMouseZPan << std::endl;
    //glTranslatef(gDispUnit3d.m_pt.x,gDispUnit3d.m_pt.y,-gMouseZPan);
    //printf("    render: %f,%f\n",gDispUnit3d.m_pt.x,gDispUnit3d.m_pt.y);

    //////////////////////////////////////////////////////////////////////////
    // Transform to anchor a selected bone as origin
    //glTranslatef(-g_visData.m_anchorOrigin.x,-g_visData.m_
    //glColor3f(1,1,0);
    //glutWireTeapot(1);

    RenderDomeFloor();
	if (gDispUnit3d.mKeypointsValid)
        RenderHumanBody();

    glutSwapBuffers();
}

void MouseButton(int button, int state, int x, int y)
{

	if (button == 3 || button == 4) //mouse wheel
    {
        //printf("wheel:: %d, %d, %d, %d\n",button, state,x,y);
        if(button==3)  //zoom in
            gGViewDistance += 10*gScaleForMouseMotion;
        else  //zoom out
            gGViewDistance -= 10*gScaleForMouseMotion;
        //if(gGViewDistance<0.01)
            //gGViewDistance = 0.01;
        printf("gGViewDistance: %f\n",gGViewDistance);
    }
    else
    {
        if (button == GLUT_LEFT_BUTTON)
        {
            gBButton1Down = (state == GLUT_DOWN) ? 1 : 0;
            gXClick = (float)x;
            gYClick = (float)y;


            //if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
                ///gCameraMode = CameraMode::CAM_ROTATE;
            //else
			if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
                gCameraMode = CameraMode::CAM_PAN;
            else
            {
                gCameraMode = CameraMode::CAM_ROTATE;
            }
        }
        //printf("Clicked: %f,%f\n",gXClick,gYClick);
    }
    glutPostRedisplay();
}

void MouseMotion(int x, int y)
{

    // If button1 pressed, zoom in/out if mouse is moved up/down.
    if (gBButton1Down)
    {
    /*  if(gCameraMode == CameraMode::CAM_ZOOM)
        {
            gGViewDistance += (y - gYClick) * gScaleForMouseMotion;

            printf("gGViewDistance: %f\n",gGViewDistance);
            //if (gCameraWorks.m_currentViewStatus.fViewDistance < VIEWING_DISTANCE_MIN)
             // gCameraWorks.m_currentViewStatus.fViewDistance = VIEWING_DISTANCE_MIN;
        }
        else */
		if (gCameraMode == CameraMode::CAM_ROTATE)
		{
			gMouseXRotate += (x - gXClick)*0.2f;
			gMouseYRotate -= (y - gYClick)*0.2f;
		}
		else if (gCameraMode == CameraMode::CAM_PAN)
		{
			gMouseXPan -= (x - gXClick) / 2 * gScaleForMouseMotion;
			gMouseYPan -= (y - gYClick) / 2 * gScaleForMouseMotion;
		}
		else if (gCameraMode == CameraMode::CAM_PAN_Z)
		{
			auto dist = sqrt(pow((x - gXClick), 2.0f) + pow((y - gYClick), 2.0f));
			if (y<gYClick)
				dist *= -1;
			gMouseZPan -= dist / 5 * gScaleForMouseMotion;
		}

        gXClick = (float)x;
        gYClick = (float)y;

        //printf("%f,%f\n",gXClick,gYClick);
        glutPostRedisplay();
    }
}

const auto GUI_NAME = "OpenPose 3-D Reconstruction";

WRender3D::WRender3D()
{
	cv::imshow(GUI_NAME, cv::Mat{ 500, 500, CV_8UC3, cv::Scalar{ 0,0,0 } });

	//Run OpenGL
	mRenderThread = std::thread{ &WRender3D::visualizationThread, this };
}

void WRender3D::workConsumer(const std::shared_ptr<std::vector<Datum3D>>& datumsPtr)
{
	try
	{
		// Profiling speed
		const auto profilerKey = op::Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);

		// User's displaying/saving/other processing here
		// datum.cvOutputData: rendered frame with pose or heatmaps
		// datum.poseKeypoints: Array<float> with the estimated pose
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			cv::Mat cvMat = datumsPtr->at(0).cvOutputData.clone();
			for (auto i = 1u; i < datumsPtr->size(); i++)
				cv::hconcat(cvMat, datumsPtr->at(i).cvOutputData, cvMat);

			// while (cvMat.cols > 1500 || cvMat.rows > 1500)
			while (cvMat.cols > 1920 || cvMat.rows > 1920)
				// while (cvMat.rows > 3500)
				cv::pyrDown(cvMat, cvMat);

			cv::imshow(GUI_NAME, cvMat);
			cv::resizeWindow(GUI_NAME, cvMat.cols, cvMat.rows);

			// OpenGL Rendering
			gDispUnit3d.mPoseKeypoints = datumsPtr->at(0).poseKeypoints3D;
			gDispUnit3d.mFaceKeypoints = datumsPtr->at(0).faceKeypoints3D;
			gDispUnit3d.mLeftHandKeypoints = datumsPtr->at(0).leftHandKeypoints3D;
			gDispUnit3d.mRightHandKeypoints = datumsPtr->at(0).rightHandKeypoints3D;
			gDispUnit3d.mKeypointsValid = true;

			// Profiling speed
			op::Profiler::timerEnd(profilerKey);
			op::Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 100);
		}

		const auto key = cv::waitKey(1) % 256; // It sleeps 1 ms just to let the user see the output. Change to 33ms for normal 30 fps display
		if (key == 27 || key == 'q')
			this->stop();
	}
	catch (const std::exception& e)
	{
		op::log("Some kind of unexpected error happened.");
		this->stop();
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

void WRender3D::visualizationThread()
{
	char *my_argv[] = { "OpenPose", NULL };
	int my_argc = 1;
	glutInit(&my_argc, my_argv);

	/* setup the size, position, and display mode for new windows */
	glutInitWindowSize(1280, 720);
	// glutInitWindowSize(640,480);
	glutInitWindowPosition(200, 0);
	// glutSetOption(GLUT_MULTISAMPLE,8);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);

	/* create and set up a window */
	glutCreateWindow(GUI_NAME);
	InitGraphics();
	glutDisplayFunc(RenderMain);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
	glutIdleFunc(IdleFunc);
	//glutReshapeFunc (reshape);

	glutMainLoop();
}
