#include <stdio.h>
#include <mutex>
#include <openpose3d/renderer.hpp>
#include <GL/glut.h>

struct Keypoints3D
{
    op::Array<float> mPoseKeypoints;
    op::Array<float> mFaceKeypoints;
    op::Array<float> mLeftHandKeypoints;
    op::Array<float> mRightHandKeypoints;
    bool validKeypoints;
    std::mutex mutex;
};

enum class CameraMode {
    CAM_DEFAULT,
    CAM_ROTATE,
    CAM_PAN,
    CAM_PAN_Z
};

Keypoints3D gKeypoints3D;
op::PoseModel sPoseModel = op::PoseModel::COCO_18;

CameraMode gCameraMode = CameraMode::CAM_DEFAULT;

const std::vector<GLfloat> LIGHT_DIFFUSE{ 1.f, 1.f, 1.f, 1.f };  // Diffuse light
const std::vector<GLfloat> LIGHT_POSITION{ 1.f, 1.f, 1.f, 0.f };  // Infinite light location
const std::vector<GLfloat> COLOR_DIFFUSE{ 0.5f, 0.5f, 0.5f, 1.f };
const std::string GUI_NAME{"OpenPose 3-D Reconstruction"};

const auto RAD_TO_DEG = 0.0174532925199433;

//View Change by Mouse
bool gBButton1Down = false;
auto gXClick = 0.f;
auto gYClick = 0.f;
auto gGViewDistance = -250.f; // -82.3994f; //-45;
auto gMouseXRotate = -915.f; // -63.2f; //0;
auto gMouseYRotate = -5.f; // 7.f; //60;
auto gMouseXPan = -70.f; // 0;
auto gMouseYPan = -30.f; // 0;
auto gMouseZPan = 0.f;
auto gScaleForMouseMotion = 0.1f;

void drawConeByTwoPts(const cv::Point3f& pt1, const cv::Point3f& pt2, const float ptSize)
{
    const GLdouble x1 = pt1.x;
    const GLdouble y1 = pt1.y;
    const GLdouble z1 = pt1.z;
    const GLdouble x2 = pt2.x;
    const GLdouble y2 = pt2.y;
    const GLdouble z2 = pt2.z;

    const double x = x2 - x1;
    const double y = y2 - y1;
    const double z = z2 - z1;

    glPushMatrix();

    glTranslated(x1, y1, z1);

    if ((x != 0.) || (y != 0.))
    {
        glRotated(std::atan2(y, x) / RAD_TO_DEG, 0., 0., 1.);
        glRotated(std::atan2(std::sqrt(x*x + y*y), z) / RAD_TO_DEG, 0., 1., 0.);
    }
    else if (z<0)
        glRotated(180, 1., 0., 0.);

    const auto height = std::sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y) + (pt1.z - pt2.z)*(pt1.z - pt2.z));
    glutSolidCone(ptSize, height, 5, 5);

    glPopMatrix();
}

void renderHumanBody(const op::Array<float>& keypoints, const std::vector<unsigned int>& pairs, const std::vector<float> colors, const float ratio)
{
    const auto person = 0;
    const auto numberPeople = keypoints.getSize(0);
    const auto numberBodyParts = keypoints.getSize(1);
    const auto numberColors = colors.size();
    const auto xOffset = -3000; // 640.f;
    const auto yOffset = 360.f;
    const auto zOffset = 1000; // 360.f;
    const auto xScale = 43.f;
    const auto yScale = 24.f;
    const auto zScale = 24.f;

    if (numberPeople > person)
    //for(int person=0;person<numberPeople;++person)
    {
        // Circle for each keypoint
        for (auto part = 0; part < numberBodyParts; part++)
        {
            // Set color
            const auto colorIndex = part * 3;
            const std::vector<float> keypointColor{
                colors[colorIndex % numberColors] / 255.f,
                colors[(colorIndex + 1) % numberColors] / 255.f,
                colors[(colorIndex + 2) % numberColors] / 255.f,
                1.f
            };
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, COLOR_DIFFUSE.data());
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, keypointColor.data());
            // Draw circle
            const auto baseIndex = 4 * part + person*numberBodyParts;
            if (keypoints[baseIndex + 3] > 0)
            {
                cv::Point3f keypoint{
                    -(keypoints[baseIndex] - xOffset) / xScale,
                    -(keypoints[baseIndex + 1] - yOffset) / yScale,
                    (keypoints[baseIndex + 2] - zOffset) / zScale
                };
                // Create and add new sphere
                glPushMatrix();
                glTranslatef(keypoint.x, keypoint.y, keypoint.z);
                // Draw sphere
                glutSolidSphere(0.5 * ratio, 20, 20);
                glPopMatrix();
            }
        }

        // Lines connecting each keypoint pair
        for (auto pair = 0; pair < pairs.size(); pair += 2)
        {
            // Set color
            const auto colorIndex = pairs[pair+1] * 3;
            const std::vector<float> keypointColor{
                colors[colorIndex % numberColors] / 255.f,
                colors[(colorIndex + 1) % numberColors] / 255.f,
                colors[(colorIndex + 2) % numberColors] / 255.f,
                1.f
            };
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, COLOR_DIFFUSE.data());
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, keypointColor.data());
            // Draw line
            const auto baseIndexPairA = 4 * pairs[pair] + person*numberBodyParts;
            const auto baseIndexPairB = 4 * pairs[pair + 1] + person*numberBodyParts;
            if (keypoints[baseIndexPairA + 3] > 0 && keypoints[baseIndexPairB + 3] > 0)
            {
                cv::Point3f pairKeypointA{
                    -(keypoints[baseIndexPairA] - xOffset) / xScale,
                    -(keypoints[baseIndexPairA + 1] - yOffset) / yScale,
                    (keypoints[baseIndexPairA + 2] - zOffset) / zScale
                };
                cv::Point3f pairKeypointB{
                    -(keypoints[baseIndexPairB] - xOffset) / xScale,
                    -(keypoints[baseIndexPairB + 1] - yOffset) / yScale,
                    (keypoints[baseIndexPairB + 2] - zOffset) / zScale
                };
                drawConeByTwoPts(pairKeypointA, pairKeypointB, 0.5f * ratio);
            }
        }
    }
}

void initGraphics(void)
{
    // Enable a single OpenGL light
    glLightfv(GL_LIGHT0, GL_AMBIENT, LIGHT_DIFFUSE.data());
    glLightfv(GL_LIGHT0, GL_DIFFUSE, LIGHT_DIFFUSE.data());
    glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION.data());
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
void idleFunc()
{
    glutPostRedisplay();
    glutSwapBuffers();
}

void renderFloor()
{
    glDisable(GL_LIGHTING);

    const cv::Point3f gGloorCenter{ 0,0,0 };   //ankle
    const cv::Point3f Noise{ 0,1,0 };

    cv::Point3f upright = Noise - gGloorCenter;
    upright = 1.0 / sqrt(upright.x *upright.x + upright.y *upright.y + upright.z *upright.z)*upright;
    const cv::Point3f gGloorAxis2 = cv::Point3f{ 1,0,0 }.cross(upright);
    const cv::Point3f gGloorAxis1 = gGloorAxis2.cross(upright);

    const auto gridNum = 10;
    const auto width = 50.;//sqrt(Distance(gGloorPts.front(),gGloorCenter)*2 /gridNum) * 1.2;
    const cv::Point3f origin = gGloorCenter - gGloorAxis1*(width*gridNum / 2) - gGloorAxis2*(width*gridNum / 2);
    const cv::Point3f axis1 = gGloorAxis1 * width;
    const cv::Point3f axis2 = gGloorAxis2 * width;
    for (auto y = 0; y <= gridNum; ++y)
    {
        for (auto x = 0; x <= gridNum; ++x)
        {
            if ((x + y) % 2 == 0)
                glColor4f(0.2f, 0.2f, 0.2f, 1.f); //black
            else
                glColor4f(0.5f, 0.5f, 0.5f, 1.f); //grey

            const cv::Point3f p1 = origin + axis1*x + axis2*y;
            const cv::Point3f p2 = p1 + axis1;
            const cv::Point3f p3 = p1 + axis2;
            const cv::Point3f p4 = p1 + axis1 + axis2;

            glBegin(GL_QUADS);

            glVertex3f(p1.x, p1.y, p1.z);
            glVertex3f(p2.x, p2.y, p2.z);
            glVertex3f(p4.x, p4.y, p4.z);
            glVertex3f(p3.x, p3.y, p3.z);
            glEnd();
        }
    }
    glEnable(GL_LIGHTING);
}

void renderMain(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();
    //gluLookAt(0,0,0, 0, 0, 1, 0, -1, 0);
    gluLookAt(
        0.0, 0.0, 5.0,  // eye is at (0,0,5)
        0.0, 0.0, 0.0,  // center is at (0,0,0)
        0.0, 1.0, 0.  // up is in positive Y direction
    );

    glTranslatef(0, 0, gGViewDistance);
    glRotatef(-gMouseYRotate, 1.f, 0.f, 0.f);
    glRotatef(-gMouseXRotate, 0.f, 1.f, 0.f);

    glTranslatef(-gMouseXPan, gMouseYPan, -gMouseZPan);

    renderFloor();
    std::unique_lock<std::mutex> lock{gKeypoints3D.mutex};
    if (gKeypoints3D.validKeypoints)
    {
        renderHumanBody(gKeypoints3D.mPoseKeypoints, op::POSE_BODY_PART_PAIRS_RENDER[(int)sPoseModel], op::POSE_COLORS[(int)sPoseModel], 1.f);
        renderHumanBody(gKeypoints3D.mFaceKeypoints, op::FACE_PAIRS_RENDER, op::FACE_COLORS_RENDER, 0.5f);
        renderHumanBody(gKeypoints3D.mLeftHandKeypoints, op::HAND_PAIRS_RENDER, op::HAND_COLORS_RENDER, 0.5f);
        renderHumanBody(gKeypoints3D.mRightHandKeypoints, op::HAND_PAIRS_RENDER, op::HAND_COLORS_RENDER, 0.5f);
    }
    lock.unlock();

    glutSwapBuffers();
}

void mouseButton(const int button, const int state, const int x, const int y)
{

    if (button == 3 || button == 4) //mouse wheel
    {
        if (button == 3)  //zoom in
            gGViewDistance += 10 * gScaleForMouseMotion;
        else  //zoom out
            gGViewDistance -= 10 * gScaleForMouseMotion;
        op::log("gGViewDistance: " + std::to_string(gGViewDistance));
    }
    else
    {
        if (button == GLUT_LEFT_BUTTON)
        {
            gBButton1Down = (state == GLUT_DOWN) ? 1 : 0;
            gXClick = (float)x;
            gYClick = (float)y;

            if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
                gCameraMode = CameraMode::CAM_PAN;
            else
                gCameraMode = CameraMode::CAM_ROTATE;
        }
        op::log("Clicked: [" + std::to_string(gXClick) + "," + std::to_string(gYClick) + "]");
    }
    glutPostRedisplay();
}

void mouseMotion(const int x, const int y)
{

    // If button1 pressed, zoom in/out if mouse is moved up/down.
    if (gBButton1Down)
    {
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
            auto dist = std::sqrt(pow((x - gXClick), 2.0f) + pow((y - gYClick), 2.0f));
            if (y < gYClick)
                dist *= -1;
            gMouseZPan -= dist / 5 * gScaleForMouseMotion;
        }

        gXClick = (float)x;
        gYClick = (float)y;

        glutPostRedisplay();
        op::log("gMouseXRotate = " + std::to_string(gMouseXRotate));
        op::log("gMouseYRotate = " + std::to_string(gMouseYRotate));
        op::log("gMouseXPan = " + std::to_string(gMouseXPan));
        op::log("gMouseYPan = " + std::to_string(gMouseYPan));
        op::log("gMouseZPan = " + std::to_string(gMouseZPan));
    }
}

WRender3D::WRender3D(const op::PoseModel poseModel)
{
    // Update sPoseModel
    sPoseModel = poseModel;
    // Init display
    cv::imshow(GUI_NAME, cv::Mat( 500, 500, CV_8UC3, cv::Scalar{ 0,0,0 } ));
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
            // Display all views
            cv::imshow(GUI_NAME, cvMat);
            cv::resizeWindow(GUI_NAME, cvMat.cols, cvMat.rows);
            // OpenGL Rendering
            std::unique_lock<std::mutex> lock{gKeypoints3D.mutex};
            gKeypoints3D.mPoseKeypoints = datumsPtr->at(0).poseKeypoints3D;
            gKeypoints3D.mFaceKeypoints = datumsPtr->at(0).faceKeypoints3D;
            gKeypoints3D.mLeftHandKeypoints = datumsPtr->at(0).leftHandKeypoints3D;
            gKeypoints3D.mRightHandKeypoints = datumsPtr->at(0).rightHandKeypoints3D;
            gKeypoints3D.validKeypoints = true;
            lock.unlock();
            // Profiling speed
            op::Profiler::timerEnd(profilerKey);
            op::Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 100);
        }
        // Render images
        cv::waitKey(1); // It sleeps 1 ms just to let the user see the output. Change to 33ms for normal 30 fps display if too fast
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
    char *my_argv[] = { NULL };
    int my_argc = 0;
    glutInit(&my_argc, my_argv);

    // setup the size, position, and display mode for new windows
    glutInitWindowSize(1280, 720);
    glutInitWindowPosition(200, 0);
    // glutSetOption(GLUT_MULTISAMPLE,8);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);

    // create and set up a window
    glutCreateWindow(GUI_NAME.c_str());
    initGraphics();
    glutDisplayFunc(renderMain);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);
    glutIdleFunc(idleFunc);

    glutMainLoop();

    this->stop();
}
