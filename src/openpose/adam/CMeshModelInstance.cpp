#include "CMeshModelInstance.h"
#include <Eigen/Dense>
#include <igl/per_vertex_normals.h>
#include <assert.h>
#include <totalmodel.h>

// Function equivalent and improved from igl::per_vertex_normals
// [270, 452] ms
template <typename T>
inline T getNormTriplet(const T* const ptr)
{
    return std::sqrt(ptr[0]*ptr[0] + ptr[1]*ptr[1] + ptr[2]*ptr[2]);
}
template <typename T>
inline void normalizeTriplet(T* ptr, const T norm)
{
    ptr[0] /= norm;
    ptr[1] /= norm;
    ptr[2] /= norm;
}
template <typename T>
inline void normalizeTriplet(T* ptr)
{
    const auto norm = getNormTriplet(ptr);
    normalizeTriplet(ptr, norm);
}
void per_vertex_normals(
  const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V,
  const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>& F,
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& N
)
{
    Eigen::Matrix<double, Eigen::Dynamic,3, Eigen::RowMajor> FN;
    FN.resize(F.rows(),3);
    auto* FN_data = FN.data();
    const auto* const F_data = F.data();
    const auto* const V_data = V.data();
    // loop over faces
    for (int i = 0; i < F.rows();i++)
    {
        const auto baseIndex = 3*i;
        const auto F_data0 = 3*F_data[baseIndex];
        const auto F_data1 = 3*F_data[baseIndex+1];
        const auto F_data2 = 3*F_data[baseIndex+2];
        const Eigen::Matrix<double, 1, 3> v1(
            V_data[F_data1] - V_data[F_data0],
            V_data[F_data1+1] - V_data[F_data0+1],
            V_data[F_data1+2] - V_data[F_data0+2]);
        const Eigen::Matrix<double, 1, 3> v2(
            V_data[F_data2] - V_data[F_data0],
            V_data[F_data2+1] - V_data[F_data0+1],
            V_data[F_data2+2] - V_data[F_data0+2]);
        FN.row(i) = v1.cross(v2);
        auto* fnRowPtr = &FN_data[baseIndex];
        const double norm = getNormTriplet(fnRowPtr);
        if (norm == 0)
        {
            fnRowPtr[0] = 0;
            fnRowPtr[1] = 0;
            fnRowPtr[2] = 0;
        }
        else
            normalizeTriplet(fnRowPtr, norm);
    }

    // Resize for output
    N.resize(V.rows(),3);
    std::fill(N.data(), N.data() + N.rows() * N.cols(), 0.0);

    Eigen::Matrix<double, Eigen::Dynamic, 1> A(F.rows(), 1);
    auto* A_data = A.data();
    const auto Fcols = F.cols();
    const auto Vcols = V.cols();

  // Projected area helper
  const auto & proj_doublearea =
    [&V_data,&F_data, &Vcols, &Fcols](const int x, const int y, const int f)
    ->double
  {
    const auto baseIndex = f*Fcols;
    const auto baseIndex2 = F_data[baseIndex + 2]*Vcols;
    const auto rx = V_data[F_data[baseIndex]*Vcols + x] - V_data[baseIndex2 + x];
    const auto sx = V_data[F_data[baseIndex + 1]*Vcols + x] - V_data[baseIndex2 + x];
    const auto ry = V_data[F_data[baseIndex]*Vcols + y] - V_data[baseIndex2 + y];
    const auto sy = V_data[F_data[baseIndex + 1]*Vcols + y] - V_data[baseIndex2 + y];
    return rx*sy - ry*sx;
  };

  for (auto f = 0;f<F.rows();f++)
  {
    const auto dblAd1 = proj_doublearea(0,1,f);
    const auto dblAd2 = proj_doublearea(1,2,f);
    const auto dblAd3 = proj_doublearea(2,0,f);
    A_data[f] = std::sqrt(dblAd1*dblAd1 + dblAd2*dblAd2 + dblAd3*dblAd3);
  }

    auto* N_data = N.data();
    // loop over faces
    for (int i = 0 ; i < F.rows();i++)
    {
        const auto baseIndex = i*Fcols;
        // throw normal at each corner
        for (int j = 0; j < 3;j++)
        {
            // auto* nRowPtr = &N_data[3*F(i,j)];
            auto* nRowPtr = &N_data[3*F_data[baseIndex + j]];
            const auto* const fnRowPtr = &FN_data[3*i];
            for (int subIndex = 0; subIndex < FN.cols(); subIndex++)
                nRowPtr[subIndex] += A_data[i] * fnRowPtr[subIndex];
            // Vector equilvanet
            // N.row(F(i,j)) += A_data[i] * FN.row(i);
        }
    }

    // take average via normalization
    // loop over faces
    for (int i = 0;i<N.rows();i++)
        normalizeTriplet(&N_data[3*i]);
    // Matrix equivalent
    // N.rowwise().normalize();
}

void CMeshModelInstance::RecomputeNormal(const TotalModel& model)
{
    // Compute Normal
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_3(m_vertices.size(), 3);
    auto* V_3_data = V_3.data();

    for (int r = 0; r < V_3.rows(); ++r)
    {
        auto* v3rowPtr = &V_3_data[3*r];
        v3rowPtr[0] = m_vertices[r].x; // V_3(r, 0)
        v3rowPtr[1] = m_vertices[r].y; // V_3(r, 1)
        v3rowPtr[2] = m_vertices[r].z; // V_3(r, 2)
    }
    // Eigen::MatrixXd NV;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> NV;

    if (m_meshType==MESH_TYPE_SMPL)
    {
        std::string errorMessage("Not supporting MESH_TYPE_SMPL currently");
        throw std::runtime_error(errorMessage);
        // igl::per_vertex_normals(V_3, g_smpl.faces_, NV);
    }
    if (m_meshType == MESH_TYPE_TOTAL || m_meshType == MESH_TYPE_ADAM)
    {
        // igl::per_vertex_normals(V_3, model.m_faces, NV);
        per_vertex_normals(V_3, model.m_faces, NV);
        // Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> NVAux;
        // igl::per_vertex_normals(V_3, model.m_faces, NVAux);
        // std::cout << (NV - NVAux).norm() << std::endl;
        // assert((NV - NVAux).norm() < 1e-6);
    }
    m_normals.resize(NV.rows());
    auto* NV_data = NV.data();
    for (int r = 0; r < NV.rows(); ++r)
    {
        const auto* const nvRow = &NV_data[3*r];
        m_normals[r] = cv::Point3f(nvRow[0], nvRow[1], nvRow[2]); // cv::Point3f(NV(r, 0), NV(r, 1), NV(r, 2))
    }
}
