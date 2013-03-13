/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkRegistrationHelper.h"
#include <cv.h>
#include <highgui.h>

namespace mitk {

//-----------------------------------------------------------------------------
void GenerateEulerRxMatrix(const double& rx, CvMat &matrix3x3)
{
  double cosRx = cos(rx);
  double sinRx = sin(rx);
  cvSetIdentity(&matrix3x3);
  CV_MAT_ELEM(matrix3x3, float, 1, 1) = cosRx;
  CV_MAT_ELEM(matrix3x3, float, 1, 2) = sinRx;
  CV_MAT_ELEM(matrix3x3, float, 2, 1) = -sinRx;
  CV_MAT_ELEM(matrix3x3, float, 2, 2) = cosRx;
}


//-----------------------------------------------------------------------------
void GenerateEulerRyMatrix(const double& ry, CvMat &matrix3x3)
{
  double cosRy = cos(ry);
  double sinRy = sin(ry);
  cvSetIdentity(&matrix3x3);
  CV_MAT_ELEM(matrix3x3, float, 0, 0) = cosRy;
  CV_MAT_ELEM(matrix3x3, float, 0, 2) = -sinRy;
  CV_MAT_ELEM(matrix3x3, float, 2, 0) = sinRy;
  CV_MAT_ELEM(matrix3x3, float, 2, 2) = cosRy;
}


//-----------------------------------------------------------------------------
void GenerateEulerRzMatrix(const double& rz, CvMat &matrix3x3)
{
  double cosRz = cos(rz);
  double sinRz = sin(rz);
  cvSetIdentity(&matrix3x3);
  CV_MAT_ELEM(matrix3x3, float, 0, 0) = cosRz;
  CV_MAT_ELEM(matrix3x3, float, 0, 1) = sinRz;
  CV_MAT_ELEM(matrix3x3, float, 1, 0) = -sinRz;
  CV_MAT_ELEM(matrix3x3, float, 1, 1) = cosRz;
}


//-----------------------------------------------------------------------------
void GenerateEulerRotationMatrix(const double& rx, const double& ry, const double& rz, CvMat &matrix3x3)
{
  CvMat *rotationAboutX = cvCreateMat(3, 3, CV_32FC1);
  CvMat *rotationAboutY = cvCreateMat(3, 3, CV_32FC1);
  CvMat *rotationAboutZ = cvCreateMat(3, 3, CV_32FC1);
  CvMat *rotationRyRx   = cvCreateMat(3, 3, CV_32FC1);

  GenerateEulerRxMatrix(rx, *rotationAboutX);
  GenerateEulerRyMatrix(ry, *rotationAboutY);
  GenerateEulerRzMatrix(rz, *rotationAboutZ);

  cvGEMM(rotationAboutY, rotationAboutX, 1, NULL, 0, rotationRyRx);
  cvGEMM(rotationAboutZ, rotationRyRx, 1, NULL, 0, &matrix3x3);

  cvReleaseMat(&rotationAboutX);
  cvReleaseMat(&rotationAboutY);
  cvReleaseMat(&rotationAboutZ);
  cvReleaseMat(&rotationRyRx);
}


//-----------------------------------------------------------------------------
CvMat* ConvertEulerToRodrigues(
    const double& rx,
    const double& ry,
    const double& rz
    )
{
  CvMat *rotationMatrix = cvCreateMat(3, 3, CV_32FC1);
  CvMat *rotationVector = cvCreateMat(1, 3, CV_32FC1);

  GenerateEulerRotationMatrix(rx, ry, rz, *rotationMatrix);
  cvRodrigues2(rotationMatrix, rotationVector);

  cvReleaseMat(&rotationMatrix);
  return rotationVector;
}


//-----------------------------------------------------------------------------
CvMat* ConvertRodriguesToEuler(
    const CvMat& rotationVector
    )
{
  CvPoint3D64f vec;
  CvMat *result = cvCreateMat(1, 3, CV_32FC1);
  CvMat *rotationMatrix = cvCreateMat(3, 3, CV_32FC1);
  CvMat *r = cvCreateMat(3, 3, CV_32FC1);
  CvMat *q = cvCreateMat(3, 3, CV_32FC1);

  cvRodrigues2(&rotationVector, rotationMatrix);
  cvRQDecomp3x3(rotationMatrix, r, q, NULL, NULL, NULL, &vec);

  CV_MAT_ELEM(*result, float, 0, 0) = vec.x;
  CV_MAT_ELEM(*result, float, 0, 1) = vec.y;
  CV_MAT_ELEM(*result, float, 0, 2) = vec.z;

  cvReleaseMat(&rotationMatrix);
  cvReleaseMat(&r);
  cvReleaseMat(&q);

  return result;
}


//-----------------------------------------------------------------------------
CvMat* Construct4x4TransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    )
{
  CvMat *result = cvCreateMat(4, 4, CV_32FC1);
  CvMat *rotationMatrix = cvCreateMat(3, 3, CV_32FC1);

  GenerateEulerRotationMatrix(rx, ry, rz, *rotationMatrix);

  cvSetIdentity(result);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      CV_MAT_ELEM(*result, float, i, j) = CV_MAT_ELEM(*rotationMatrix, float, i, j);
    }
  }
  CV_MAT_ELEM(*result, float, 0, 3) = tx;
  CV_MAT_ELEM(*result, float, 1, 3) = ty;
  CV_MAT_ELEM(*result, float, 2, 3) = tz;

  cvReleaseMat(&rotationMatrix);

  return result;
}


//-----------------------------------------------------------------------------
CvMat* Construct4x4TransformationMatrixFromDegrees(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    )
{
  const double pi = 3.14159265358979323846;

  double radians[3];
  radians[0] = rx * pi / 180;
  radians[1] = ry * pi / 180;
  radians[2] = rz * pi / 180;

  return Construct4x4TransformationMatrix(radians[0], radians[1], radians[2], tx, ty, tz);
}


//-----------------------------------------------------------------------------
void TransformBy4x4Matrix(
    const CvMat &input3D,
    const CvMat &matrix4x4,
    const bool &isNormals,
    CvMat& output3DPoints
    )
{
  CvMat *rotationMatrix = cvCreateMat(3, 3, CV_32FC1);
  cvSetIdentity(rotationMatrix);

  CvMat *translationMatrix = cvCreateMat(1, 3, CV_32FC1);
  cvSetZero(translationMatrix);

  CvMat *transposedTransformedPoints = cvCreateMat(input3D.cols, input3D.rows, CV_32FC1);

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      CV_MAT_ELEM(*rotationMatrix, float, i, j) = CV_MAT_ELEM(matrix4x4, float, i, j);
    }
  }
  CV_MAT_ELEM(*translationMatrix, float, 0, 0) = CV_MAT_ELEM(matrix4x4, float, 0, 3);
  CV_MAT_ELEM(*translationMatrix, float, 0, 1) = CV_MAT_ELEM(matrix4x4, float, 1, 3);
  CV_MAT_ELEM(*translationMatrix, float, 0, 2) = CV_MAT_ELEM(matrix4x4, float, 2, 3);

  cvGEMM(rotationMatrix, &input3D, 1, NULL, 0, transposedTransformedPoints, CV_GEMM_B_T); // ie. [3x3][Nx3]^T = [3xN].
  cvTranspose(transposedTransformedPoints, &output3DPoints);

  if (!isNormals) // if its a normal, we would subtract the transformed origin, so
  {
    for (int i = 0; i < input3D.rows; i++)
    {
      CV_MAT_ELEM(output3DPoints, float, i, 0) = CV_MAT_ELEM(output3DPoints, float, i, 0) + CV_MAT_ELEM(*translationMatrix, float, 0, 0);
      CV_MAT_ELEM(output3DPoints, float, i, 1) = CV_MAT_ELEM(output3DPoints, float, i, 1) + CV_MAT_ELEM(*translationMatrix, float, 0, 1);
      CV_MAT_ELEM(output3DPoints, float, i, 2) = CV_MAT_ELEM(output3DPoints, float, i, 2) + CV_MAT_ELEM(*translationMatrix, float, 0, 2);
    }
  }

  cvReleaseMat(&rotationMatrix);
  cvReleaseMat(&translationMatrix);
  cvReleaseMat(&transposedTransformedPoints);
}

//-----------------------------------------------------------------------------
} // end namespace
