/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkRegistrationHelper_h
#define mitkRegistrationHelper_h

#include <cv.h>
#include <cstdlib>
#include <iostream>

/**
 * \file mitkRegistrationHelper
 * \brief Interface to OpenCV functions to help with registration.
 */
namespace mitk {

/**
 * \brief Generates a rotation about X-axis, given a Euler angle in radians.
 * \param rx angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRxMatrix(const double& rx, CvMat &matrix3x3);


/**
 * \brief Generates a rotation about Y-axis, given a Euler angle in radians.
 * \param ry angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRyMatrix(const double& ry, CvMat &matrix3x3);


/**
 * \brief Generates a rotation about Z-axis, given a Euler angle in radians.
 * \param rz angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRzMatrix(const double& rz, CvMat &matrix3x3);


/**
 * \brief Generates a rotation matrix, given Euler angles in radians.
 * \param rx angle in radians
 * \param ry angle in radians
 * \param rz angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRotationMatrix(const double& rx, const double& ry, const double& rz, CvMat &matrix3x3);

/**
 * \brief Converts Euler angles in radians to the Rodrigues rotation vector (axis-angle convention) mentioned in OpenCV.
 * \param rx Euler angle rotation about x-axis in radians
 * \param ry Euler angle rotation about y-axis in radians
 * \param rz Euler angle rotation about z-axis in radians
 * \return A new [1x3] matrix that the caller must then de-allocate.
 */
CvMat* ConvertEulerToRodrigues(
    const double& rx,
    const double& ry,
    const double& rz
    );


/**
 * \brief Converts from the Rodrigues rotation vector mentioned in OpenCV to 3 Euler angles in radians.
 * \param rotationVector a [1x3] Rodrigues rotation vector
 * \return A new [1x3] matrix that the caller must then de-allocate containing rx, ry, rz in radians, as a result of calling OpenCV's RQDecomp3x3.
 */
CvMat* ConvertRodriguesToEuler(
    const CvMat& rotationVector
    );


/**
 * \brief From rotations in radians and translations in millimetres, constructs a 4x4 transformation matrix, using OpenCV conventions.
 * \param rx Euler rotation about x-axis in radians
 * \param ry Euler rotation about y-axis in radians
 * \param rz Euler rotation about z-axis in radians
 * \param tx translation in millimetres along x-axis
 * \param ty translation in millimetres along y-axis
 * \param tz translation in millimetres along z-axis
 * \return a new [4x4] matrix that the caller must then de-allocate.
 */
CvMat* Construct4x4TransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    );


/**
 * \brief From rotations in degrees (+/- 180), converts to radians, then passes on to Construct4x4TransformationMatrix.
 * \return a new [4x4] matrix that the caller must then de-allocate.
 */
CvMat* Construct4x4TransformationMatrixFromDegrees(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    );

/**
 * \brief Transforms a [Nx3] matrix of points (or normals) by a [4x4] matrix.
 * \param input3D [Nx3] matrix of points/normals
 * \param matrix4x4 [4x4] transformation matrix
 * \param isNormals if true, will assume that the input3D are normals, and if false, just 3D points.
 * \param output3DPoints [Nx3] matrix of points as output.
 */
void TransformBy4x4Matrix(
    const CvMat &input3D,
    const CvMat &matrix4x4,
    const bool &isNormals,
    CvMat& output3DPoints
    );

} // end namespace

#endif // MITKREGISTRATIONHELPER_H
