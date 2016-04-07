/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLiuLeastSquaresWithNormalsRegistration_h
#define niftkLiuLeastSquaresWithNormalsRegistration_h

#include <niftkPointRegExports.h>

#include <cv.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

namespace niftk {

/**
* \file niftkLuiLeastSquaresWithNormalsRegistration.h
* \brief Performs SVD based registration of two point sets with surface normals, as in
* <a href="http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=758228">
* Marker orientation in fiducial registration, Liu, Fitzpatrick, 2003, 10.1117/12.480860
* </a>.
*
* IMPORTANT: Must throw mitk::Exception or subclasses for all errors.
*/

/**
* @brief Does Point Based Registration of two same sized, corresponding point sets.
* @param fixedPoints fixed point set.
* @param movingPoints moving point set.
* @param outputMatrix output 4 x 4 homogeneous rigid body transformation.
* @return fiducial registration error
*/
extern "C++" NIFTKPOINTREG_EXPORT
double PointAndNormalBasedRegistrationUsingSVD(const std::vector<cv::Point3d>& fixedPoints,
                                               const std::vector<cv::Point3d>& fixedNormals,
                                               const std::vector<cv::Point3d>& movingPoints,
                                               const std::vector<cv::Point3d>& movingNormals,
                                               cv::Matx44d& outputMatrix);


/**
* @brief Overloaded method for MITK and VTK data types.
*
* Calls the above method. Converts (copies) the point sets from
* the mitk::PointSet into a vector of cv::Point3D in order.
* The PointID in mitk::PointSet is not used, so the points
* must be in the right order, and corresponding.
*/
extern "C++" NIFTKPOINTREG_EXPORT
double PointAndNormalBasedRegistrationUsingSVD(const mitk::PointSet::Pointer fixedPoints,
                                               const mitk::PointSet::Pointer fixedNormals,
                                               const mitk::PointSet::Pointer movingPoints,
                                               const mitk::PointSet::Pointer movingNormals,
                                               vtkMatrix4x4& matrix);

} // end namespace

#endif
