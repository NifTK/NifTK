/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkArunLeastSquaresPointRegistration_h
#define niftkArunLeastSquaresPointRegistration_h

#include "niftkPointRegExports.h"

#include <cv.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

namespace niftk {

/**
 * \file niftkArunLeastSquaresPointRegistration.h
 * \brief Performs SVD based registration of two point sets, as in
 * <a href="http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=4767965">Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987, 10.1109/TPAMI.1987.4767965</a>.
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
double PointBasedRegistrationUsingSVD(const std::vector<cv::Point3d>& fixedPoints,
                                      const std::vector<cv::Point3d>& movingPoints,
                                      cv::Matx44d& outputMatrix);

/**
 * @brief Overloaded method for MITK and VTK data types.
 *
 * Calls the above method. Converts (copies) the point sets.
 */
extern "C++" NIFTKPOINTREG_EXPORT
double PointBasedRegistrationUsingSVD(const mitk::PointSet::Pointer& fixedPoints,
                                      const mitk::PointSet::Pointer& movingPoints,
                                      vtkMatrix4x4& matrix);
} // end namespace

#endif
