/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKPOINTUTILS_H
#define MITKPOINTUTILS_H

#include "niftkMitkExtExports.h"
#include "mitkVector.h"
#include "mitkPositionEvent.h"

/**
 * \file mitkPointUtils.h
 * \brief A list of utility methods for working with MIT points and stuff.
 */
namespace mitk {

/**
 * \brief Given a double[3] of x,y,z voxel spacing, calculates a step size along a ray, as 1/3 of the smallest voxel dimension.
 */
NIFTKMITKEXT_EXPORT double CalculateStepSize(double *spacing);

/**
 * \brief Returns true if a and b are different (up to a given tolerance, currently 0.01), and false otherwise.
 */
NIFTKMITKEXT_EXPORT bool AreDifferent(const mitk::Point3D& a, const mitk::Point3D& b);

/**
 * \brief Returns the squared Euclidean distance between a and b.
 */
NIFTKMITKEXT_EXPORT float GetSquaredDistanceBetweenPoints(const mitk::Point3D& a, const mitk::Point3D& b);

/**
 * \brief Returns as output the vector difference of a-b.
 */
NIFTKMITKEXT_EXPORT void GetDifference(const mitk::Point3D& a, const mitk::Point3D& b, mitk::Point3D& output);

/**
 * \brief Given a vector, will calculate the length.
 */
NIFTKMITKEXT_EXPORT double Length(mitk::Point3D& vector);

/**
 * \brief Given a vector, will normalise it to unit length.
 */
NIFTKMITKEXT_EXPORT void Normalise(mitk::Point3D& vector);

} // end namespace mitk




#endif
