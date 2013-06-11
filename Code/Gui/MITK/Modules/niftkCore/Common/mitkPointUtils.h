/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_PointUtils_h
#define mitk_PointUtils_h

#include "niftkCoreExports.h"
#include <mitkVector.h>
#include <mitkPositionEvent.h>
#include <mitkPointSet.h>

/**
 * \file mitkPointUtils.h
 * \brief A list of utility methods for working with MIT points and stuff.
 */
namespace mitk {

/**
 * \brief Given a double[3] of x,y,z voxel spacing, calculates a step size along a ray, as 1/3 of the smallest voxel dimension.
 */
NIFTKCORE_EXPORT double CalculateStepSize(double *spacing);

/**
 * \brief Returns true if a and b are different (up to a given tolerance, currently 0.01), and false otherwise.
 */
NIFTKCORE_EXPORT bool AreDifferent(const mitk::Point3D& a, const mitk::Point3D& b);

/**
 * \brief Returns the squared Euclidean distance between a and b.
 */
NIFTKCORE_EXPORT float GetSquaredDistanceBetweenPoints(const mitk::Point3D& a, const mitk::Point3D& b);

/**
 * \brief Returns as output the vector difference of a-b.
 */
NIFTKCORE_EXPORT void GetDifference(const mitk::Point3D& a, const mitk::Point3D& b, mitk::Point3D& output);

/**
 * \brief Given a vector, will calculate the length.
 */
NIFTKCORE_EXPORT double Length(mitk::Point3D& vector);

/**
 * \brief Given a vector, will normalise it to unit length.
 */
NIFTKCORE_EXPORT void Normalise(mitk::Point3D& vector);

/**
 * \brief Copies input to output, i.e. the output is erased, and re-populated.
 */
NIFTKCORE_EXPORT int CopyPointSets(const mitk::PointSet& input, mitk::PointSet& output);

} // end namespace mitk




#endif
