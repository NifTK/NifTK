/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMathsUtils_h
#define niftkMathsUtils_h

#include "niftkCommonWin32ExportHeader.h"
#include <vector>
#include <cstdlib>

/**
* \file niftkMathsUtils.h
* \brief A list of general maths functions.
*/
namespace niftk {

/**
* \brief Returns true if fabs(value) is less than a small tolerance, which defaults to 0.000001.
*/
extern "C++" NIFTKCOMMON_WINEXPORT bool IsCloseToZero(const double& value, const double& tolerance = 0.000001);


/**
* \brief Takes a vector of pairs and finds the minimum value in each dimension. Returns the
* minimum values. Optionally returns the indexes of the minium values.
*/
extern "C++" NIFTKCOMMON_WINEXPORT std::pair < double, double > FindMinimumValues
  ( std::vector < std::pair < double, double>  > inputValues,
    std::pair <unsigned int , unsigned int > * indexes = NULL );


/**
* \brief To return the sample mean of a vector.
*/
extern "C++" NIFTKCOMMON_WINEXPORT double Mean(const std::vector<double>& input);

/**
* \brief To return the median of a vector.
*/
extern "C++" NIFTKCOMMON_WINEXPORT double Median(std::vector<double> input);

/**
* \brief To return the sample standard deviation of a vector.
*/
extern "C++" NIFTKCOMMON_WINEXPORT double StdDev(const std::vector<double>& input);


/**
* \brief Assuming input contains squared errors, will sum them, divide by N, and take sqrt for an RMS measure.
*/
extern "C++" NIFTKCOMMON_WINEXPORT double RMS(const std::vector<double>& input);


/**
* \brief Returns -1.0 if value < 0 or 1.0 if value >= 0
*/
extern "C++" NIFTKCOMMON_WINEXPORT double ModifiedSignum(double value);


/**
* \brief Returns 0.0 of value < 0 or sqrt(value) if value >= 0
*/
extern "C++" NIFTKCOMMON_WINEXPORT double SafeSQRT(double value);

/**
 * \brief Calculates the Mahalanobis distance between two vectors. For this implementation the
 * \brief vector elements must be independent so that the covariance matrix is diagonal and can
 * \brief be expressed as vector of the diagonal elements. This is done here to avoid dependencies
 * \brief on a matrix library. For a more general Mahalanobis distance use ITK/VNL.
 * \brief Throws a logic error if vectors are not equal length.
 * \params v1, v2 two vectors to measure the distance between.
 * \params cov the diagonal elements of the covariance matrix
 * \return the Mahalanobis distance between the two vectors
 * */
extern "C++" NIFTKCOMMON_WINEXPORT double MahalanobisDistance (
    const std::vector < double >& v1 , const std::vector < double >& v2,
    const std::vector < double >& covariance );

/**
 * \brief Checks whether first two parameters are equal within tolerance. Throws and exception if not.
 */
extern "C++" NIFTKCOMMON_WINEXPORT void CheckDoublesEquals(double expected, double actual, double tol);

} // end namespace

#endif
