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

} // end namespace

#endif
