/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-11-23 17:00:47 +0000 (Tue, 23 Nov 2010) $
 Revision          : $Revision: 4215 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string/case_conv.hpp"
#include "boost/math/special_functions/round.hpp"
#include "ConversionUtils.h"
#include "math.h"
#include <algorithm>

namespace niftk
{

std::string ConvertToString(int x)
{
  return boost::lexical_cast<std::string>(x);
}

std::string ConvertToString(unsigned int x)
{
  return boost::lexical_cast<std::string>(x);
}

std::string ConvertToString(long int x)
{
  return boost::lexical_cast<std::string>(x);
}

std::string ConvertToString(long unsigned int x)
{
  return boost::lexical_cast<std::string>(x);
}

std::string ConvertToString(double x)
{
  return boost::lexical_cast<std::string>(x);
}

std::string ConvertToString(bool x)
{
  return boost::lexical_cast<std::string>(x);
}

std::string ConvertToString(float x)
{
  return boost::lexical_cast<std::string>(x);
}

int ConvertToInt(std::string x)
{
  return boost::lexical_cast<int>(x);
}

double ConvertToDouble(std::string x)
{
  return boost::lexical_cast<double>(x);
}

bool ConvertToBool(const std::string x)
{
  if ("true" == x || "True" == x) 
    {
      return true;
    }
  else
    {
      return false;
    }
}

double CalculateVarianceFromFWHM(double fwhm)
{
  // http://mathworld.wolfram.com/GaussianFunction.html
  double variance = ((fwhm * fwhm) / (8.0 * log(2.0)));
  return variance;
}

double CalculateStdDevFromFWHM(double fwhm)
{
  // http://mathworld.wolfram.com/GaussianFunction.html
  double stdDev = fwhm / (2.0 * sqrt(2.0 * log(2.0)));
  return stdDev;
}

double ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(
  double millimetreCoordinateOfFirstVoxel,
  int numberOfVoxelsInThatAxis,
  double voxelSpacingInThatAxis)
{
  return (static_cast<double>(numberOfVoxelsInThatAxis - 1) 
      * voxelSpacingInThatAxis / 2.0) 
    + millimetreCoordinateOfFirstVoxel;
}

double ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(
    double millimetreCoordinateOfMiddleVoxel,
    int numberOfVoxelsInThatAxis,
    double voxelSpacingInThatAxis)
{
  return (static_cast<double>(numberOfVoxelsInThatAxis - 1) 
      * voxelSpacingInThatAxis / -2.0) 
    + millimetreCoordinateOfMiddleVoxel;
}

double fixRangeTo1(double d)
{
  return std::min(std::max(d, -1.0), 1.0);
  
}

int Round(double d)
{
  return boost::math::iround<double>(d);
}

double Round(double d, int numberDecimalPlaces)
{
  double f = pow((double)10, numberDecimalPlaces);
  return ((double)Round(d*f))/f;
}

std::string GetLastNCharacters(std::string s, int n)
{
  int length = s.length();
  if (n >= length)
  {
    return s;
  }
  else
  {
    int start = length - n;

    if (start < 0)
    {
      start = 0;
    }

    return s.substr(start, n);
  }
}

} // end namespace

