/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __NIFTK_CONVERSIONUTILS_H
#define __NIFTK_CONVERSIONUTILS_H

#include <NifTKConfigure.h>
#include "niftkCommonWin32ExportHeader.h"

#include <string>

#define NIFTK_PI 3.1415926535897932384626433832795

namespace niftk
{

  NIFTKCOMMON_WINEXPORT std::string ConvertToString(int x);
  NIFTKCOMMON_WINEXPORT std::string ConvertToString(unsigned int x);
  NIFTKCOMMON_WINEXPORT std::string ConvertToString(long int x);
  NIFTKCOMMON_WINEXPORT std::string ConvertToString(long unsigned int x);
  NIFTKCOMMON_WINEXPORT std::string ConvertToString(double x);
  NIFTKCOMMON_WINEXPORT std::string ConvertToString(bool x);
  NIFTKCOMMON_WINEXPORT std::string ConvertToString(float x);
  NIFTKCOMMON_WINEXPORT bool ConvertToBool(std::string x);
  NIFTKCOMMON_WINEXPORT int ConvertToInt(std::string x);
  NIFTKCOMMON_WINEXPORT double ConvertToDouble(std::string x);

  /** Rounds to integer. */
  NIFTKCOMMON_WINEXPORT int Round(double d);

  /** Rounds to set number of decimal places. */
  NIFTKCOMMON_WINEXPORT double Round(double x, int decimalPlaces);

  NIFTKCOMMON_WINEXPORT double fixRangeTo1(double i);

  /** Returns the last n characters from n. */
  NIFTKCOMMON_WINEXPORT std::string GetLastNCharacters(std::string s, int n);

  /** For Gaussian blurring and similar. */
  NIFTKCOMMON_WINEXPORT double CalculateVarianceFromFWHM(double fwhm);

  /** For Gaussian blurring and similar. */
  NIFTKCOMMON_WINEXPORT double CalculateStdDevFromFWHM(double fwhm);

  /** Converts first voxel origin to middle of image origin. */
  NIFTKCOMMON_WINEXPORT double ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(
      double millimetreCoordinateOfFirstVoxel,
      int numberOfVoxelsInThatAxis,
      double voxelSpacingInThatAxis);

  /** Converts middle of image origin to first voxel origin. */
  NIFTKCOMMON_WINEXPORT double ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(
      double millimetreCoordinateOfMiddleVoxel,
      int numberOfVoxelsInThatAxis,
      double voxelSpacingInThatAxis);

} // end namespace

#endif
