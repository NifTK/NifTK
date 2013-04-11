/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoHandeyeFromTwoDirectories.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include "FileHelper.h"

namespace mitk {

//-----------------------------------------------------------------------------
StereoHandeyeFromTwoDirectories::StereoHandeyeFromTwoDirectories()
{

}


//-----------------------------------------------------------------------------
StereoHandeyeFromTwoDirectories::~StereoHandeyeFromTwoDirectories()
{

}


//-----------------------------------------------------------------------------
double StereoHandeyeFromTwoDirectories::Calibrate(const std::string& leftDirectoryName,
    const std::string& rightDirectoryName,
    const int& numberCornersX,
    const int& numberCornersY,
    const float& sizeSquareMillimeters,
    const std::string& outputFileName,
    const bool& writeImages
    )
{
  return 0.0;
}

} // end namespace
