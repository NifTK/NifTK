/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkCorrectImageDistortion.h"
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
CorrectImageDistortion::CorrectImageDistortion()
{

}


//-----------------------------------------------------------------------------
CorrectImageDistortion::~CorrectImageDistortion()
{

}


//-----------------------------------------------------------------------------
bool CorrectImageDistortion::Correct(
    const std::string& inputImageFileName,
    const std::string& inputIntrinsicsFileName,
    const std::string& inputDistortionCoefficientsFileName,
    const std::string& outputImageFileName
    )
{
  bool isSuccessful = false;

  try
  {
    CorrectDistortionInImageFile(
        inputImageFileName,
        inputIntrinsicsFileName,
        inputDistortionCoefficientsFileName,
        outputImageFileName);

    // No exceptions ... so all OK.
    isSuccessful = true;
  }
  catch(std::logic_error e)
  {
    std::cerr << "CorrectVideoFileDistortion::Correct: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
