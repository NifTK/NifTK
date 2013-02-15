/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackLapUS.h"
#include "mitkTrackLapUSProcessor.h"
#include <cv.h>
#include <highgui.h>


namespace mitk {

//-----------------------------------------------------------------------------
TrackLapUS::TrackLapUS()
{

}


//-----------------------------------------------------------------------------
TrackLapUS::~TrackLapUS()
{

}


//-----------------------------------------------------------------------------
bool TrackLapUS::Track(
    const std::string& inputImageFileName,
    const std::string& inputIntrinsicsFileNameLeft,
    const std::string& inputDistortionCoefficientsFileNameLeft,
    const std::string& inputIntrinsicsFileNameRight,
    const std::string& inputDistortionCoefficientsFileNameRight,
    const std::string& outputImageFileName,
    bool writeInterleaved
    )
{
  bool isSuccessful = false;

  try
  {
    CvMat *intrinsicLeft = (CvMat*)cvLoad(inputIntrinsicsFileNameLeft.c_str());
    if (intrinsicLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera intrinsic params");
    }

    CvMat *distortionLeft = (CvMat*)cvLoad(inputDistortionCoefficientsFileNameLeft.c_str());
    if (distortionLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera distortion params");
    }

    CvMat *intrinsicRight = (CvMat*)cvLoad(inputIntrinsicsFileNameRight.c_str());
    if (intrinsicRight == NULL)
    {
      throw std::logic_error("Failed to load right camera intrinsic params");
    }

    CvMat *distortionRight = (CvMat*)cvLoad(inputDistortionCoefficientsFileNameRight.c_str());
    if (distortionRight == NULL)
    {
      throw std::logic_error("Failed to load right camera distortion params");
    }

    TrackLapUSProcessor::Pointer processor = TrackLapUSProcessor::New(writeInterleaved, inputImageFileName, outputImageFileName);
    processor->SetMatrices(*intrinsicLeft, *distortionLeft, *intrinsicRight, *distortionRight);
    processor->Initialize();
    processor->Run();

    cvReleaseMat(&intrinsicLeft);
    cvReleaseMat(&distortionLeft);
    cvReleaseMat(&intrinsicRight);
    cvReleaseMat(&intrinsicRight);

    // No exceptions ... so all OK.
    isSuccessful = true;
  }
  catch(std::logic_error e)
  {
    std::cerr << "TrackLapUS::Track: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
