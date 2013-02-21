/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoPointProjectionIntoTwoImages.h"
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
StereoPointProjectionIntoTwoImages::StereoPointProjectionIntoTwoImages()
{

}


//-----------------------------------------------------------------------------
StereoPointProjectionIntoTwoImages::~StereoPointProjectionIntoTwoImages()
{

}


//-----------------------------------------------------------------------------
bool StereoPointProjectionIntoTwoImages::Project(const std::string& input3DFileName,
    const std::string& inputLeftImageName,
    const std::string& inputRightImageName,
    const std::string& outputLeftImageName,
    const std::string& outputRightImageName,
    const std::string& intrinsicLeftFileName,
    const std::string& distortionLeftFileName,
    const std::string& rotationLeftFileName,
    const std::string& translationLeftFileName,
    const std::string& intrinsicRightFileName,
    const std::string& distortionRightFileName,
    const std::string& rightToLeftRotationFileName,
    const std::string& rightToLeftTranslationFileName,
    const std::string& inputLeft2DGoldStandardFileName,
    const std::string& inputRight2DGoldStandardFileName
    )
{
  bool isSuccessful = false;

  try
  {
    std::cout << "Matt, StereoPointProjectionIntoTwoImages::Project" << std::endl;
/*
    IplImage *inputLeftImage = cvLoadImage(inputLeftImageName.c_str());
    if (inputLeftImage == NULL)
    {
      throw std::logic_error("Could not load input left image!");
    }

    IplImage *inputRightImage = cvLoadImage(inputRightImageName.c_str());
    if (inputRightImage == NULL)
    {
      throw std::logic_error("Could not load input right image!");
    }

    CvMat *intrinsicLeft = (CvMat*)cvLoad(intrinsicLeftFileName.c_str());
    if (intrinsicLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera intrinsic params");
    }

    CvMat *distortionLeft = (CvMat*)cvLoad(distortionLeftFileName.c_str());
    if (distortionLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera distortion params");
    }

    CvMat *rotationLeft = (CvMat*)cvLoad(rotationLeftFileName.c_str());
    if (rotationLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera rotation params");
    }

    CvMat *translationLeft = (CvMat*)cvLoad(translationLeftFileName.c_str());
    if (translationLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera translation params");
    }

    CvMat *intrinsicRight = (CvMat*)cvLoad(intrinsicRightFileName.c_str());
    if (intrinsicRight == NULL)
    {
      throw std::logic_error("Failed to load right camera intrinsic params");
    }

    CvMat *distortionRight = (CvMat*)cvLoad(distortionRightFileName.c_str());
    if (distortionRight == NULL)
    {
      throw std::logic_error("Failed to load right camera distortion params");
    }

    CvMat *rightToLeftRotation = (CvMat*)cvLoad(rightToLeftRotationFileName.c_str());
    if (rightToLeftRotation == NULL)
    {
      throw std::logic_error("Failed to load right to left rotation params");
    }

    CvMat *rightToLeftTranslation = (CvMat*)cvLoad(rightToLeftTranslationFileName.c_str());
    if (rightToLeftTranslation == NULL)
    {
      throw std::logic_error("Failed to load right to left translation params");
    }

    IplImage *outputLeftImage = cvCloneImage(inputLeftImage);
    IplImage *outputRightImage = cvCloneImage(inputRightImage);
*/
    isSuccessful = true;
  }
  catch(std::logic_error e)
  {
    std::cerr << "StereoPointProjectionIntoTwoImages::Project: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
