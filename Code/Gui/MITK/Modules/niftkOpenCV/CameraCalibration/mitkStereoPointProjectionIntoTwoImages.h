/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkStereoPointProjectionIntoTwoImages_h
#define mitkStereoPointProjectionIntoTwoImages_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class StereoPointProjectionIntoTwoImages
 * \brief Takes stereo calibration data, two input images and a bunch of 3D points, and projects them to two 2D images.
 */
class NIFTKOPENCV_EXPORT StereoPointProjectionIntoTwoImages : public itk::Object
{

public:

  mitkClassMacro(StereoPointProjectionIntoTwoImages, itk::Object);
  itkNewMacro(StereoPointProjectionIntoTwoImages);

  bool Project(const std::string& input3DFileName,
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
      );

protected:

  StereoPointProjectionIntoTwoImages();
  virtual ~StereoPointProjectionIntoTwoImages();

  StereoPointProjectionIntoTwoImages(const StereoPointProjectionIntoTwoImages&); // Purposefully not implemented.
  StereoPointProjectionIntoTwoImages& operator=(const StereoPointProjectionIntoTwoImages&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
