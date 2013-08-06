/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkRegisterProbeModelToStereoPair_h
#define mitkRegisterProbeModelToStereoPair_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class RegisterProbeModelToStereoPair
 * \brief Takes a VTK probe model, and registers it to a stereo pair of images
 */
class NIFTKOPENCV_EXPORT RegisterProbeModelToStereoPair : public itk::Object
{

public:

  mitkClassMacro(RegisterProbeModelToStereoPair, itk::Object);
  itkNewMacro(RegisterProbeModelToStereoPair);

  bool DoRegistration(
      const std::string& input3DModel,
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
      const float& rx,
      const float& ry,
      const float& rz,
      const float& tx,
      const float& ty,
      const float& tz
      );

protected:

  RegisterProbeModelToStereoPair();
  virtual ~RegisterProbeModelToStereoPair();

  RegisterProbeModelToStereoPair(const RegisterProbeModelToStereoPair&); // Purposefully not implemented.
  RegisterProbeModelToStereoPair& operator=(const RegisterProbeModelToStereoPair&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
