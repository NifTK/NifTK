/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCorrectImageDistortion_h
#define mitkCorrectImageDistortion_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class CorrectImageDistoration
 * \brief Takes an input video image (eg. jpg, png), and distortion corrects it, writing to output (.jpg, png).
 */
class NIFTKOPENCV_EXPORT CorrectImageDistortion : public itk::Object
{

public:

  mitkClassMacro(CorrectImageDistortion, itk::Object);
  itkNewMacro(CorrectImageDistortion);

  bool Correct(
      const std::string& inputImageFileName,
      const std::string& inputIntrinsicsFileName,
      const std::string& inputDistortionCoefficientsFileName,
      const std::string& outputImageFileName
      );

protected:

  CorrectImageDistortion();
  virtual ~CorrectImageDistortion();

  CorrectImageDistortion(const CorrectImageDistortion&); // Purposefully not implemented.
  CorrectImageDistortion& operator=(const CorrectImageDistortion&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
