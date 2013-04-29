/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKCORRECTVIDEOFILEDISTORTION_H
#define MITKCORRECTVIDEOFILEDISTORTION_H

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class CorrectVideoFileDistoration
 * \brief Takes an input video file (.avi), and distortion corrects it, writing to output (.avi).
 */
class NIFTKOPENCV_EXPORT CorrectVideoFileDistortion : public itk::Object
{

public:

  mitkClassMacro(CorrectVideoFileDistortion, itk::Object);
  itkNewMacro(CorrectVideoFileDistortion);

  bool Correct(
      const std::string& inputImageFileName,
      const std::string& inputIntrinsicsFileNameLeft,
      const std::string& inputDistortionCoefficientsFileNameLeft,
      const std::string& inputIntrinsicsFileNameRight,
      const std::string& inputDistortionCoefficientsFileNameRight,
      const std::string& outputImageFileName,
      bool writeInterleaved
      );

protected:

  CorrectVideoFileDistortion();
  virtual ~CorrectVideoFileDistortion();

  CorrectVideoFileDistortion(const CorrectVideoFileDistortion&); // Purposefully not implemented.
  CorrectVideoFileDistortion& operator=(const CorrectVideoFileDistortion&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
