/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkSplitVideo_h
#define mitkSplitVideo_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class SplitVideo
 * \brief Takes an input video file (.avi or .264) together with the corresponding .framemap.log file, a start frame and an end frame. The file is split and the resulting file written out together a
 * suitable .framemap.log file
 */
class NIFTKOPENCV_EXPORT SplitVideo : public itk::Object
{

public:

  mitkClassMacroItkParent(SplitVideo, itk::Object);
  itkNewMacro(SplitVideo);

  bool Split(
      const std::string& inputImageFileName,
      const unsigned int& startFrame,
      const unsigned int& endFrame
      );

protected:

  SplitVideo();
  virtual ~SplitVideo();

  SplitVideo(const SplitVideo&); // Purposefully not implemented.
  SplitVideo& operator=(const SplitVideo&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
