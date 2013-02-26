/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKTRACKLAPUS_H
#define MITKTRACKLAPUS_H

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class TrackLapUS
 * \brief Takes an input video file (.avi), and tracks laparoscopic ultrasound, writing to output file (.avi).
 */
class NIFTKOPENCV_EXPORT TrackLapUS : public itk::Object
{

public:

  mitkClassMacro(TrackLapUS, itk::Object);
  itkNewMacro(TrackLapUS);

  bool Track(
      const std::string& inputImageFileName,
      const std::string& inputIntrinsicsFileNameLeft,
      const std::string& inputDistortionCoefficientsFileNameLeft,
      const std::string& inputIntrinsicsFileNameRight,
      const std::string& inputDistortionCoefficientsFileNameRight,
      const std::string& outputImageFileName,
      bool writeInterleaved
      );

protected:

  TrackLapUS();
  virtual ~TrackLapUS();

  TrackLapUS(const TrackLapUS&); // Purposefully not implemented.
  TrackLapUS& operator=(const TrackLapUS&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
