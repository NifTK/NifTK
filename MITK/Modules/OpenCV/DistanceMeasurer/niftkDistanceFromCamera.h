/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDistanceFromCamera_h
#define niftkDistanceFromCamera_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace niftk
{

/**
 * \class DistanceFromCamera
 * \brief Given 2 images and a camera calibration, will
 * estimate the median distance of SIFT keypoints from the camera.
 */
class NIFTKOPENCV_EXPORT DistanceFromCamera : public itk::Object
{

public:

  mitkClassMacroItkParent(DistanceFromCamera, itk::Object)
  itkNewMacro(DistanceFromCamera)

  double GetDistance();

protected:

  DistanceFromCamera();
  virtual ~DistanceFromCamera();

  DistanceFromCamera(const DistanceFromCamera&); // Purposefully not implemented.
  DistanceFromCamera& operator=(const DistanceFromCamera&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
