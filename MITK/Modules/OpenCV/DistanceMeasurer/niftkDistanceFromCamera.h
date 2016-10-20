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
#include <mitkImage.h>
#include <mitkDataNode.h>
#include <mitkCommon.h>
#include <mitkCameraIntrinsics.h>
#include <memory>

namespace niftk
{

class DistanceFromCameraPrivate;

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

  double GetDistance(const mitk::DataNode::Pointer& leftImage,
                     const mitk::DataNode::Pointer& rightImage,
                     const mitk::DataNode::Pointer& leftMask = nullptr,
                     const mitk::DataNode::Pointer& rightMask = nullptr
                    );

  double GetDistance(const mitk::Image::Pointer& leftImage,
                     const mitk::Image::Pointer& rightImage,
                     const mitk::CameraIntrinsics::Pointer& leftIntrinsic,
                     const mitk::CameraIntrinsics::Pointer& rightIntrinsic,
                     const itk::Matrix<float, 4, 4>& stereoExtrinsics,
                     const mitk::Image::Pointer& leftMask = nullptr,
                     const mitk::Image::Pointer& rightMask = nullptr
                    );

protected:

  DistanceFromCamera();
  virtual ~DistanceFromCamera();

  DistanceFromCamera(const DistanceFromCamera&); // Purposefully not implemented.
  DistanceFromCamera& operator=(const DistanceFromCamera&); // Purposefully not implemented.

private:

  std::unique_ptr<DistanceFromCameraPrivate> m_Impl;

}; // end class

} // end namespace

#endif
