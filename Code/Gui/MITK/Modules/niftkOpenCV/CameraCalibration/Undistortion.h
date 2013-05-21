/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftk_Undistortion_h
#define niftk_Undistortion_h

#include "niftkOpenCVExports.h"
#include <mitkDataStorage.h>
#include <mitkCameraIntrinsics.h>
#include <opencv2/core/types_c.h>


namespace niftk
{


// caches distortion maps and node parameters
class NIFTKOPENCV_EXPORT Undistortion
{
public:
  static const char*    s_ImageIsUndistortedPropertyName;       // mitk::BoolProperty
  // FIXME: this one should go to our calibration class/module, not here
  static const char*    s_CameraCalibrationPropertyName;        // mitk::CameraIntrinsicsProperty


public:
  // passed-in node has to have right property
  Undistortion(mitk::DataNode::Pointer node);
  virtual ~Undistortion();


public:
  static void LoadCalibration(const std::string& filename, mitk::DataNode::Pointer node);
  static void LoadCalibration(const std::string& filename, mitk::Image::Pointer img);

  // FIXME: output node as parameter?
  virtual void Run(mitk::DataNode::Pointer output);


protected:
  void PrepareOutput(mitk::DataNode::Pointer output);
  void ValidateInput(bool& recomputeCache);

  virtual void Process(const IplImage* input, IplImage* output, bool recomputeCache);


protected:
  // the node that this class is to operate on
  mitk::DataNode::Pointer     m_Node;
  // the image attached to our node
  mitk::Image::Pointer        m_Image;
  // the intrinsic parameters belonging to the image
  mitk::CameraIntrinsics::Pointer     m_Intrinsics;
};


} // namespace


#endif // niftk_Undistortion_h
