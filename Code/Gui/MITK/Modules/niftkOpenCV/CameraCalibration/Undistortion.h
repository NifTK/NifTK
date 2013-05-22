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


// Idea is to create one of these Undistortion objects for a given DataNode,
// which then does all the magic for computing the undistortion and caches some data.
// Output, however, is not tied into an instance of the class. Instead it's passed into Run().
class NIFTKOPENCV_EXPORT Undistortion
{
public:
  static const char*    s_ImageIsUndistortedPropertyName;       // mitk::BoolProperty
  // FIXME: this one should go to our calibration class/module, not here
  static const char*    s_CameraCalibrationPropertyName;        // mitk::CameraIntrinsicsProperty


public:
  // node should have Image data attached, at least when Run() is called.
  Undistortion(mitk::DataNode::Pointer node);
  virtual ~Undistortion();


public:
  // loads calibration from a text file (not the opencv xml format!).
  // if filename is empty then it will dream up some parameters for the given image.
  static void LoadCalibration(const std::string& filename, mitk::DataNode::Pointer node);
  static void LoadCalibration(const std::string& filename, mitk::Image::Pointer img);

  // FIXME: should undistorting an already undistorted image fail? or silently ignore?
  virtual void Run(mitk::DataNode::Pointer output);


protected:
  // make sure that output node has an image attached with the correct size/etc.
  void PrepareOutput(mitk::DataNode::Pointer output);
  // check that we have an image to work on, it has the correct depth/channels, etc
  void ValidateInput(bool& recomputeCache);

  // FIXME: presumably this is virtual so that we could derive a gpu version.
  //        but then the ipl parameters are no use!
  // throws exceptions if anything is wrong.
  virtual void Process(const IplImage* input, IplImage* output, bool recomputeCache);


protected:
  // the node that this class is to operate on
  mitk::DataNode::Pointer     m_Node;
  // the image attached to our node
  mitk::Image::Pointer        m_Image;
  // the intrinsic parameters belonging to the image
  mitk::CameraIntrinsics::Pointer     m_Intrinsics;

  // cached for repeated use in Process()
  IplImage*        m_MapX;
  IplImage*        m_MapY;
};


} // namespace


#endif // niftk_Undistortion_h
