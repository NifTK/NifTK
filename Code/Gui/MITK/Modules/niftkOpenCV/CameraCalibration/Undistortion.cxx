/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "Undistortion.h"
#include <mitkCameraIntrinsicsProperty.h>


namespace niftk
{


const char*    Undistortion::s_CameraCalibrationPropertyName       = "niftk.CameraCalibration";
const char*    Undistortion::s_ImageIsUndistortedPropertyName      = "niftk.ImageIsUndistorted";


//-----------------------------------------------------------------------------
Undistortion::Undistortion(mitk::DataNode::Pointer node)
  : m_Node(node)
{
}


//-----------------------------------------------------------------------------
Undistortion::~Undistortion()
{
}


//-----------------------------------------------------------------------------
void Undistortion::Process(const IplImage* input, IplImage* output, bool recomputeCache)
{
}


//-----------------------------------------------------------------------------
void Undistortion::Run()
{
  mitk::CameraIntrinsics::Pointer   nodeIntrinsic;

  // do we have a node?
  // if so try to find the attached image
  // if not check if we have been created with an image
  if (m_Node.IsNotNull())
  {
    // note that the image data attached to the node overrides whatever previous image we had our hands on
    mitk::BaseData::Pointer data = m_Node->GetData();
    if (data.IsNotNull())
    {
      mitk::Image::Pointer img = dynamic_cast<mitk::Image*>(data.GetPointer());
      if (img.IsNotNull())
      {
        m_Image = img;
      }
      else
      {
        throw std::runtime_error("DataNode has non-image data attached");
      }
    }
    else
    {
      throw std::runtime_error("DataNode does not have any data object attached");
    }

    // getting here... we have a DataNode with an Image attached
    mitk::BaseProperty::Pointer cambp = m_Node->GetProperty(s_CameraCalibrationPropertyName);
    if (cambp.IsNotNull())
    {
      mitk::CameraIntrinsicsProperty::Pointer cam = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cambp.GetPointer());
      // node has a property of the right name but the type is wrong
      if (cam.IsNull())
      {
        throw std::runtime_error("Wrong property type for camera calibration data");
      }
      nodeIntrinsic = cam->GetValue();
    }
  }

  // we have either been created with a node or with an image
  if (m_Image.IsNull())
  {
    throw std::runtime_error("No image data to work on");
  }

  // we may or may not have checked yet whether image has a property
  // even if node has one already we check anyway for sanity-check purposes
  mitk::BaseProperty::Pointer   imgprop = m_Node->GetProperty(s_CameraCalibrationPropertyName);
  if (imgprop.IsNotNull())
  {
    mitk::CameraIntrinsicsProperty::Pointer imgcalib = dynamic_cast<mitk::CameraIntrinsicsProperty*>(imgprop.GetPointer());
    if (imgcalib.IsNotNull())
    {
      // if both node and attached image have calibration props then both need to match!
      // otherwise something in our system causes inconsistent data.
      mitk::CameraIntrinsics::Pointer c = imgcalib->GetValue();
      if (nodeIntrinsic.IsNotNull())
      {
        if (!c->Equals(nodeIntrinsic.GetPointer()))
        {
          throw std::runtime_error("Inconsistent calibration data on DataNode and its Image");
        }
      }
      else
      {
        nodeIntrinsic = c;
      }
    }
    else
    {
      throw std::runtime_error("Wrong property type for camera calibration data");
    }
  }


  if (nodeIntrinsic.IsNull())
  {
    throw std::runtime_error("No camera calibration data found");
  }

  bool  needToRecomputeDistortionCache = true;
  if (m_Intrinsics.IsNotNull())
  {
    needToRecomputeDistortionCache = !m_Intrinsics->Equals(nodeIntrinsic.GetPointer());
    // we need to copy it because somebody could modify that instance of CameraIntrinsics
    m_Intrinsics = nodeIntrinsic->Clone();
  }

}


} // namespace
