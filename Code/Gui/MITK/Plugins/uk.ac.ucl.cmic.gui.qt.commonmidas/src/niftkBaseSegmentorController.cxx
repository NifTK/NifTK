/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorController.h"

#include <mitkDataStorageUtils.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include "niftkBaseSegmentorView.h"

//-----------------------------------------------------------------------------
niftkBaseSegmentorController::niftkBaseSegmentorController(niftkBaseSegmentorView* segmentorView)
  : m_SegmentorView(segmentorView)
{
  // Create an own tool manager and connect it to the data storage straight away.
  m_ToolManager = mitk::ToolManager::New(segmentorView->GetDataStorage());

  this->RegisterTools();
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController::~niftkBaseSegmentorController()
{
}


//-----------------------------------------------------------------------------
mitk::DataStorage* niftkBaseSegmentorController::GetDataStorage() const
{
  return m_SegmentorView->GetDataStorage();
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentorController::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorController::RegisterTools()
{
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkBaseSegmentorController::GetWorkingData()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetWorkingData();
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorController::GetWorkingImageFromToolManager(int index)
{
  mitk::Image* result = nullptr;

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  if (workingData.size() > 0 && index >= 0 && index < (int)workingData.size())
  {
    mitk::DataNode::Pointer node = workingData[index];

    if (node.IsNotNull())
    {
      mitk::Image* image = dynamic_cast<mitk::Image*>( node->GetData() );
      if (image)
      {
        result = image;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorController::GetReferenceNodeFromToolManager()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetReferenceData(0);
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorController::GetReferenceImageFromToolManager()
{
  mitk::Image* result = nullptr;

  mitk::DataNode* node = this->GetReferenceNodeFromToolManager();
  if (node)
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>( node->GetData() );
    if (image)
    {
      result = image;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorController::GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer segmentationNode)
{
  mitk::DataNode* result = mitk::FindFirstParentImage(m_SegmentorView->GetDataStorage(), segmentationNode, false);
  return result;
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorController::GetReferenceImage()
{
  mitk::Image* result = this->GetReferenceImageFromToolManager();
  return result;
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorController::IsNodeAReferenceImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeAGreyScaleImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorController::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorController::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkBaseSegmentorController::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::ToolManager::DataVectorType result(1);
  result[0] = node;
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorController::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::DataNode::Pointer result = node;
  return result;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorController::ApplyDisplayOptions(mitk::DataNode* node)
{
  if (!node) return;

  bool isBinary(false);
  if (node->GetBoolProperty("binary", isBinary) && isBinary)
  {
    node->ReplaceProperty("reslice interpolation", mitk::VtkResliceInterpolationProperty::New(VTK_RESLICE_NEAREST), const_cast<const mitk::BaseRenderer*>((mitk::BaseRenderer*)NULL));
    node->SetBoolProperty("outline binary", true);
    node->SetFloatProperty ("outline width", 1.0);
    node->SetBoolProperty("showVolume", false);
    node->SetBoolProperty("volumerendering", false);
    node->SetOpacity(1.0);
  }
}


//-----------------------------------------------------------------------------
MIDASOrientation niftkBaseSegmentorController::GetOrientationAsEnum()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
  const mitk::SliceNavigationController* sliceNavigationController = m_SegmentorView->GetSliceNavigationController();
  if (sliceNavigationController != NULL)
  {
    mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

    if (viewDirection == mitk::SliceNavigationController::Axial)
    {
      orientation = MIDAS_ORIENTATION_AXIAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Sagittal)
    {
      orientation = MIDAS_ORIENTATION_SAGITTAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Frontal)
    {
      orientation = MIDAS_ORIENTATION_CORONAL;
    }
  }
  return orientation;
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetAxisFromReferenceImage(const MIDASOrientation& orientation)
{
  int axis = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    axis = niftk::GetThroughPlaneAxis(referenceImage, orientation);
  }
  return axis;
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetReferenceImageAxialAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_AXIAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetReferenceImageCoronalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_CORONAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetReferenceImageSagittalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_SAGITTAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetViewAxis()
{
  int axisNumber = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  MIDASOrientation orientation = this->GetOrientationAsEnum();
  if (referenceImage.IsNotNull() && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    axisNumber = niftk::GetThroughPlaneAxis(referenceImage, orientation);
  }
  return axisNumber;
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetUpDirection()
{
  int upDirection = 0;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  MIDASOrientation orientation = this->GetOrientationAsEnum();
  if (referenceImage.IsNotNull() && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    upDirection = niftk::GetUpDirection(referenceImage, orientation);
  }
  return upDirection;
}
