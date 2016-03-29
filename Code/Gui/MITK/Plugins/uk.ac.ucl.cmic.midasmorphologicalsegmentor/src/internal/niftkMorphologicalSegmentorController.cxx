/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorController.h"

#include <niftkMIDASPaintbrushTool.h>

#include "niftkMorphologicalSegmentorView.h"

//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorController::niftkMorphologicalSegmentorController(niftkMorphologicalSegmentorView* segmentorView)
  : niftkBaseSegmentorController(segmentorView),
    m_MorphologicalSegmentorView(segmentorView)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  toolManager->RegisterTool("MIDASPaintbrushTool");

  int paintbrushToolId = toolManager->GetToolIdByToolType<niftk::MIDASPaintbrushTool>();
  niftk::MIDASPaintbrushTool* paintbrushTool = dynamic_cast<niftk::MIDASPaintbrushTool*>(toolManager->GetToolById(paintbrushToolId));
  assert(paintbrushTool);

  paintbrushTool->SegmentationEdited.AddListener(mitk::MessageDelegate1<niftkMorphologicalSegmentorView, int>(m_MorphologicalSegmentorView, &niftkMorphologicalSegmentorView::OnSegmentationEdited));
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorController::~niftkMorphologicalSegmentorController()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  int paintbrushToolId = toolManager->GetToolIdByToolType<niftk::MIDASPaintbrushTool>();
  niftk::MIDASPaintbrushTool* paintbrushTool = dynamic_cast<niftk::MIDASPaintbrushTool*>(toolManager->GetToolById(paintbrushToolId));
  assert(paintbrushTool);

  paintbrushTool->SegmentationEdited.RemoveListener(mitk::MessageDelegate1<niftkMorphologicalSegmentorView, int>(m_MorphologicalSegmentorView, &niftkMorphologicalSegmentorView::OnSegmentationEdited));
}


//-----------------------------------------------------------------------------
bool niftkMorphologicalSegmentorController::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return m_MorphologicalSegmentorView->m_PipelineManager->IsNodeASegmentationImage(node);
}


//-----------------------------------------------------------------------------
bool niftkMorphologicalSegmentorController::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return m_MorphologicalSegmentorView->m_PipelineManager->IsNodeAWorkingImage(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkMorphologicalSegmentorController::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  return m_MorphologicalSegmentorView->m_PipelineManager->GetWorkingDataFromSegmentationNode(node);
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkMorphologicalSegmentorController::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  return m_MorphologicalSegmentorView->m_PipelineManager->GetSegmentationNodeFromWorkingData(node);
}


//-----------------------------------------------------------------------------
bool niftkMorphologicalSegmentorController::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  return m_MorphologicalSegmentorView->m_PipelineManager->CanStartSegmentationForBinaryNode(node);
}


