/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorController.h"

#include <mitkDataStorageUtils.h>

#include <niftkGeneralSegmentorGUI.h>

#include "niftkGeneralSegmentorView.h"

//-----------------------------------------------------------------------------
niftkGeneralSegmentorController::niftkGeneralSegmentorController(niftkGeneralSegmentorView* segmentorView)
  : niftkBaseSegmentorController(segmentorView),
    m_GeneralSegmentorView(segmentorView)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  toolManager->RegisterTool("MIDASDrawTool");
  toolManager->RegisterTool("MIDASSeedTool");
  toolManager->RegisterTool("MIDASPolyTool");
  toolManager->RegisterTool("MIDASPosnTool");
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorController::~niftkGeneralSegmentorController()
{
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  assert(node);
  bool result = false;

  if (mitk::IsNodeABinaryImage(node))
  {

    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      mitk::DataStorage* dataStorage = this->GetDataStorage();
      mitk::DataNode::Pointer seedsNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::SEEDS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer currentContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer drawContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::DRAW_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seePriorContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::PRIOR_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seeNextContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::NEXT_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer regionGrowingImageNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::REGION_GROWING_NAME.c_str(), node, true);

      if (seedsNode.IsNotNull()
          && currentContoursNode.IsNotNull()
          && drawContoursNode.IsNotNull()
          && seePriorContoursNode.IsNotNull()
          && seeNextContoursNode.IsNotNull()
          && regionGrowingImageNode.IsNotNull()
          )
      {
        result = true;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkGeneralSegmentorController::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  assert(node);
  mitk::ToolManager::DataVectorType result;

  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      mitk::DataStorage* dataStorage = this->GetDataStorage();
      mitk::DataNode::Pointer seedsNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::SEEDS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer currentContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer drawContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::DRAW_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seePriorContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::PRIOR_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seeNextContoursNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::NEXT_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer regionGrowingImageNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::REGION_GROWING_NAME.c_str(), node, true);
      mitk::DataNode::Pointer initialSegmentationImageNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::INITIAL_SEGMENTATION_NAME.c_str(), node, true);
      mitk::DataNode::Pointer initialSeedsNode = dataStorage->GetNamedDerivedNode(niftk::MIDASTool::INITIAL_SEEDS_NAME.c_str(), node, true);

      if (seedsNode.IsNotNull()
          && currentContoursNode.IsNotNull()
          && drawContoursNode.IsNotNull()
          && seePriorContoursNode.IsNotNull()
          && seeNextContoursNode.IsNotNull()
          && regionGrowingImageNode.IsNotNull()
          && initialSegmentationImageNode.IsNotNull()
          && initialSeedsNode.IsNotNull()
          )
      {
        // The order of this list must match the order they were created in.
        result.push_back(node);
        result.push_back(seedsNode);
        result.push_back(currentContoursNode);
        result.push_back(drawContoursNode);
        result.push_back(seePriorContoursNode);
        result.push_back(seeNextContoursNode);
        result.push_back(regionGrowingImageNode);
        result.push_back(initialSegmentationImageNode);
        result.push_back(initialSeedsNode);
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  bool canRestart = false;

  if (node.IsNotNull() && mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);
    if (parent.IsNotNull())
    {
      if (mitk::IsNodeAGreyScaleImage(parent))
      {
        canRestart = true;
      }
    }
  }

  return canRestart;
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorGUI* niftkGeneralSegmentorController::CreateSegmentorGUI(QWidget *parent)
{
  m_GeneralSegmentorGUI = new niftkGeneralSegmentorGUI(parent);
  m_GeneralSegmentorView->m_GeneralSegmentorGUI = m_GeneralSegmentorGUI;

  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(CleanButtonClicked()), SLOT(OnCleanButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(WipeButtonClicked()), SLOT(OnWipeButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(WipePlusButtonClicked()), SLOT(OnWipePlusButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(WipeMinusButtonClicked()), SLOT(OnWipeMinusButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(PropagateUpButtonClicked()), SLOT(OnPropagateUpButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(PropagateDownButtonClicked()), SLOT(OnPropagateDownButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(Propagate3DButtonClicked()), SLOT(OnPropagate3DButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(OKButtonClicked()), SLOT(OnOKButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(CancelButtonClicked()), SLOT(OnCancelButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(RestartButtonClicked()), SLOT(OnRestartButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(ResetButtonClicked()), SLOT(OnResetButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(ThresholdApplyButtonClicked()), SLOT(OnThresholdApplyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(ThresholdingCheckBoxToggled(bool)), SLOT(OnThresholdingCheckBoxToggled(bool)));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(SeePriorCheckBoxToggled(bool)), SLOT(OnSeePriorCheckBoxToggled(bool)));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(SeeNextCheckBoxToggled(bool)), SLOT(OnSeeNextCheckBoxToggled(bool)));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(ThresholdValueChanged()), SLOT(OnThresholdValueChanged()));

  /// Transfer the focus back to the main window if any button is pressed.
  /// This is needed so that the key interactions (like 'a'/'z' for changing slice) keep working.
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(NewSegmentationButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(CleanButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(WipeButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(WipePlusButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(WipeMinusButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(PropagateUpButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(PropagateDownButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(Propagate3DButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(OKButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(CancelButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(RestartButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(ResetButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(ThresholdApplyButtonClicked()), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(ThresholdingCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(SeePriorCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
  m_GeneralSegmentorView->connect(m_GeneralSegmentorGUI, SIGNAL(SeeNextCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));

  return m_GeneralSegmentorGUI;
}
