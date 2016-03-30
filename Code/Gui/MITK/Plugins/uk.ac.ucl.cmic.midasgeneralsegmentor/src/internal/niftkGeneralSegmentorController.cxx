/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorController.h"

#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkPointSet.h>

#include <mitkDataStorageUtils.h>

#include <niftkGeneralSegmentorUtils.h>
#include <niftkMIDASDrawTool.h>
#include <niftkMIDASPolyTool.h>
#include <niftkMIDASSeedTool.h>

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


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::HasInitialisedWorkingData()
{
  bool result = false;

  mitk::ToolManager::DataVectorType nodes = this->GetWorkingData();
  if (nodes.size() > 0)
  {
    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::StoreInitialSegmentation()
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::ToolManager::DataVectorType workingData = toolManager->GetWorkingData();

  mitk::DataNode* segmentationNode = workingData[niftk::MIDASTool::SEGMENTATION];
  mitk::DataNode* seedsNode = workingData[niftk::MIDASTool::SEEDS];
  mitk::DataNode* initialSegmentationNode = workingData[niftk::MIDASTool::INITIAL_SEGMENTATION];
  mitk::DataNode* initialSeedsNode = workingData[niftk::MIDASTool::INITIAL_SEEDS];

  initialSegmentationNode->SetData(dynamic_cast<mitk::Image*>(segmentationNode->GetData())->Clone());
  initialSeedsNode->SetData(dynamic_cast<mitk::PointSet*>(seedsNode->GetData())->Clone());
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer niftkGeneralSegmentorController::CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::Tool* drawTool = toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>());
  assert(drawTool);

  mitk::ColorProperty::Pointer col = mitk::ColorProperty::New(r, g, b);

  mitk::DataNode::Pointer helperImageNode = drawTool->CreateEmptySegmentationNode( referenceImage, name, col->GetColor());
  helperImageNode->SetColor(col->GetColor());
  helperImageNode->SetProperty("binaryimage.selectedcolor", col);
  helperImageNode->SetBoolProperty("helper object", true);
  helperImageNode->SetBoolProperty("visible", visible);
  helperImageNode->SetProperty("layer", mitk::IntProperty::New(layer));

  this->ApplyDisplayOptions(helperImageNode);

  return helperImageNode;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer niftkGeneralSegmentorController::CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  mitk::ContourModelSet::Pointer contourSet = mitk::ContourModelSet::New();

  mitk::DataNode::Pointer contourSetNode = mitk::DataNode::New();

  contourSetNode->SetProperty("color", mitk::ColorProperty::New(r, g, b));
  contourSetNode->SetProperty("contour.color", mitk::ColorProperty::New(r, g, b));
  contourSetNode->SetFloatProperty("opacity", 1.0f);
  contourSetNode->SetProperty("name", mitk::StringProperty::New(name));
  contourSetNode->SetBoolProperty("helper object", true);
  contourSetNode->SetBoolProperty("visible", visible);
  contourSetNode->SetProperty("layer", mitk::IntProperty::New(layer));
  contourSetNode->SetData(contourSet);

  return contourSetNode;
}


//-----------------------------------------------------------------------------
mitk::PointSet* niftkGeneralSegmentorController::GetSeeds()
{
  mitk::PointSet* result = nullptr;

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode* seedsNode = toolManager->GetWorkingData(niftk::MIDASTool::SEEDS);
  if (seedsNode)
  {
    result = dynamic_cast<mitk::PointSet*>(seedsNode->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::InitialiseSeedsForWholeVolume()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  MIDASOrientation orientation = this->GetOrientationAsEnum();
  if (orientation == MIDAS_ORIENTATION_UNKNOWN)
  {
    orientation = MIDAS_ORIENTATION_CORONAL;
  }

  int axis = this->GetAxisFromReferenceImage(orientation);
  if (axis == -1)
  {
    axis = 0;
  }

  mitk::PointSet *seeds = this->GetSeeds();
  assert(seeds);

  mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
  assert(workingImage);

  try
  {
    AccessFixedDimensionByItk_n(workingImage,
        niftk::ITKInitialiseSeedsForVolume, 3,
        (*seeds,
         axis
        )
      );
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during niftk::ITKInitialiseSeedsForVolume, so have not initialised seeds correctly, caused by:" << e.what();
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::RecalculateMinAndMaxOfImage()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    double min = referenceImage->GetStatistics()->GetScalarValueMinNoRecompute();
    double max = referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute();
    m_GeneralSegmentorGUI->SetLowerAndUpperIntensityRanges(min, max);
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::RecalculateMinAndMaxOfSeedValues()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  mitk::PointSet::Pointer seeds = this->GetSeeds();

  if (referenceImage.IsNotNull() && seeds.IsNotNull())
  {
    double min = 0;
    double max = 0;

    int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
    int axisNumber = this->GetViewAxis();

    if (sliceNumber != -1 && axisNumber != -1)
    {
      try
      {
        AccessFixedDimensionByItk_n(referenceImage, niftk::ITKRecalculateMinAndMaxOfSeedValues, 3, (*(seeds.GetPointer()), axisNumber, sliceNumber, min, max));
        m_GeneralSegmentorGUI->SetSeedMinAndMaxValues(min, max);
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception, so abandoning recalculating min and max of seeds values, due to:" << e.what();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::UpdateCurrentSliceContours(bool updateRendering)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  int axisNumber = this->GetViewAxis();

  mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
  assert(workingImage);

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[niftk::MIDASTool::CONTOURS]->GetData());

  // TODO
  // This assertion fails sometimes if both the morphological and irregular (this) volume editor is
  // switched on and you are using the paintbrush tool of the morpho editor.
//  assert(contourSet);

  if (contourSet)
  {
    if (sliceNumber >= 0 && axisNumber >= 0)
    {
      niftk::GenerateOutlineFromBinaryImage(workingImage, axisNumber, sliceNumber, sliceNumber, contourSet);

      if (contourSet->GetSize() > 0)
      {
        workingData[niftk::MIDASTool::CONTOURS]->Modified();

        if (updateRendering)
        {
          this->RequestRenderWindowUpdate();
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::OnSeePriorCheckBoxToggled(bool checked)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();

  if (checked)
  {
    this->UpdatePriorAndNext();
  }
  workingData[niftk::MIDASTool::PRIOR_CONTOURS]->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::OnSeeNextCheckBoxToggled(bool checked)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();

  if (checked)
  {
    this->UpdatePriorAndNext();
  }
  workingData[niftk::MIDASTool::NEXT_CONTOURS]->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::UpdatePriorAndNext(bool updateRendering)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  int axisNumber = this->GetViewAxis();

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(0);

  if (m_GeneralSegmentorGUI->IsSeePriorCheckBoxChecked())
  {
    mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[niftk::MIDASTool::PRIOR_CONTOURS]->GetData());
    niftk::GenerateOutlineFromBinaryImage(segmentationImage, axisNumber, sliceNumber-1, sliceNumber, contourSet);

    if (contourSet->GetSize() > 0)
    {
      workingData[niftk::MIDASTool::PRIOR_CONTOURS]->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }

  if (m_GeneralSegmentorGUI->IsSeeNextCheckBoxChecked())
  {
    mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[niftk::MIDASTool::NEXT_CONTOURS]->GetData());
    niftk::GenerateOutlineFromBinaryImage(segmentationImage, axisNumber, sliceNumber+1, sliceNumber, contourSet);

    if (contourSet->GetSize() > 0)
    {
      workingData[niftk::MIDASTool::NEXT_CONTOURS]->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceNumber)
{
  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  return this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceNumber, *seeds);
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceNumber, mitk::PointSet& seeds)
{
  bool sliceDoesHaveUnenclosedSeeds = false;

  if (!this->HasInitialisedWorkingData())
  {
    return sliceDoesHaveUnenclosedSeeds;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(0);

  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  niftk::MIDASPolyTool *polyTool = static_cast<niftk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>()));
  assert(polyTool);

  mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();
  mitk::ContourModel* polyToolContour = polyTool->GetContour();
  if (polyToolContour != NULL && polyToolContour->GetNumberOfVertices() >= 2)
  {
    polyToolContours->AddContourModel(polyToolContour);
  }

  mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::CONTOURS]->GetData());
  mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::DRAW_CONTOURS]->GetData());

  double lowerThreshold = m_GeneralSegmentorGUI->GetLowerThreshold();
  double upperThreshold = m_GeneralSegmentorGUI->GetUpperThreshold();

  int axisNumber = this->GetViewAxis();

  if (axisNumber != -1 && sliceNumber != -1)
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
        niftk::ITKSliceDoesHaveUnEnclosedSeeds, 3,
          (seeds,
           *segmentationContours,
           *polyToolContours,
           *drawToolContours,
           *segmentationImage,
            lowerThreshold,
            upperThreshold,
            thresholdOn,
            axisNumber,
            sliceNumber,
            sliceDoesHaveUnenclosedSeeds
          )
      );
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception during niftk::ITKSliceDoesHaveUnEnclosedSeeds, so will return false, caused by:" << e.what();
    }
  }

  return sliceDoesHaveUnenclosedSeeds;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::FilterSeedsToCurrentSlice(
    mitk::PointSet& inputPoints,
    int& axisNumber,
    int& sliceNumber,
    mitk::PointSet& outputPoints
    )
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage,
          niftk::ITKFilterSeedsToCurrentSlice, 3,
          (inputPoints,
           axisNumber,
           sliceNumber,
           outputPoints
          )
        );
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning FilterSeedsToCurrentSlice, caused by:" << e.what();
    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::FilterSeedsToEnclosedSeedsOnCurrentSlice(
    mitk::PointSet& inputPoints,
    bool& thresholdOn,
    int& sliceNumber,
    mitk::PointSet& outputPoints
    )
{
  outputPoints.Clear();

  mitk::PointSet::Pointer singleSeedPointSet = mitk::PointSet::New();

  mitk::PointSet::PointsConstIterator inputPointsIt = inputPoints.Begin();
  mitk::PointSet::PointsConstIterator inputPointsEnd = inputPoints.End();
  for ( ; inputPointsIt != inputPointsEnd; ++inputPointsIt)
  {
    mitk::PointSet::PointType point = inputPointsIt->Value();
    mitk::PointSet::PointIdentifier pointID = inputPointsIt->Index();

    singleSeedPointSet->Clear();
    singleSeedPointSet->InsertPoint(0, point);

    bool unenclosed = this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceNumber, *(singleSeedPointSet.GetPointer()));

    if (!unenclosed)
    {
      outputPoints.InsertPoint(pointID, point);
    }
  }
}


/**************************************************************
 * Start of: Functions for simply tool toggling
 *************************************************************/

//-----------------------------------------------------------------------------
void niftkGeneralSegmentorController::ToggleTool(int toolId)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  int activeToolId = toolManager->GetActiveToolID();

  if (toolId == activeToolId)
  {
    toolManager->ActivateTool(-1);
  }
  else
  {
    toolManager->ActivateTool(toolId);
  }
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::SelectSeedTool()
{
  /// Note:
  /// If the tool selection box is disabled then the tools are not registered to
  /// the tool manager ( RegisterClient() ). Then if you activate a tool and another
  /// tool was already active, then its interaction event observer service tries to
  /// be unregistered. But since the tools was not registered into the tool manager,
  /// the observer service is still null, and the attempt to unregister it causes crash.
  ///
  /// Consequence:
  /// We should not do anything with the tools until they are registered to the
  /// tool manager.

  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int seedToolId = toolManager->GetToolIdByToolType<niftk::MIDASSeedTool>();

    if (seedToolId != activeToolId)
    {
      toolManager->ActivateTool(seedToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::SelectDrawTool()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int drawToolId = toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>();

    if (drawToolId != activeToolId)
    {
      toolManager->ActivateTool(drawToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::SelectPolyTool()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int polyToolId = toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>();

    if (polyToolId != activeToolId)
    {
      toolManager->ActivateTool(polyToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::UnselectTools()
{
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();

    if (toolManager->GetActiveToolID() != -1)
    {
      toolManager->ActivateTool(-1);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::SelectViewMode()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    if (!this->HasInitialisedWorkingData())
    {
      QList<mitk::DataNode::Pointer> selectedNodes = this->GetDataManagerSelection();
      foreach (mitk::DataNode::Pointer selectedNode, selectedNodes)
      {
        selectedNode->SetVisibility(!selectedNode->IsVisible(0));
      }
      this->RequestRenderWindowUpdate();

      return true;
    }

    mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
    bool segmentationNodeIsVisible = workingData[niftk::MIDASTool::SEGMENTATION]->IsVisible(0);
    workingData[niftk::MIDASTool::SEGMENTATION]->SetVisibility(!segmentationNodeIsVisible);
    this->RequestRenderWindowUpdate();

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorController::CleanSlice()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    /// TODO Temporarily switched off during refactoring, it has to be switched back on
    /// after dependent functions are also moved.
//    this->OnCleanButtonClicked();
    return true;
  }

  return false;
}


/**************************************************************
 * End of: Functions for simply tool toggling
 *************************************************************/
