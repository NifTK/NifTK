/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorController.h"

#include <QMessageBox>

#include <mitkAnnotationProperty.h>
#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkITKImageImport.h>
#include <mitkOperationEvent.h>
#include <mitkPointSet.h>
#include <mitkUndoController.h>

#include <QmitkRenderWindow.h>

#include <itkIsImageBinary.h>
#include <mitkImageAccessByItk.h>

#include <niftkDataStorageUtils.h>
#include <niftkIBaseView.h>
#include <niftkGeneralSegmentorUtils.h>
#include <niftkDrawTool.h>
#include <niftkSeedTool.h>
#include <niftkPolyTool.h>
#include <niftkPosnTool.h>

#include "Internal/niftkGeneralSegmentorGUI.h"

namespace niftk
{

class GeneralSegmentorControllerPrivate
{
  Q_DECLARE_PUBLIC(GeneralSegmentorController)

  GeneralSegmentorController* const q_ptr;

public:

  GeneralSegmentorControllerPrivate(GeneralSegmentorController* q);
  ~GeneralSegmentorControllerPrivate();

  /// \brief All the GUI controls for the main view part.
  GeneralSegmentorGUI* m_GUI;

  /// \brief This class hooks into the Global Interaction system to respond to Key press events.
  ToolKeyPressStateMachine::Pointer m_ToolKeyPressStateMachine;

  /// \brief Selected orientation in the viewer.
  ImageOrientation m_Orientation;

  /// \brief Index of the selected slice in world space.
  int m_SelectedSliceIndex;

  /// \brief Keeps track of the previous slice axis.
  /// The slice axis is in terms of the reference image coordinates (voxel space), not the coordinates
  /// of the renderer (orientation in world space).
  int m_SliceAxis;

  /// \brief Keeps track of the previous slice index.
  /// The slice index is in terms of the reference image coordinates (voxel space), not the coordinates
  /// of the renderer (world space).
  int m_SliceIndex;

  /// \brief We track the previous selected position, as it is used in calculations of which slice we are on,
  /// as under certain conditions, you can't just take the slice index from the slice navigation controller.
  mitk::Point3D m_SelectedPosition;

  /// \brief Flag to stop re-entering code, while updating.
  bool m_IsUpdating;

  /// \brief Flag to stop re-entering code, while trying to delete/clear the pipeline.
  bool m_IsDeleting;

  /// \brief Additional flag to stop re-entering code, specifically to block
  /// slice change commands from the slice navigation controller.
  bool m_IsChangingSlice;

  /// \brief Tells if the segmentation has been created from scratch or an existing segmentation is being edited.
  /// Its value is 'true' if an existing segmentation is 're-edited', otherwise false.
  bool m_WasRestarted;
};

//-----------------------------------------------------------------------------
GeneralSegmentorControllerPrivate::GeneralSegmentorControllerPrivate(GeneralSegmentorController* generalSegmentorController)
  : q_ptr(generalSegmentorController),
    m_ToolKeyPressStateMachine(nullptr),
    m_Orientation(IMAGE_ORIENTATION_UNKNOWN),
    m_SelectedSliceIndex(-1),
    m_SliceAxis(-1),
    m_SliceIndex(-1),
    m_IsUpdating(false),
    m_IsDeleting(false),
    m_IsChangingSlice(false),
    m_WasRestarted(false)
{
  Q_Q(GeneralSegmentorController);

  mitk::ToolManager* toolManager = q->GetToolManager();
  toolManager->RegisterTool("DrawTool");
  toolManager->RegisterTool("SeedTool");
  toolManager->RegisterTool("PolyTool");
  toolManager->RegisterTool("PosnTool");

  q->GetToolByType<DrawTool>()->InstallEventFilter(q);
  q->GetToolByType<SeedTool>()->InstallEventFilter(q);
  q->GetToolByType<PolyTool>()->InstallEventFilter(q);
  q->GetToolByType<PosnTool>()->InstallEventFilter(q);

//  m_ToolKeyPressStateMachine = MIDASToolKeyPressStateMachine::New("niftkToolKeyPressStateMachine", q);
  m_ToolKeyPressStateMachine = ToolKeyPressStateMachine::New(q);

  m_SelectedPosition.Fill(0);
}


//-----------------------------------------------------------------------------
GeneralSegmentorControllerPrivate::~GeneralSegmentorControllerPrivate()
{
}


//-----------------------------------------------------------------------------
GeneralSegmentorController::GeneralSegmentorController(IBaseView* view)
  : BaseSegmentorController(view),
    d_ptr(new GeneralSegmentorControllerPrivate(this))
{
}


//-----------------------------------------------------------------------------
GeneralSegmentorController::~GeneralSegmentorController()
{
  this->GetToolByType<DrawTool>()->RemoveEventFilter(this);
  this->GetToolByType<SeedTool>()->RemoveEventFilter(this);
  this->GetToolByType<PolyTool>()->RemoveEventFilter(this);
  this->GetToolByType<PosnTool>()->RemoveEventFilter(this);
}


//-----------------------------------------------------------------------------
BaseGUI* GeneralSegmentorController::CreateGUI(QWidget* parent)
{
  return new GeneralSegmentorGUI(parent);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::SetupGUI(QWidget* parent)
{
  Q_D(GeneralSegmentorController);

  BaseSegmentorController::SetupGUI(parent);

  d->m_GUI = dynamic_cast<GeneralSegmentorGUI*>(this->GetSegmentorGUI());

  this->connect(d->m_GUI, SIGNAL(CleanButtonClicked()), SLOT(OnCleanButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeButtonClicked()), SLOT(OnWipeButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipePlusButtonClicked()), SLOT(OnWipePlusButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeMinusButtonClicked()), SLOT(OnWipeMinusButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateUpButtonClicked()), SLOT(OnPropagateUpButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateDownButtonClicked()), SLOT(OnPropagateDownButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(Propagate3DButtonClicked()), SLOT(OnPropagate3DButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(OKButtonClicked()), SLOT(OnOKButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(CancelButtonClicked()), SLOT(OnCancelButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(RestartButtonClicked()), SLOT(OnRestartButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ResetButtonClicked()), SLOT(OnResetButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdApplyButtonClicked()), SLOT(OnThresholdApplyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdingCheckBoxToggled(bool)), SLOT(OnThresholdingCheckBoxToggled(bool)));
  this->connect(d->m_GUI, SIGNAL(SeePriorCheckBoxToggled(bool)), SLOT(OnSeePriorCheckBoxToggled(bool)));
  this->connect(d->m_GUI, SIGNAL(SeeNextCheckBoxToggled(bool)), SLOT(OnSeeNextCheckBoxToggled(bool)));
  this->connect(d->m_GUI, SIGNAL(RetainMarksCheckBoxToggled(bool)), SLOT(OnRetainMarksCheckBoxToggled(bool)));
  this->connect(d->m_GUI, SIGNAL(ThresholdValueChanged()), SLOT(OnThresholdValueChanged()));

  /// Transfer the focus back to the main window if any button is pressed.
  /// This is needed so that the key interactions (like 'a'/'z' for changing slice) keep working.
  this->connect(d->m_GUI, SIGNAL(NewSegmentationButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(CleanButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipePlusButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeMinusButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateUpButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateDownButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(Propagate3DButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(OKButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(CancelButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(RestartButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ResetButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdApplyButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdingCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(SeePriorCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(SeeNextCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(RetainMarksCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::IsNodeAValidReferenceImage(const mitk::DataNode* node)
{
  if (!node)
  {
    return false;
  }

  mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
  unsigned pixelComponents = image->GetPixelType().GetNumberOfComponents();

  /// Only grey scale, RGB or RGBA reference images are supported.
  return pixelComponents == 1 || pixelComponents == 3 || pixelComponents == 4;
}


//-----------------------------------------------------------------------------
std::vector<mitk::DataNode*> GeneralSegmentorController::GetWorkingNodesFrom(mitk::DataNode* segmentationNode)
{
  assert(segmentationNode);
  std::vector<mitk::DataNode*> workingNodes;

  if (niftk::IsNodeABinaryImage(segmentationNode))
  {
    mitk::DataStorage* dataStorage = this->GetDataStorage();
    mitk::DataNode* seedsNode = dataStorage->GetNamedDerivedNode(Tool::SEEDS_NAME.c_str(), segmentationNode, true);
    mitk::DataNode* currentContoursNode = dataStorage->GetNamedDerivedNode(Tool::CONTOURS_NAME.c_str(), segmentationNode, true);
    mitk::DataNode* drawContoursNode = dataStorage->GetNamedDerivedNode(Tool::DRAW_CONTOURS_NAME.c_str(), segmentationNode, true);
    mitk::DataNode* seePriorContoursNode = dataStorage->GetNamedDerivedNode(Tool::PRIOR_CONTOURS_NAME.c_str(), segmentationNode, true);
    mitk::DataNode* seeNextContoursNode = dataStorage->GetNamedDerivedNode(Tool::NEXT_CONTOURS_NAME.c_str(), segmentationNode, true);
    mitk::DataNode* regionGrowingImageNode = dataStorage->GetNamedDerivedNode(Tool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
    mitk::DataNode* initialSegmentationNode = dataStorage->GetNamedDerivedNode(Tool::INITIAL_SEGMENTATION_NAME.c_str(), segmentationNode, true);
    mitk::DataNode* initialSeedsNode = dataStorage->GetNamedDerivedNode(Tool::INITIAL_SEEDS_NAME.c_str(), segmentationNode, true);

    if (seedsNode
        && currentContoursNode
        && drawContoursNode
        && seePriorContoursNode
        && seeNextContoursNode
        && regionGrowingImageNode
        && initialSegmentationNode
        && initialSeedsNode
        )
    {
      // The order of this list must match the order they were created in.
      workingNodes.push_back(segmentationNode);
      workingNodes.push_back(seedsNode);
      workingNodes.push_back(currentContoursNode);
      workingNodes.push_back(drawContoursNode);
      workingNodes.push_back(seePriorContoursNode);
      workingNodes.push_back(seeNextContoursNode);
      workingNodes.push_back(regionGrowingImageNode);
      workingNodes.push_back(initialSegmentationNode);
      workingNodes.push_back(initialSeedsNode);
    }
  }

  return workingNodes;
}


//-----------------------------------------------------------------------------
int GeneralSegmentorController::GetReferenceImageSliceAxis() const
{
  int referenceImageSliceAxis = -1;
  const mitk::Image* referenceImage = this->GetReferenceImage();
  ImageOrientation orientation = this->GetOrientation();
  if (referenceImage && orientation != IMAGE_ORIENTATION_UNKNOWN)
  {
    referenceImageSliceAxis = niftk::GetThroughPlaneAxis(referenceImage, orientation);
  }
  return referenceImageSliceAxis;
}


//-----------------------------------------------------------------------------
int GeneralSegmentorController::GetReferenceImageSliceIndex() const
{
  int referenceImageSliceIndex = -1;

  const mitk::Image* referenceImage = this->GetReferenceImage();
  mitk::SliceNavigationController* snc = this->GetSliceNavigationController();

  if (referenceImage && snc)
  {
    const mitk::PlaneGeometry* planeGeometry = snc->GetCurrentPlaneGeometry();
    if (planeGeometry)
    {
      mitk::Point3D originInMm = planeGeometry->GetOrigin();
      mitk::Point3D originInVx;
      referenceImage->GetGeometry()->WorldToIndex(originInMm, originInVx);

      int viewAxis = this->GetReferenceImageSliceAxis();
      referenceImageSliceIndex = (int)(originInVx[viewAxis] + 0.5);
    }
  }
  return referenceImageSliceIndex;
}


//-----------------------------------------------------------------------------
int GeneralSegmentorController::GetReferenceImageSliceUpDirection()
{
  int upDirection = 0;
  const mitk::Image* referenceImage = this->GetReferenceImage();
  ImageOrientation orientation = this->GetOrientation();
  if (referenceImage && orientation != IMAGE_ORIENTATION_UNKNOWN)
  {
    upDirection = niftk::GetUpDirection(referenceImage, orientation);
  }
  return upDirection;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnNewSegmentationButtonClicked()
{
  Q_D(GeneralSegmentorController);

  /// Note:
  /// The 'new segmentation' button is enabled only when a reference image is set.
  /// A reference image gets set when the selection in the data manager changes to
  /// a valid reference image or a segmentation that was created by this segmentor.
  /// Hence, we can assume that we have a valid tool manager and reference image.

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  const mitk::Image* referenceImage = this->GetReferenceImage();
  assert(referenceImage);

  QList<mitk::DataNode::Pointer> selectedNodes = this->GetView()->GetDataManagerSelection();
  if (selectedNodes.size() != 1)
  {
    return;
  }

  mitk::DataNode* selectedNode = selectedNodes.at(0);

  /// Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation;
  bool isRestarting = false;

  if (this->IsNodeAValidSegmentationImage(selectedNode)
      && this->GetWorkingNodesFrom(selectedNode).empty())
  {
    try
    {
      bool binary;
      int backgroundValue, foregroundValue;
      mitk::Image* segmentationImage = dynamic_cast<mitk::Image*>(selectedNode->GetData());
      const mitk::Image* segmentationImageConst = segmentationImage;

      AccessFixedDimensionByItk_n(segmentationImageConst,
          ITKGetBinaryImagePixelValues, 3,
          (binary,
           backgroundValue,
           foregroundValue
          )
        );

      if (!binary)
      {
        QMessageBox::warning(
              d->m_GUI->GetParent(),
              tr("Start segmentation"),
              tr("The selected image is not a binary image. Cannot continue."),
              QMessageBox::Ok);

        return;
      }
      else
      {
        if (backgroundValue != 0 || foregroundValue != 1)
        {
          QString message =
              QString("The pixel values of the mask will be changed to 0-s and 1-s, respectively. "
                      "Current values are %1 and %2.\n"
                      "The modified image will not be saved to the file system."
                      ).arg(backgroundValue).arg(foregroundValue);

          QMessageBox::warning(
                d->m_GUI->GetParent(),
                tr("Start segmentation"),
                message,
                QMessageBox::Ok);

          int newBackgroundValue = 0;
          int newForegroundValue = 1;
          AccessFixedDimensionByItk_n(segmentationImage,
              ITKSetBinaryImagePixelValues, 3,
              (backgroundValue,
               foregroundValue,
               newBackgroundValue,
               newForegroundValue
              )
            );
        }
      }
    }
    catch (const mitk::AccessByItkException& e)
    {
      // mitk::Image is of wrong pixel type or dimension,
      // insert error handling here
    }
    newSegmentation =  selectedNode;
    isRestarting = true;
  }
  else
  {
    newSegmentation = this->CreateNewSegmentation();

    // The above method returns nullptr if the user exited the colour selection dialog box.
    if (newSegmentation.IsNull())
    {
      return;
    }
  }

  // Override the base colour to be orange, and we revert this when OK pressed at the end.
  mitk::Color tmpColor;
  tmpColor[0] = 1.0;
  tmpColor[1] = 0.65;
  tmpColor[2] = 0.0;
  mitk::ColorProperty::Pointer tmpColorProperty = mitk::ColorProperty::New(tmpColor);
  newSegmentation->SetColor(tmpColor);
  newSegmentation->SetProperty("binaryimage.selectedcolor", tmpColorProperty);

  // Set initial properties.
  newSegmentation->SetProperty("layer", mitk::IntProperty::New(90));
  newSegmentation->SetFloatProperty("opacity", 1.0f);
  newSegmentation->SetBoolProperty(ContourTool::EDITING_PROPERTY_NAME.c_str(), false);

  // Make sure these are up to date, even though we don't use them right now.
  referenceImage->GetStatistics()->GetScalarValueMin();
  referenceImage->GetStatistics()->GetScalarValueMax();

  // This creates the point set for the seeds.
  mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
  mitk::DataNode::Pointer pointSetNode = mitk::DataNode::New();
  pointSetNode->SetData(pointSet);
  pointSetNode->SetProperty("name", mitk::StringProperty::New(Tool::SEEDS_NAME));
  pointSetNode->SetFloatProperty("opacity", 1.0f);
  pointSetNode->SetProperty("point line width", mitk::IntProperty::New(1));
  pointSetNode->SetProperty("point 2D size", mitk::IntProperty::New(5));
  pointSetNode->SetBoolProperty("helper object", true);
  pointSetNode->SetBoolProperty("show distant lines", false);
  pointSetNode->SetFloatProperty("Pointset.2D.distance to plane", 0.1);
  pointSetNode->SetBoolProperty("show distances", false);
  pointSetNode->SetProperty("layer", mitk::IntProperty::New(99));
  pointSetNode->SetColor(1.0, 0.0, 0.0);

  // Create all the contours.
  mitk::Color currentContoursColour;
  currentContoursColour.Set(0.0f, 1.0f, 0.0f);
  mitk::DataNode::Pointer currentContours = this->CreateContourSet(currentContoursColour, Tool::CONTOURS_NAME, true, 97);
  mitk::DataNode::Pointer drawContours = this->CreateContourSet(DrawTool::CONTOUR_COLOR, Tool::DRAW_CONTOURS_NAME, true, 98);
  mitk::Color nextContoursColour;
  nextContoursColour.Set(0.0f, 1.0f, 1.0f);
  mitk::DataNode::Pointer nextContoursNode = this->CreateContourSet(nextContoursColour, Tool::NEXT_CONTOURS_NAME, false, 95);
  mitk::Color priorContoursColour;
  priorContoursColour.Set(0.68f, 0.85f, 0.90f);
  mitk::DataNode::Pointer priorContoursNode = this->CreateContourSet(priorContoursColour, Tool::PRIOR_CONTOURS_NAME, false, 96);

  // Create the region growing image.
  mitk::Color regionGrowingImageColour;
  regionGrowingImageColour.Set(0.0f, 0.0f, 1.0f);
  mitk::DataNode::Pointer regionGrowingImageNode = this->CreateHelperImage(referenceImage, regionGrowingImageColour, Tool::REGION_GROWING_NAME, false, 94);

  // Create nodes to store the original segmentation and seeds, so that it can be restored if the Restart button is pressed.
  mitk::DataNode::Pointer initialSegmentationNode = mitk::DataNode::New();
  initialSegmentationNode->SetProperty("name", mitk::StringProperty::New(Tool::INITIAL_SEGMENTATION_NAME));
  initialSegmentationNode->SetBoolProperty("helper object", true);
  initialSegmentationNode->SetBoolProperty("visible", false);
  initialSegmentationNode->SetProperty("layer", mitk::IntProperty::New(99));
  initialSegmentationNode->SetFloatProperty("opacity", 1.0f);
  initialSegmentationNode->SetColor(tmpColor);
  initialSegmentationNode->SetProperty("binaryimage.selectedcolor", tmpColorProperty);

  mitk::DataNode::Pointer initialSeedsNode = mitk::DataNode::New();
  initialSeedsNode->SetProperty("name", mitk::StringProperty::New(Tool::INITIAL_SEEDS_NAME));
  initialSeedsNode->SetBoolProperty("helper object", true);
  initialSeedsNode->SetBoolProperty("visible", false);
  initialSeedsNode->SetBoolProperty("show distant lines", false);
  initialSeedsNode->SetFloatProperty("Pointset.2D.distance to plane", 0.1);
  initialSeedsNode->SetBoolProperty("show distances", false);
  initialSeedsNode->SetProperty("layer", mitk::IntProperty::New(99));
  initialSeedsNode->SetColor(1.0, 0.0, 0.0);

  /// TODO
  /// We should not refer to mitk::RenderingManager::GetInstance() because the DnD display uses its
  /// own rendering manager, not this one, like the MITK display.
//    mitk::IRenderingManager* renderingManager = 0;
//    mitk::IRenderWindowPart* renderWindowPart = this->GetView()->GetActiveRenderWindowPart();
//    if (renderWindowPart)
//    {
//      renderingManager = renderWindowPart->GetRenderingManager();
//    }
//    if (renderingManager)
//    {
//      // Make sure these points and contours are not rendered in 3D, as there can be many of them if you "propagate",
//      // and furthermore, there seem to be several seg faults rendering contour code in 3D. Haven't investigated yet.
//      QList<vtkRenderWindow*> renderWindows = renderingManager->GetAllRegisteredVtkRenderWindows();
//      for (QList<vtkRenderWindow*>::const_iterator iter = renderWindows.begin(); iter != renderWindows.end(); ++iter)
//      {
//        if ( mitk::BaseRenderer::GetInstance((*iter))->GetMapperID() == mitk::BaseRenderer::Standard3D )
//        {
//          pointSetNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          seePriorNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          seeNextNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          currentContours->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          drawContours->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          initialSegmentationNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          initialSeedsNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//        }
//      }
//    }

  // Adding to data storage, where the ordering affects the layering.
  this->GetDataStorage()->Add(priorContoursNode, newSegmentation);
  this->GetDataStorage()->Add(nextContoursNode, newSegmentation);
  this->GetDataStorage()->Add(regionGrowingImageNode, newSegmentation);
  this->GetDataStorage()->Add(currentContours, newSegmentation);
  this->GetDataStorage()->Add(drawContours, newSegmentation);
  this->GetDataStorage()->Add(pointSetNode, newSegmentation);
  this->GetDataStorage()->Add(initialSegmentationNode, newSegmentation);
  this->GetDataStorage()->Add(initialSeedsNode, newSegmentation);

  newSegmentation->SetBoolProperty("midas.general_segmentor.see_prior", false);
  newSegmentation->SetBoolProperty("midas.general_segmentor.see_next", false);
  newSegmentation->SetBoolProperty("midas.general_segmentor.retain_marks", false);
  newSegmentation->SetBoolProperty("midas.general_segmentor.thresholding", false);
  newSegmentation->SetFloatProperty("midas.general_segmentor.lower_threshold", 0.0f);
  newSegmentation->SetFloatProperty("midas.general_segmentor.upper_threshold", 0.0f);
  /// We use this annotation property not to store an annotation, actually, but to store
  /// the selected position in the viewer after any slice change.
  mitk::AnnotationProperty::Pointer selectedPositionProperty = mitk::AnnotationProperty::New();
  selectedPositionProperty->SetPosition(this->GetSelectedPosition());
  newSegmentation->SetProperty("midas.general_segmentor.selected_position", selectedPositionProperty);

  // Set working data. See header file, as the order here is critical, and should match the documented order.
  std::vector<mitk::DataNode*> workingNodes(9);
  workingNodes[Tool::SEGMENTATION] = newSegmentation;
  workingNodes[Tool::SEEDS] = pointSetNode;
  workingNodes[Tool::CONTOURS] = currentContours;
  workingNodes[Tool::DRAW_CONTOURS] = drawContours;
  workingNodes[Tool::PRIOR_CONTOURS] = priorContoursNode;
  workingNodes[Tool::NEXT_CONTOURS] = nextContoursNode;
  workingNodes[Tool::REGION_GROWING] = regionGrowingImageNode;
  workingNodes[Tool::INITIAL_SEGMENTATION] = initialSegmentationNode;
  workingNodes[Tool::INITIAL_SEEDS] = initialSeedsNode;
  toolManager->SetWorkingData(workingNodes);

  int sliceAxis = this->GetReferenceImageSliceAxis();
  int sliceIndex = this->GetReferenceImageSliceIndex();

  if (sliceAxis == -1 || sliceIndex == -1)
  {
    this->RemoveWorkingNodes();

    QMessageBox::warning(
          d->m_GUI->GetParent(),
          "",
          "Cannot determine the axis and index of the current slice.\n"
          "Make sure that a 2D render window is selected. Cannot continue.",
          QMessageBox::Ok);
    return;
  }

  this->WaitCursorOn();

  if (isRestarting)
  {
    this->InitialiseSeedsForSlice(sliceAxis, sliceIndex);
    this->UpdateCurrentSliceContours();
  }

  this->StoreInitialSegmentation();

  // Setup GUI.
  d->m_GUI->SetAllWidgetsEnabled(true);

  this->GetView()->FocusOnCurrentWindow();

  this->UpdateCurrentSliceContours(false);
  this->UpdateRegionGrowing(false);
  this->RequestRenderWindowUpdate();

  d->m_SliceAxis = sliceAxis;
  d->m_SliceIndex = sliceIndex;
  d->m_SelectedPosition = this->GetSelectedPosition();

  d->m_WasRestarted = isRestarting;

  if (!isRestarting)
  {
    this->GetView()->SetDataManagerSelection(newSegmentation);
  }

  this->WaitCursorOff();
}


/**************************************************************
 * Start of: Functions to create reference data (hidden nodes)
 *************************************************************/

//-----------------------------------------------------------------------------
mitk::DataNode::Pointer GeneralSegmentorController::CreateHelperImage(const mitk::Image* referenceImage, const mitk::Color& colour, const std::string& name, bool visible, int layer)
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::Tool* drawTool = this->GetToolByType<DrawTool>();
  assert(drawTool);

  mitk::DataNode::Pointer helperImageNode = drawTool->CreateEmptySegmentationNode(referenceImage, name, colour);
  helperImageNode->SetColor(colour);
  helperImageNode->SetProperty("binaryimage.selectedcolor", mitk::ColorProperty::New(colour));
  helperImageNode->SetBoolProperty("helper object", true);
  helperImageNode->SetBoolProperty("visible", visible);
  helperImageNode->SetProperty("layer", mitk::IntProperty::New(layer));

  this->ApplyDisplayOptions(helperImageNode);

  return helperImageNode;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer GeneralSegmentorController::CreateContourSet(const mitk::Color& colour, const std::string& name, bool visible, int layer)
{
  mitk::ContourModelSet::Pointer contourSet = mitk::ContourModelSet::New();

  mitk::DataNode::Pointer contourSetNode = mitk::DataNode::New();

  contourSetNode->SetProperty("color", mitk::ColorProperty::New(colour));
  contourSetNode->SetProperty("contour.color", mitk::ColorProperty::New(colour));
  contourSetNode->SetFloatProperty("opacity", 1.0f);
  contourSetNode->SetProperty("name", mitk::StringProperty::New(name));
  contourSetNode->SetBoolProperty("helper object", true);
  contourSetNode->SetBoolProperty("visible", visible);
  contourSetNode->SetProperty("layer", mitk::IntProperty::New(layer));
  contourSetNode->SetData(contourSet);

  return contourSetNode;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::StoreInitialSegmentation()
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  std::vector<mitk::DataNode*> workingNodes = toolManager->GetWorkingData();

  mitk::DataNode* segmentationNode = workingNodes[Tool::SEGMENTATION];
  mitk::DataNode* seedsNode = workingNodes[Tool::SEEDS];
  mitk::DataNode* initialSegmentationNode = workingNodes[Tool::INITIAL_SEGMENTATION];
  mitk::DataNode* initialSeedsNode = workingNodes[Tool::INITIAL_SEEDS];

  initialSegmentationNode->SetData(dynamic_cast<mitk::Image*>(segmentationNode->GetData())->Clone());
  initialSeedsNode->SetData(dynamic_cast<mitk::PointSet*>(seedsNode->GetData())->Clone());
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* /*renderer*/)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  std::vector<mitk::DataNode*> workingNodes = this->GetWorkingNodes();
  if (!workingNodes.empty() && node == workingNodes[Tool::SEGMENTATION])
  {
    bool segmentationNodeVisibility;
    if (node->GetVisibility(segmentationNodeVisibility, 0) && segmentationNodeVisibility)
    {
      workingNodes[Tool::SEEDS]->SetVisibility(true);
      workingNodes[Tool::CONTOURS]->SetVisibility(true);
      workingNodes[Tool::DRAW_CONTOURS]->SetVisibility(true);
      if (d->m_GUI->IsSeePriorCheckBoxChecked())
      {
        workingNodes[Tool::PRIOR_CONTOURS]->SetVisibility(true);
      }
      if (d->m_GUI->IsSeeNextCheckBoxChecked())
      {
        workingNodes[Tool::NEXT_CONTOURS]->SetVisibility(true);
      }
      if (d->m_GUI->IsThresholdingCheckBoxChecked())
      {
        workingNodes[Tool::REGION_GROWING]->SetVisibility(true);
      }
      workingNodes[Tool::INITIAL_SEGMENTATION]->SetVisibility(false);
      workingNodes[Tool::INITIAL_SEEDS]->SetVisibility(false);

      mitk::ToolManager::Pointer toolManager = this->GetToolManager();
      PolyTool* polyTool = this->GetToolByType<PolyTool>();
      assert(polyTool);
      polyTool->SetFeedbackContourVisible(toolManager->GetActiveTool() == polyTool);
    }
    else
    {
      for (std::size_t i = 1; i < workingNodes.size(); ++i)
      {
        workingNodes[i]->SetVisibility(false);
      }
    }
  }
}


/**************************************************************
 * End of: Functions to create reference data (hidden nodes)
 *************************************************************/


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnReferenceNodesChanged()
{
  BaseSegmentorController::OnReferenceNodesChanged();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnWorkingNodesChanged()
{
  Q_D(GeneralSegmentorController);

  /// This will update the GUI controls.
  BaseSegmentorController::OnWorkingNodesChanged();

  if (this->HasWorkingNodes())
  {
    /// Now we select the same position in the viewer where we were last time.

    d->m_IsChangingSlice = true;

    mitk::DataNode* segmentationNode = this->GetWorkingNode();
    mitk::BaseProperty* property = segmentationNode->GetProperty("midas.general_segmentor.selected_position");
    if (auto annotationProperty = dynamic_cast<mitk::AnnotationProperty*>(property))
    {
      this->SetSelectedPosition(annotationProperty->GetPosition());
    }

    d->m_Orientation = this->GetOrientation();
    d->m_SelectedSliceIndex = this->GetSliceIndex();
    d->m_SliceAxis = this->GetReferenceImageSliceAxis();
    d->m_SliceIndex = this->GetReferenceImageSliceIndex();
    d->m_SelectedPosition = this->GetSelectedPosition();

    d->m_IsChangingSlice = false;
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdateGUI() const
{
  Q_D(const GeneralSegmentorController);

  mitk::DataNode* referenceNode = this->GetReferenceNode();
  if (referenceNode)
  {
    if (niftk::IsNodeAGreyScaleImage(referenceNode))
    {
      d->m_GUI->SetThresholdingCheckBoxEnabled(true);
      d->m_GUI->SetThresholdingCheckBoxToolTip("Tick this in if you want to apply thresholding within the current regions.");
    }
    else
    {
      d->m_GUI->SetThresholdingCheckBoxEnabled(false);
      d->m_GUI->SetThresholdingCheckBoxToolTip("Thresholding is not supported for RGB images.");
    }

    const mitk::Image* referenceImage = this->GetReferenceImage();

    float lowestPixelValue;
    if (!referenceNode->GetFloatProperty("midas.general_segmentor.lowest_pixel_value", lowestPixelValue))
    {
      lowestPixelValue = referenceImage->GetStatistics()->GetScalarValueMin();
    }

    float highestPixelValue;
    if (!referenceNode->GetFloatProperty("midas.general_segmentor.highest_pixel_value", highestPixelValue))
    {
      highestPixelValue = referenceImage->GetStatistics()->GetScalarValueMax();
    }

    d->m_GUI->SetLowerAndUpperIntensityRanges(lowestPixelValue, highestPixelValue);
  }
  else
  {
    d->m_GUI->SetSeePriorCheckBoxChecked(false);
    d->m_GUI->SetSeeNextCheckBoxChecked(false);
    d->m_GUI->SetRetainMarksCheckBoxChecked(false);
    d->m_GUI->SetThresholdingCheckBoxEnabled(false);
    d->m_GUI->SetThresholdingCheckBoxToolTip("");
    d->m_GUI->SetThresholdingCheckBoxChecked(false);
    d->m_GUI->SetThresholdingWidgetsEnabled(false);
    d->m_GUI->SetLowerAndUpperIntensityRanges(0.0, 0.0);
    d->m_GUI->SetLowerThreshold(0.0);
    d->m_GUI->SetUpperThreshold(0.0);
    d->m_GUI->SetSeedMinAndMaxValues(0.0, 0.0);
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    bool boolValue;
    float floatValue;

    if (!segmentationNode->GetBoolProperty("midas.general_segmentor.see_prior", boolValue))
    {
      boolValue = false;
      segmentationNode->SetBoolProperty("midas.general_segmentor.see_prior", boolValue);
    }
    d->m_GUI->SetSeePriorCheckBoxChecked(boolValue);

    if (!segmentationNode->GetBoolProperty("midas.general_segmentor.see_next", boolValue))
    {
      boolValue = false;
      segmentationNode->SetBoolProperty("midas.general_segmentor.see_next", boolValue);
    }
    d->m_GUI->SetSeeNextCheckBoxChecked(boolValue);

    if (!segmentationNode->GetBoolProperty("midas.general_segmentor.retain_marks", boolValue))
    {
      boolValue = false;
      segmentationNode->SetBoolProperty("midas.general_segmentor.retain_marks", boolValue);
    }
    d->m_GUI->SetRetainMarksCheckBoxChecked(boolValue);

    if (!segmentationNode->GetBoolProperty("midas.general_segmentor.thresholding", boolValue))
    {
      boolValue = false;
      segmentationNode->SetBoolProperty("midas.general_segmentor.thresholding", boolValue);
    }
    d->m_GUI->SetThresholdingCheckBoxChecked(boolValue);
    d->m_GUI->SetThresholdingWidgetsEnabled(boolValue);

    if (!segmentationNode->GetFloatProperty("midas.general_segmentor.lower_threshold", floatValue))
    {
      floatValue = 0.0f;
      segmentationNode->SetBoolProperty("midas.general_segmentor.", floatValue);
    }
    d->m_GUI->SetLowerThreshold(floatValue);

    if (!segmentationNode->GetFloatProperty("midas.general_segmentor.upper_threshold", floatValue))
    {
      floatValue = 0.0f;
      segmentationNode->SetBoolProperty("midas.general_segmentor.", floatValue);
    }
    d->m_GUI->SetUpperThreshold(floatValue);

    this->RecalculateMinAndMaxOfSeedValues();
  }
  else
  {
    d->m_GUI->SetSeePriorCheckBoxChecked(false);
    d->m_GUI->SetSeeNextCheckBoxChecked(false);
    d->m_GUI->SetRetainMarksCheckBoxChecked(false);
    d->m_GUI->SetThresholdingCheckBoxEnabled(false);
    d->m_GUI->SetThresholdingCheckBoxToolTip("");
    d->m_GUI->SetThresholdingCheckBoxChecked(false);
    d->m_GUI->SetThresholdingWidgetsEnabled(false);
    d->m_GUI->SetLowerAndUpperIntensityRanges(0.0, 0.0);
    d->m_GUI->SetLowerThreshold(0.0);
    d->m_GUI->SetUpperThreshold(0.0);
    d->m_GUI->SetSeedMinAndMaxValues(0.0, 0.0);
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnViewGetsVisible()
{
  /// TODO
//  mitk::GlobalInteraction::GetInstance()->AddListener(d->m_ToolKeyPressStateMachine);

  // Connect registered tools back to here, so we can do seed processing logic here.
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  PolyTool* polyTool = this->GetToolByType<PolyTool>();
  polyTool->ContoursHaveChanged += mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);

  DrawTool* drawTool = this->GetToolByType<DrawTool>();
  drawTool->ContoursHaveChanged += mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnViewGetsHidden()
{
  BaseSegmentorController::OnViewGetsHidden();

  /// TODO
//  mitk::GlobalInteraction::GetInstance()->RemoveListener(d->m_ToolKeyPressStateMachine);

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  PolyTool* polyTool = this->GetToolByType<PolyTool>();
  polyTool->ContoursHaveChanged -= mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);

  DrawTool* drawTool = this->GetToolByType<DrawTool>();
  drawTool->ContoursHaveChanged -= mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnSelectedSliceChanged(ImageOrientation orientation, int selectedSliceIndex)
{
  Q_D(GeneralSegmentorController);

  if (orientation != d->m_Orientation || selectedSliceIndex != d->m_SelectedSliceIndex)
  {
    if (this->HasWorkingNodes()
        && orientation != IMAGE_ORIENTATION_UNKNOWN)
    {
      int sliceAxis = this->GetReferenceImageSliceAxis();
      int sliceIndex = this->GetReferenceImageSliceIndex();
      mitk::Point3D selectedPosition = this->GetSelectedPosition();

      if (sliceAxis == -1 || sliceIndex == -1)
      {
        QMessageBox::warning(
              d->m_GUI->GetParent(),
              "",
              "Cannot determine the axis and index of the current slice.\n"
              "Make sure that a 2D render window is selected. Cannot continue.",
              QMessageBox::Ok);
        return;
      }

      if (!d->m_IsUpdating
          && !d->m_IsChangingSlice)
      {
        bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();

        mitk::Operation* doOp;
        mitk::Operation* undoOp;
        mitk::OperationEvent* opEvent;

        DrawTool* drawTool = this->GetToolByType<DrawTool>();

        mitk::PointSet::Pointer copyOfCurrentSeeds = mitk::PointSet::New();
        mitk::PointSet::Pointer newSeeds = mitk::PointSet::New();
        std::vector<int> outputRegion;

        mitk::PointSet* seeds = this->GetSeeds();
        bool oldSliceIsEmpty = false;
        bool newSliceIsEmpty = true;

        bool wasUpdating = d->m_IsUpdating;
        d->m_IsUpdating = true;

        ///////////////////////////////////////////////////////
        // See: https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/1742
        //      for the whole logic surrounding changing slice.
        ///////////////////////////////////////////////////////

        try
        {
          const mitk::Image* referenceImage = this->GetReferenceImage();
          const mitk::Image* segmentationImage = this->GetWorkingImage();
          assert(referenceImage && segmentationImage);

          AccessFixedDimensionByItk_n(segmentationImage,
              ITKSliceIsEmpty, 3,
              (sliceAxis,
               sliceIndex,
               newSliceIsEmpty
              )
            );

          bool operationCancelled = false;

          if (orientation == d->m_Orientation
              && std::abs(d->m_SliceIndex - sliceIndex) == 1
              && d->m_GUI->IsRetainMarksCheckBoxChecked())
          {
            QMessageBox::StandardButton answer = QMessageBox::NoButton;

            if (!isThresholdingOn)
            {
              AccessFixedDimensionByItk_n(segmentationImage,
                  ITKSliceIsEmpty, 3,
                  (d->m_SliceAxis,
                   d->m_SliceIndex,
                   oldSliceIsEmpty
                  )
                );
            }

            if (oldSliceIsEmpty)
            {
              answer = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
                                                      tr("The previous slice is empty - retain marks cannot be performed.\n"
                                                         "Use the 'wipe' functionality to erase slices instead"),
                                                      QMessageBox::Ok
                                   );
            }
            else if (!newSliceIsEmpty)
            {
              answer = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
                                                      tr("The new slice is not empty - retain marks will overwrite the slice.\n"
                                                         "Are you sure?"),
                                                      QMessageBox::Yes | QMessageBox::No);
            }

            if (answer == QMessageBox::Ok || answer == QMessageBox::No )
            {
              operationCancelled = true;
            }
            else
            {
              AccessFixedDimensionByItk_n(segmentationImage,
                  ITKPreprocessingOfSeedsForChangingSlice, 3,
                  (seeds,
                   d->m_SliceAxis,
                   d->m_SliceIndex,
                   sliceAxis,
                   sliceIndex,
                   false, // We propagate seeds at current position, so no optimisation
                   newSliceIsEmpty,
                   copyOfCurrentSeeds,
                   newSeeds,
                   outputRegion
                  )
                );

              if (isThresholdingOn)
              {
                QString message = tr("Thresholding slice %1 before copying marks to slice %2").arg(d->m_SliceIndex).arg(sliceIndex);
                OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
                doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, true);
                undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, true);
                opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
                mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
                this->ExecuteOperation(doOp);

                drawTool->ClearWorkingData();
                this->UpdateCurrentSliceContours();
              }

              itk::Orientation itkOrientation = GetItkOrientation(this->GetOrientation());

              // Do retain marks, which copies slice from beforeSliceIndex to afterSliceIndex
              QString message = tr("Retaining marks in slice %1 and copying to %2").arg(d->m_SliceIndex).arg(sliceIndex);
              OpRetainMarks::ProcessorPointer processor = OpRetainMarks::ProcessorType::New();
              doOp = new OpRetainMarks(OP_RETAIN_MARKS, true, sliceAxis, d->m_SliceIndex, sliceIndex, itkOrientation, outputRegion, processor);
              undoOp = new OpRetainMarks(OP_RETAIN_MARKS, false, sliceAxis, d->m_SliceIndex, sliceIndex, itkOrientation, outputRegion, processor);
              opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
              this->ExecuteOperation(doOp);
            }
          }
          else // so, switching orientation, jumping slice or "Retain Marks" is off.
          {
            AccessFixedDimensionByItk_n(segmentationImage,
                ITKPreprocessingOfSeedsForChangingSlice, 3,
                (seeds,
                 d->m_SliceAxis,
                 d->m_SliceIndex,
                 sliceAxis,
                 sliceIndex,
                 true, // optimise seed position on current slice.
                 newSliceIsEmpty,
                 copyOfCurrentSeeds,
                 newSeeds,
                 outputRegion
                )
              );

            if (isThresholdingOn)
            {
              OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
              doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, true);
              undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, true);
              opEvent = new mitk::OperationEvent(this, doOp, undoOp, "Apply threshold");
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
              this->ExecuteOperation(doOp);

              drawTool->ClearWorkingData();
              this->UpdateCurrentSliceContours();
            }
            else // threshold box not checked
            {
              bool oldSliceHasUnenclosedSeeds = this->DoesSliceHaveUnenclosedSeeds(false, d->m_SliceAxis, d->m_SliceIndex);

              if (oldSliceHasUnenclosedSeeds)
              {
                OpWipe::ProcessorPointer processor = OpWipe::ProcessorType::New();
                doOp = new OpWipe(OP_WIPE, true, d->m_SliceAxis, d->m_SliceIndex, outputRegion, newSeeds, processor);
                undoOp = new OpWipe(OP_WIPE, false, d->m_SliceAxis, d->m_SliceIndex, outputRegion, copyOfCurrentSeeds, processor);
                opEvent = new mitk::OperationEvent(this, doOp, undoOp, "Wipe command");
                mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
                this->ExecuteOperation(doOp);
              }
              else // so, we don't have unenclosed seeds
              {
                // There may be the case where the user has simply drawn a region, and put a seed in the middle.
                // So, we do a region growing, without intensity limits. (we already know there are no unenclosed seeds).

                this->UpdateRegionGrowing(false,
                                          d->m_SliceAxis,
                                          d->m_SliceIndex,
                                          referenceImage->GetStatistics()->GetScalarValueMinNoRecompute(),
                                          referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute(),
                                          false);

                // Then we "apply" this region growing.
                OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
                doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, false);
                undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, false);
                opEvent = new mitk::OperationEvent(this, doOp, undoOp, "Apply threshold");
                mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
                this->ExecuteOperation(doOp);

                drawTool->ClearWorkingData();

              } // end if/else unenclosed seeds
            } // end if/else thresholding on
          } // end if/else retain marks.

          if (!operationCancelled)
          {
            QString oldOrientationName = QString::fromStdString(GetOrientationName(d->m_Orientation));
            QString newOrientationName = QString::fromStdString(GetOrientationName(orientation));
            QString message = tr("Propagate seeds on %1 slice %2 (image axis: %3, slice: %4)")
                .arg(newOrientationName).arg(selectedSliceIndex)
                .arg(sliceAxis).arg(sliceIndex);
            doOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, true, sliceAxis, sliceIndex, newSeeds);
            undoOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, false, d->m_SliceAxis, d->m_SliceIndex, copyOfCurrentSeeds);
            opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
            mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
            this->ExecuteOperation(doOp);

            message = tr("Change from %1 slice %2 to %3 slice %4 (from image axis %5 slice %6 to axis %7 slice %8)")
                .arg(oldOrientationName).arg(d->m_SelectedSliceIndex).arg(newOrientationName).arg(selectedSliceIndex)
                .arg(d->m_SliceAxis).arg(d->m_SliceIndex).arg(sliceAxis).arg(sliceIndex);
            doOp = new OpChangeSliceCommand(OP_CHANGE_SLICE, true, d->m_SelectedPosition, selectedPosition);
            undoOp = new OpChangeSliceCommand(OP_CHANGE_SLICE, false, d->m_SelectedPosition, selectedPosition);
            opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
            mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
            this->ExecuteOperation(doOp);

            mitk::ToolManager* toolManager = this->GetToolManager();
            if (PolyTool* polyTool = dynamic_cast<PolyTool*>(toolManager->GetActiveTool()))
            {
              /// This makes the poly tool save its result to the working data nodes and stay it open.
              polyTool->Deactivated();
              polyTool->Activated();
            }
          }

          this->UpdateCurrentSliceContours(false);
          this->UpdatePriorAndNext(false);
          this->UpdateRegionGrowing(false);
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not change slice: Caught mitk::AccessByItkException:" << e.what();
        }
        catch(const itk::ExceptionObject& e)
        {
          MITK_ERROR << "Could not change slice: Caught itk::ExceptionObject:" << e.what();
        }

        d->m_IsUpdating = wasUpdating;

        this->RequestRenderWindowUpdate();
      } // if not being updated and not changing slice

      d->m_SliceAxis = sliceAxis;
      d->m_SliceIndex = sliceIndex;
      d->m_SelectedPosition = selectedPosition;

      mitk::AnnotationProperty::Pointer selectedPositionProperty = mitk::AnnotationProperty::New();
      selectedPositionProperty->SetPosition(selectedPosition);
      mitk::DataNode* segmentationNode = this->GetWorkingNode();
      segmentationNode->SetProperty("midas.general_segmentor.selected_position", selectedPositionProperty);
    } // if initialised and valid orientation (2D window selected)

    d->m_Orientation = orientation;
    d->m_SelectedSliceIndex = selectedSliceIndex;

  } // if orientation or selected slice has changed
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnNodeChanged(const mitk::DataNode* node)
{
  Q_D(GeneralSegmentorController);

  if (d->m_IsDeleting
      || d->m_IsUpdating
      || !this->HasWorkingNodes()
      )
  {
    return;
  }

  std::vector<mitk::DataNode*> workingNodes = this->GetWorkingNodes();

  bool seedsChanged = false;
  bool drawContoursChanged = false;

  if (workingNodes[Tool::SEEDS] && workingNodes[Tool::SEEDS] == node)
  {
    seedsChanged = true;
  }

  if (workingNodes[Tool::DRAW_CONTOURS] && workingNodes[Tool::DRAW_CONTOURS] == node)
  {
    drawContoursChanged = true;
  }

  if (!seedsChanged && !drawContoursChanged)
  {
    return;
  }

  mitk::DataNode* segmentationNode = workingNodes[Tool::SEGMENTATION];

  mitk::PointSet* seeds = this->GetSeeds();
  if (seeds && seeds->GetSize() > 0)
  {
    bool contourIsBeingEdited = false;
    if (segmentationNode == node)
    {
      segmentationNode->GetBoolProperty(ContourTool::EDITING_PROPERTY_NAME.c_str(), contourIsBeingEdited);
    }

    if (!contourIsBeingEdited)
    {
      if (seedsChanged)
      {
        this->RecalculateMinAndMaxOfSeedValues();
      }

      if (seedsChanged || drawContoursChanged)
      {
        this->UpdateRegionGrowing();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnNodeRemoved(const mitk::DataNode* removedNode)
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();

  if (segmentationNode == removedNode)
  {
    QMessageBox::StandardButtons saveSegmentation = QMessageBox::question(
          this->GetGUI()->GetParent(),
          "Save segmentation?",
          "You have moved away the segmentation from its reference image. "
          "You cannot continue editing the image without a reference image. "
          "Do you want to save the segmentation or cancel it and discard the "
          "changes?",
          QMessageBox::Ok | QMessageBox::Cancel,
          QMessageBox::Ok);

    if (saveSegmentation == QMessageBox::Ok)
    {
      this->OnOKButtonClicked();
    }
    else // if (saveSegmentation == QMessageBox::Cancel)
    {
      this->OnCancelButtonClicked();
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnContoursChanged()
{
  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
mitk::PointSet* GeneralSegmentorController::GetSeeds() const
{
  if (auto seedsNode = this->GetWorkingNode(Tool::SEEDS))
  {
    return dynamic_cast<mitk::PointSet*>(seedsNode->GetData());
  }

  return nullptr;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::InitialiseSeedsForSlice(int sliceAxis, int sliceIndex)
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  const mitk::Image* segmentationImage = this->GetWorkingImage(Tool::SEGMENTATION);
  assert(segmentationImage);

  try
  {
    AccessFixedDimensionByItk_n(segmentationImage,
        ITKInitialiseSeedsForSlice, 3,
        (seeds,
         sliceAxis,
         sliceIndex
        )
      );
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during ITKInitialiseSeedsForSlice, so have not initialised seeds correctly, caused by:" << e.what();
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::RecalculateMinAndMaxOfSeedValues() const
{
  Q_D(const GeneralSegmentorController);

  const mitk::Image* referenceImage = this->GetReferenceImage();

  if (!referenceImage || referenceImage->GetPixelType().GetNumberOfComponents() != 1)
  {
    d->m_GUI->SetSeedMinAndMaxValues(0.0, 0.0);
    return;
  }

  mitk::PointSet* seeds = this->GetSeeds();
  if (seeds)
  {
    double min = 0.0;
    double max = 0.0;

    int sliceIndex = this->GetReferenceImageSliceIndex();
    int sliceAxis = this->GetReferenceImageSliceAxis();

    if (sliceIndex == -1 || sliceAxis == -1)
    {
      QMessageBox::warning(
            d->m_GUI->GetParent(),
            "",
            "Cannot determine the axis and index of the current slice.\n"
            "Make sure that a 2D render window is selected. Cannot continue.",
            QMessageBox::Ok);
      return;
    }

    try
    {
      AccessFixedDimensionByItk_n(referenceImage, ITKRecalculateMinAndMaxOfSeedValues, 3, (seeds, sliceAxis, sliceIndex, min, max));
      d->m_GUI->SetSeedMinAndMaxValues(min, max);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning recalculating min and max of seeds values, due to:" << e.what();
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdateCurrentSliceContours(bool updateRendering)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  int sliceIndex = this->GetReferenceImageSliceIndex();
  int sliceAxis = this->GetReferenceImageSliceAxis();

  if (sliceIndex == -1 || sliceAxis == -1)
  {
    QMessageBox::warning(
          d->m_GUI->GetParent(),
          "",
          "Cannot determine the axis and index of the current slice.\n"
          "Make sure that a 2D render window is selected. Cannot continue.",
          QMessageBox::Ok);
    return;
  }

  const mitk::Image* segmentationImage = this->GetWorkingImage(Tool::SEGMENTATION);
  assert(segmentationImage);

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode* contoursNode = this->GetWorkingNode(Tool::CONTOURS);
  mitk::ContourModelSet* contours = dynamic_cast<mitk::ContourModelSet*>(contoursNode->GetData());

  // TODO
  // This assertion fails sometimes if both the morphological and irregular (this) volume editor is
  // switched on and you are using the paintbrush tool of the morpho editor.
//  assert(contourSet);

  if (contours)
  {
    GenerateOutlineFromBinaryImage(segmentationImage, sliceAxis, sliceIndex, sliceIndex, contours);

    if (contours->GetSize() > 0)
    {
      contoursNode->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnSeePriorCheckBoxToggled(bool checked)
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  segmentationNode->SetBoolProperty("midas.general_segmentor.see_prior", checked);

  if (checked)
  {
    this->UpdatePriorAndNext();
  }

  this->GetWorkingNode(Tool::PRIOR_CONTOURS)->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnSeeNextCheckBoxToggled(bool checked)
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  segmentationNode->SetBoolProperty("midas.general_segmentor.see_next", checked);

  if (checked)
  {
    this->UpdatePriorAndNext();
  }

  this->GetWorkingNode(Tool::NEXT_CONTOURS)->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnRetainMarksCheckBoxToggled(bool checked)
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  segmentationNode->SetBoolProperty("midas.general_segmentor.retain_marks", checked);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnThresholdingCheckBoxToggled(bool checked)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    // So, if there is NO working data, we leave the widgets disabled regardless.
    d->m_GUI->SetThresholdingWidgetsEnabled(false);
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  segmentationNode->SetBoolProperty("midas.general_segmentor.thresholding", checked);

  this->RecalculateMinAndMaxOfSeedValues();

  d->m_GUI->SetThresholdingWidgetsEnabled(checked);

  if (checked)
  {
    this->UpdateRegionGrowing();
  }

  this->GetWorkingNode(Tool::REGION_GROWING)->SetVisibility(checked);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnThresholdValueChanged()
{
  Q_D(GeneralSegmentorController);

  mitk::DataNode* segmentationNode = this->GetWorkingNode();

  float lowerThreshold = d->m_GUI->GetLowerThreshold();
  segmentationNode->GetFloatProperty("midas.general_segmentor.lower_threshold", lowerThreshold);

  float upperThreshold = d->m_GUI->GetUpperThreshold();
  segmentationNode->GetFloatProperty("midas.general_segmentor.upper_threshold", upperThreshold);

  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdateRegionGrowing(bool updateRendering)
{
  Q_D(GeneralSegmentorController);

  bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();

  if (isThresholdingOn)
  {
    int sliceAxis = this->GetReferenceImageSliceAxis();
    int sliceIndex = this->GetReferenceImageSliceIndex();

    if (sliceIndex == -1 || sliceAxis == -1)
    {
      QMessageBox::warning(
            d->m_GUI->GetParent(),
            "",
            "Cannot determine the axis and index of the current slice.\n"
            "Make sure that a 2D render window is selected. Cannot continue.",
            QMessageBox::Ok);
      return;
    }

    double lowerThreshold = d->m_GUI->GetLowerThreshold();
    double upperThreshold = d->m_GUI->GetUpperThreshold();
    bool skipUpdate = !isThresholdingOn;

    this->UpdateRegionGrowing(isThresholdingOn, sliceAxis, sliceIndex, lowerThreshold, upperThreshold, skipUpdate);

    if (updateRendering)
    {
      this->RequestRenderWindowUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdateRegionGrowing(
    bool isVisible,
    int sliceAxis,
    int sliceIndex,
    double lowerThreshold,
    double upperThreshold,
    bool skipUpdate
    )
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  const mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    mitk::DataNode* segmentationNode = this->GetWorkingNode();
    const mitk::Image* segmentationImage = this->GetWorkingImage();

    if (segmentationImage && segmentationNode)
    {
      this->GetWorkingNode(Tool::REGION_GROWING)->SetVisibility(isVisible);

      bool wasUpdating = d->m_IsUpdating;
      d->m_IsUpdating = true;

      mitk::DataNode* regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(Tool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image* regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager* toolManager = this->GetToolManager();
      assert(toolManager);

      PolyTool* polyTool = this->GetToolByType<PolyTool>();
      assert(polyTool);

      mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();

      mitk::ContourModel* polyToolContour = polyTool->GetContour();
      if (polyToolContour && polyToolContour->GetNumberOfVertices() >= 2)
      {
        polyToolContours->AddContourModel(polyToolContour);
      }

      mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingNode(Tool::CONTOURS)->GetData());
      mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingNode(Tool::DRAW_CONTOURS)->GetData());

      if (sliceAxis != -1 && sliceIndex != -1)
      {
        try
        {
          if (referenceImage->GetPixelType().GetNumberOfComponents() == 1)
          {
            AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
                ITKUpdateRegionGrowing, 3,
                (skipUpdate,
                 segmentationImage,
                 seeds,
                 segmentationContours,
                 drawToolContours,
                 polyToolContours,
                 sliceAxis,
                 sliceIndex,
                 lowerThreshold,
                 upperThreshold,
                 regionGrowingImage  // This is the image we are writing to.
                )
            );
          }
          else
          {
            AccessFixedTypeByItk_n(referenceImage, // The reference image is the RGB image (read only).
                ITKUpdateRegionGrowing,
                MITK_ACCESSBYITK_COMPOSITE_PIXEL_TYPES_SEQ,
                (3),
                (skipUpdate,
                 segmentationImage,
                 seeds,
                 segmentationContours,
                 drawToolContours,
                 polyToolContours,
                 sliceAxis,
                 sliceIndex,
                 regionGrowingImage  // This is the image we are writing to.
                )
            );
          }

          regionGrowingImage->Modified();
          regionGrowingNode->Modified();
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do region growing: Caught exception, so abandoning ITK pipeline update:" << e.what();
        }
      }
      else
      {
        MITK_ERROR << "Could not do region growing: Error sliceAxis=" << sliceAxis << ", sliceIndex=" << sliceIndex;
      }

      d->m_IsUpdating = wasUpdating;
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdatePriorAndNext(bool updateRendering)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  int sliceIndex = this->GetReferenceImageSliceIndex();
  int sliceAxis = this->GetReferenceImageSliceAxis();

  if (sliceIndex == -1 || sliceAxis == -1)
  {
    QMessageBox::warning(
          d->m_GUI->GetParent(),
          "",
          "Cannot determine the axis and index of the current slice.\n"
          "Make sure that a 2D render window is selected. Cannot continue.",
          QMessageBox::Ok);
    return;
  }

  const mitk::Image* segmentationImage = this->GetWorkingImage();

  if (d->m_GUI->IsSeePriorCheckBoxChecked())
  {
    mitk::DataNode* priorContoursNode = this->GetWorkingNode(Tool::PRIOR_CONTOURS);
    mitk::ContourModelSet::Pointer priorContours = dynamic_cast<mitk::ContourModelSet*>(priorContoursNode->GetData());
    GenerateOutlineFromBinaryImage(segmentationImage, sliceAxis, sliceIndex - 1, sliceIndex, priorContours);

    if (priorContours->GetSize() > 0)
    {
      priorContoursNode->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }

  if (d->m_GUI->IsSeeNextCheckBoxChecked())
  {
    mitk::DataNode* nextContoursNode = this->GetWorkingNode(Tool::NEXT_CONTOURS);
    mitk::ContourModelSet::Pointer nextContours = dynamic_cast<mitk::ContourModelSet*>(nextContoursNode->GetData());
    GenerateOutlineFromBinaryImage(segmentationImage, sliceAxis, sliceIndex + 1, sliceIndex, nextContours);

    if (nextContours->GetSize() > 0)
    {
      nextContoursNode->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceAxis, int sliceIndex)
{
  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  return this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceAxis, sliceIndex, seeds);
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceAxis, int sliceIndex, const mitk::PointSet* seeds)
{
  Q_D(GeneralSegmentorController);

  bool sliceDoesHaveUnenclosedSeeds = false;

  if (!this->HasWorkingNodes())
  {
    return sliceDoesHaveUnenclosedSeeds;
  }

  const mitk::Image* referenceImage = this->GetReferenceImage();
  const mitk::Image* segmentationImage = this->GetWorkingImage(Tool::SEGMENTATION);

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  PolyTool* polyTool = this->GetToolByType<PolyTool>();
  assert(polyTool);

  mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();
  mitk::ContourModel* polyToolContour = polyTool->GetContour();
  if (polyToolContour && polyToolContour->GetNumberOfVertices() >= 2)
  {
    polyToolContours->AddContourModel(polyToolContour);
  }

  mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingNode(Tool::CONTOURS)->GetData());
  mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingNode(Tool::DRAW_CONTOURS)->GetData());

  double lowerThreshold = d->m_GUI->GetLowerThreshold();
  double upperThreshold = d->m_GUI->GetUpperThreshold();

  if (sliceAxis != -1 && sliceIndex != -1)
  {
    try
    {
      if (referenceImage->GetPixelType().GetNumberOfComponents() == 1)
      {
        // The reference image is the grey scale (read only).
        AccessFixedDimensionByItk_n(referenceImage,
          ITKSliceDoesHaveUnenclosedSeeds, 3,
            (seeds,
             segmentationContours,
             polyToolContours,
             drawToolContours,
             segmentationImage,
             lowerThreshold,
             upperThreshold,
             thresholdOn,
             sliceAxis,
             sliceIndex,
             sliceDoesHaveUnenclosedSeeds
            )
        );
      }
      else
      {
        // The reference image is RGB (read only).
        AccessFixedTypeByItk_n(
              referenceImage,
              ITKSliceDoesHaveUnenclosedSeedsNoThresholds,
              MITK_ACCESSBYITK_COMPOSITE_PIXEL_TYPES_SEQ,
              (3),
              (seeds,
               segmentationContours,
               polyToolContours,
               drawToolContours,
               segmentationImage,
               sliceAxis,
               sliceIndex,
               sliceDoesHaveUnenclosedSeeds)
              );
      }
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception during ITKSliceDoesHaveUnenclosedSeeds, so will return false, caused by:" << e.what();
    }
  }

  return sliceDoesHaveUnenclosedSeeds;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::FilterSeedsToCurrentSlice(
    const mitk::PointSet* inputPoints,
    int sliceAxis,
    int sliceIndex,
    mitk::PointSet* outputPoints
    )
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  const mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage,
          ITKFilterSeedsToCurrentSlice, 3,
          (inputPoints,
           sliceAxis,
           sliceIndex,
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
void GeneralSegmentorController::FilterSeedsToEnclosedSeedsOnSlice(
    const mitk::PointSet* inputPoints,
    bool thresholdOn,
    int sliceAxis,
    int sliceIndex,
    mitk::PointSet* outputPoints
    )
{
  outputPoints->Clear();

  mitk::PointSet::Pointer singleSeedPointSet = mitk::PointSet::New();

  mitk::PointSet::PointsConstIterator inputPointsIt = inputPoints->Begin();
  mitk::PointSet::PointsConstIterator inputPointsEnd = inputPoints->End();
  for ( ; inputPointsIt != inputPointsEnd; ++inputPointsIt)
  {
    mitk::PointSet::PointType point = inputPointsIt->Value();
    mitk::PointSet::PointIdentifier pointID = inputPointsIt->Index();

    singleSeedPointSet->Clear();
    singleSeedPointSet->InsertPoint(0, point);

    bool unenclosed = this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceAxis, sliceIndex, singleSeedPointSet);

    if (!unenclosed)
    {
      outputPoints->InsertPoint(pointID, point);
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnAnyButtonClicked()
{
  /// Set the focus back to the main window. This is needed so that the keyboard shortcuts
  /// (like 'a' and 'z' for changing slice) keep on working.
  if (QmitkRenderWindow* mainWindow = this->GetView()->GetSelectedRenderWindow())
  {
    mainWindow->setFocus();
  }
}


/**************************************************************
 * Start of: Functions for OK/Reset/Cancel/Close.
 * i.e. finishing a segmentation, and destroying stuff.
 *************************************************************/

//-----------------------------------------------------------------------------
void GeneralSegmentorController::DestroyPipeline()
{
  Q_D(GeneralSegmentorController);

  const mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    bool wasDeleting = d->m_IsDeleting;
    d->m_IsDeleting = true;
    try
    {
      AccessFixedTypeByItk(
          referenceImage,
          ITKDestroyPipeline,
          MITK_ACCESSBYITK_PIXEL_TYPES_SEQ MITK_ACCESSBYITK_COMPOSITE_PIXEL_TYPES_SEQ,
          (3)
      );
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning destroying the ITK pipeline, caused by:" << e.what();
    }
    d->m_IsDeleting = wasDeleting;
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::RemoveWorkingNodes()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  bool wasDeleting = d->m_IsDeleting;
  d->m_IsDeleting = true;

  mitk::ToolManager* toolManager = this->GetToolManager();
  std::vector<mitk::DataNode*> workingNodes = this->GetWorkingNodes();

  // We don't do the first image, as thats the final segmentation.
  for (unsigned int i = 1; i < workingNodes.size(); i++)
  {
    this->GetDataStorage()->Remove(workingNodes[i]);
  }

  std::vector<mitk::DataNode*> noWorkingNodes(0);
  toolManager->SetWorkingData(noWorkingNodes);
  toolManager->ActivateTool(-1);

  d->m_IsDeleting = wasDeleting;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::RestoreInitialSegmentation()
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  assert(segmentationNode);

  mitk::DataNode* seedsNode = this->GetWorkingNode(Tool::SEEDS);
  assert(seedsNode);

  try
  {
    /// Originally, this function cleared the segmentation and the pointset, but
    /// now we rather restore the initial state of the segmentation as it was
    /// when we pressed the Create/restart segmentation button.

//    mitk::Image* segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
//    assert(segmentationImage);
//    AccessFixedDimensionByItk(segmentationImage.GetPointer(), ITKClearImage, 3);
//    segmentationImage->Modified();
//    segmentationNode->Modified();

//    mitk::PointSet::Pointer seeds = this->GetSeeds();
//    seeds->Clear();

    mitk::DataNode* initialSegmentationNode = this->GetWorkingNode(Tool::INITIAL_SEGMENTATION);
    mitk::DataNode* initialSeedsNode = this->GetWorkingNode(Tool::INITIAL_SEEDS);

    segmentationNode->SetData(dynamic_cast<mitk::Image*>(initialSegmentationNode->GetData())->Clone());
    seedsNode->SetData(dynamic_cast<mitk::PointSet*>(initialSeedsNode->GetData())->Clone());

    this->UpdateCurrentSliceContours(false);
    this->UpdateRegionGrowing(false);
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during ITKClearImage, caused by:" << e.what();
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnOKButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  // Set the colour to that which the user selected in the first place.
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  segmentationNode->SetProperty("color", segmentationNode->GetProperty("midas.tmp.selectedcolor"));
  segmentationNode->SetProperty("binaryimage.selectedcolor", segmentationNode->GetProperty("midas.tmp.selectedcolor"));

  /// Apply the thresholds if we are thresholding, and chunk out the contour segments that
  /// do not close any region with seed.
  this->OnCleanButtonClicked();

  this->DestroyPipeline();
  this->RemoveWorkingNodes();
  d->m_GUI->EnableSegmentationWidgets(false);

  this->RequestRenderWindowUpdate();
  mitk::UndoController::GetCurrentUndoModel()->Clear();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnResetButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  QMessageBox::StandardButton returnValue =
      QMessageBox::warning(
        d->m_GUI->GetParent(),
        tr("NiftyMIDAS"),
        tr("Clear all slices ? \n This is not Undo-able! \n Are you sure?"),
        QMessageBox::Yes | QMessageBox::No);

  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->ClearWorkingNodes();
  this->UpdateRegionGrowing();
  this->UpdatePriorAndNext();
  this->UpdateCurrentSliceContours();
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnCancelButtonClicked()
{
  this->DiscardSegmentation();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnViewGetsClosed()
{
  /// TODO this is not invoked at all.
  /// This function was called "ClosePart" before it was moved here from niftkGeneralSegmentorView.
  /// It was not invoked there, either. I leave this here to remind me that the segmentation should
  /// be discarded when the view is closed.
  this->DiscardSegmentation();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::DiscardSegmentation()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  assert(segmentationNode);

  this->DestroyPipeline();
  if (d->m_WasRestarted)
  {
    this->RestoreInitialSegmentation();
    this->RemoveWorkingNodes();
  }
  else
  {
    this->RemoveWorkingNodes();
    this->GetDataStorage()->Remove(segmentationNode);
  }
  d->m_GUI->EnableSegmentationWidgets(false);
  this->GetView()->SetDataManagerSelection(this->GetReferenceNode());
  this->RequestRenderWindowUpdate();
  mitk::UndoController::GetCurrentUndoModel()->Clear();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnRestartButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  QMessageBox::StandardButton returnValue =
      QMessageBox::warning(
        d->m_GUI->GetParent(), tr("NiftyMIDAS"),
        tr("Discard all changes?\nThis is not Undo-able!\nAre you sure?"),
        QMessageBox::Yes | QMessageBox::No);

  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->RestoreInitialSegmentation();
  this->UpdateRegionGrowing();
  this->UpdatePriorAndNext();
  this->UpdateCurrentSliceContours();
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::ClearWorkingNodes()
{
  if (!this->HasWorkingNodes())
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  assert(segmentationNode);

  mitk::Image* segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
  assert(segmentationImage);

  try
  {
    AccessFixedDimensionByItk(segmentationImage, ITKClearImage, 3);
    segmentationImage->Modified();
    segmentationNode->Modified();

    mitk::PointSet::Pointer seeds = this->GetSeeds();
    seeds->Clear();

    this->UpdateCurrentSliceContours(false);
    this->UpdateRegionGrowing(false);
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during ITKClearImage, caused by:" << e.what();
  }
}


/**************************************************************
 * End of: Functions for OK/Reset/Cancel/Close.
 *************************************************************/

/**************************************************************
 * Start of: Functions for simply tool toggling
 *************************************************************/

//-----------------------------------------------------------------------------
void GeneralSegmentorController::ToggleTool(int toolId)
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
bool GeneralSegmentorController::SelectSeedTool()
{
  Q_D(GeneralSegmentorController);

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

  if (this->HasWorkingNodes())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int seedToolId = toolManager->GetToolIdByToolType<SeedTool>();

    if (seedToolId != activeToolId)
    {
      toolManager->ActivateTool(seedToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::SelectDrawTool()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (this->HasWorkingNodes())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int drawToolId = toolManager->GetToolIdByToolType<DrawTool>();

    if (drawToolId != activeToolId)
    {
      toolManager->ActivateTool(drawToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::SelectPolyTool()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (this->HasWorkingNodes())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int polyToolId = toolManager->GetToolIdByToolType<PolyTool>();

    if (polyToolId != activeToolId)
    {
      toolManager->ActivateTool(polyToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::UnselectTools()
{
  Q_D(GeneralSegmentorController);

  if (this->HasWorkingNodes())
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
bool GeneralSegmentorController::SelectViewMode()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (this->HasWorkingNodes())
  {
    mitk::DataNode* segmentationNode = this->GetWorkingNode();
    segmentationNode->SetVisibility(!segmentationNode->IsVisible(0));
  }
  else
  {
    QList<mitk::DataNode::Pointer> selectedNodes = this->GetDataManagerSelection();
    foreach (mitk::DataNode::Pointer selectedNode, selectedNodes)
    {
      selectedNode->SetVisibility(!selectedNode->IsVisible(0));
    }
  }

  this->RequestRenderWindowUpdate();

  return true;
}


/**************************************************************
 * End of: Functions for simply tool toggling
 *************************************************************/

//-----------------------------------------------------------------------------
bool GeneralSegmentorController::CleanSlice()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (this->HasWorkingNodes())
  {
    this->OnCleanButtonClicked();
    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnPropagate3DButtonClicked()
{
  this->DoPropagate(false, true);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnPropagateUpButtonClicked()
{
  this->DoPropagate(true, false);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnPropagateDownButtonClicked()
{
  this->DoPropagate(false, false);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::DoPropagate(bool isUp, bool is3D)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  ImageOrientation orientation = this->GetOrientation();
  itk::Orientation itkOrientation = GetItkOrientation(orientation);

  QString message;

  if (is3D)
  {
    message = "All slices will be over-written";
  }
  else
  {
    QString orientationText;
    QString messageWithOrientation = "All slices %1 the present will be over-written";

    if (isUp)
    {
      if (orientation == IMAGE_ORIENTATION_AXIAL)
      {
        orientationText = "superior to";
      }
      else if (orientation == IMAGE_ORIENTATION_SAGITTAL)
      {
        orientationText = "right of";
      }
      else if (orientation == IMAGE_ORIENTATION_CORONAL)
      {
        orientationText = "anterior to";
      }
      else
      {
        orientationText = "up from";
      }
    }
    else if (!isUp)
    {
      if (orientation == IMAGE_ORIENTATION_AXIAL)
      {
        orientationText = "inferior to";
      }
      else if (orientation == IMAGE_ORIENTATION_SAGITTAL)
      {
        orientationText = "left of";
      }
      else if (orientation == IMAGE_ORIENTATION_CORONAL)
      {
        orientationText = "posterior to";
      }
      else
      {
        orientationText = "up from";
      }
    }

    message = tr(messageWithOrientation.toStdString().c_str()).arg(orientationText);
  }

  QMessageBox::StandardButton returnValue =
      QMessageBox::warning(
        d->m_GUI->GetParent(), tr("NiftyMIDAS"),
        tr("%1.\n"
           "Are you sure?").arg(message),
        QMessageBox::Yes | QMessageBox::No);

  if (returnValue == QMessageBox::No)
  {
    return;
  }

  const mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    mitk::DataNode* segmentationNode = this->GetWorkingNode();
    const mitk::Image* segmentationImage = this->GetWorkingImage();

    if (segmentationImage)
    {

      mitk::DataNode* regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(Tool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image* regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager* toolManager = this->GetToolManager();
      assert(toolManager);

      DrawTool* drawTool = this->GetToolByType<DrawTool>();
      assert(drawTool);

      double lowerThreshold = d->m_GUI->GetLowerThreshold();
      double upperThreshold = d->m_GUI->GetUpperThreshold();
      int sliceAxis = this->GetReferenceImageSliceAxis();
      int sliceIndex = this->GetReferenceImageSliceIndex();
      int sliceUpDirection = this->GetReferenceImageSliceUpDirection();
      if (!is3D && !isUp)
      {
        sliceUpDirection *= -1;
      }
      else if (is3D)
      {
        sliceUpDirection = 0;
      }

      if (sliceIndex == -1 || sliceAxis == -1)
      {
        QMessageBox::warning(
              d->m_GUI->GetParent(),
              "",
              "Cannot determine the axis and index of the current slice.\n"
              "Make sure that a 2D render window is selected. Cannot continue.",
              QMessageBox::Ok);
        return;
      }

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (sliceAxis != -1 && sliceIndex != -1 && itkOrientation != itk::ORIENTATION_UNKNOWN)
      {
        bool wasUpdating = d->m_IsUpdating;
        d->m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKPropagateToRegionGrowingImage, 3,
              (seeds,
               sliceAxis,
               sliceIndex,
               sliceUpDirection,
               lowerThreshold,
               upperThreshold,
               copyOfInputSeeds,
               outputSeeds,
               outputRegion,
               regionGrowingImage  // This is the image we are writing to.
              )
            );

          if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<PolyTool>())
          {
            toolManager->ActivateTool(-1);
          }

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          mitk::Operation* doOp;
          mitk::Operation* undoOp;
          mitk::OperationEvent* opEvent;

          QString message = tr("Propagate: copy region growing");
          OpPropagate::ProcessorPointer processor = OpPropagate::ProcessorType::New();
          doOp = new OpPropagate(OP_PROPAGATE, true, outputRegion, processor);
          undoOp = new OpPropagate(OP_PROPAGATE, false, outputRegion, processor);
          opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
          this->ExecuteOperation(doOp);

          message = tr("Propagate: copy seeds");
          doOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, true, sliceAxis, sliceIndex, outputSeeds);
          undoOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, false, sliceAxis, sliceIndex, copyOfInputSeeds);
          opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
          this->ExecuteOperation(doOp);

          drawTool->ClearWorkingData();
          this->UpdateCurrentSliceContours(false);
          this->UpdateRegionGrowing(false);
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not propagate: Caught mitk::AccessByItkException:" << e.what();
        }
        catch(const itk::ExceptionObject& e)
        {
          MITK_ERROR << "Could not propagate: Caught itk::ExceptionObject:" << e.what();
        }

        d->m_IsUpdating = wasUpdating;
      }
      else
      {
        MITK_ERROR << "Could not propagate: Error sliceAxis=" << sliceAxis << ", sliceIndex=" << sliceIndex << ", orientation=" << itkOrientation << ", direction=" << sliceUpDirection;
      }
    }
  }

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnWipeButtonClicked()
{
  this->DoWipe(0);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnWipePlusButtonClicked()
{
  Q_D(GeneralSegmentorController);

  ImageOrientation orientation = this->GetOrientation();

  QString orientationText;
  QString messageWithOrientation = "All slices %1 the present will be cleared \nAre you sure?";

  if (orientation == IMAGE_ORIENTATION_AXIAL)
  {
    orientationText = "superior to";
  }
  else if (orientation == IMAGE_ORIENTATION_SAGITTAL)
  {
    orientationText = "right of";
  }
  else if (orientation == IMAGE_ORIENTATION_CORONAL)
  {
    orientationText = "anterior to";
  }
  else
  {
    orientationText = "up from";
  }

  QMessageBox::StandardButton returnValue =
      QMessageBox::warning(
        d->m_GUI->GetParent(),
        tr("NiftyMIDAS"),
        tr(messageWithOrientation.toStdString().c_str()).arg(orientationText),
        QMessageBox::Yes | QMessageBox::No);

  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(1);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnWipeMinusButtonClicked()
{
  Q_D(GeneralSegmentorController);

  ImageOrientation orientation = this->GetOrientation();

  QString orientationText;
  QString messageWithOrientation = "All slices %1 the present will be cleared \nAre you sure?";

  if (orientation == IMAGE_ORIENTATION_AXIAL)
  {
    orientationText = "inferior to";
  }
  else if (orientation == IMAGE_ORIENTATION_SAGITTAL)
  {
    orientationText = "left of";
  }
  else if (orientation == IMAGE_ORIENTATION_CORONAL)
  {
    orientationText = "posterior to";
  }
  else
  {
    orientationText = "down from";
  }

  QMessageBox::StandardButton returnValue =
      QMessageBox::warning(
        d->m_GUI->GetParent(),
        tr("NiftyMIDAS"),
        tr(messageWithOrientation.toStdString().c_str()).arg(orientationText),
        QMessageBox::Yes | QMessageBox::No);

  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(-1);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::DoWipe(int direction)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  const mitk::Image* referenceImage = this->GetReferenceImage();
  if (!referenceImage)
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  mitk::Image* segmentationImage = this->GetWorkingImage();

  if (!segmentationImage || !segmentationNode)
  {
    return;
  }

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  int sliceAxis = this->GetReferenceImageSliceAxis();
  int sliceIndex = this->GetReferenceImageSliceIndex();
  int sliceUpDirection = this->GetReferenceImageSliceUpDirection();

  if (sliceIndex == -1 || sliceAxis == -1)
  {
    QMessageBox::warning(
          d->m_GUI->GetParent(),
          "",
          "Cannot determine the axis and index of the current slice.\n"
          "Make sure that a 2D render window is selected. Cannot continue.",
          QMessageBox::Ok);
    return;
  }

  if (direction != 0) // zero means, current slice.
  {
    direction *= sliceUpDirection;
  }

  mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
  mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
  std::vector<int> outputRegion;

  bool wasUpdating = d->m_IsUpdating;
  d->m_IsUpdating = true;

  try
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert(toolManager);

    DrawTool* drawTool = this->GetToolByType<DrawTool>();
    assert(drawTool);

    if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<PolyTool>())
    {
      toolManager->ActivateTool(-1);
    }

    if (direction == 0)
    {
      niftk::CopyPointSets(*seeds, *copyOfInputSeeds);
      niftk::CopyPointSets(*seeds, *outputSeeds);

      AccessFixedDimensionByItk_n(segmentationImage,
          ITKCalculateSliceRegionAsVector, 3,
          (sliceAxis,
           sliceIndex,
           outputRegion
          )
        );

    }
    else
    {
      AccessFixedDimensionByItk_n(segmentationImage, // The binary image = current segmentation
          ITKPreprocessingForWipe, 3,
          (seeds,
           sliceAxis,
           sliceIndex,
           direction,
           copyOfInputSeeds,
           outputSeeds,
           outputRegion
          )
        );
    }

    mitk::UndoStackItem::IncCurrObjectEventId();
    mitk::UndoStackItem::IncCurrGroupEventId();
    mitk::UndoStackItem::ExecuteIncrement();

    OpWipe::ProcessorPointer processor = OpWipe::ProcessorType::New();
    mitk::Operation* doOp = new OpWipe(OP_WIPE, true, sliceAxis, sliceIndex, outputRegion, outputSeeds, processor);
    mitk::Operation* undoOp = new OpWipe(OP_WIPE, false, sliceAxis, sliceIndex, outputRegion, copyOfInputSeeds, processor);
    mitk::OperationEvent* opEvent = new mitk::OperationEvent(this, doOp, undoOp, "Wipe command");
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
    this->ExecuteOperation(doOp);

    drawTool->ClearWorkingData();
    this->UpdateCurrentSliceContours();

    this->RequestRenderWindowUpdate();
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Could not do wipe command: Caught mitk::AccessByItkException:" << e.what();
  }
  catch(const itk::ExceptionObject& e)
  {
    MITK_ERROR << "Could not do wipe command: Caught itk::ExceptionObject:" << e.what();
  }

  d->m_IsUpdating = wasUpdating;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnThresholdApplyButtonClicked()
{
  this->DoThresholdApply(false, false, false);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::DoThresholdApply(
    bool optimiseSeeds,
    bool newSliceEmpty,
    bool newCheckboxStatus)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  const mitk::Image* referenceImage = this->GetReferenceImage();
  if (!referenceImage)
  {
    return;
  }

  ImageOrientation orientation = this->GetOrientation();
  int selectedSliceIndex = this->GetSliceIndex();

  int sliceIndex = this->GetReferenceImageSliceIndex();
  int sliceAxis = this->GetReferenceImageSliceAxis();

  if (sliceIndex == -1 || sliceAxis == -1)
  {
    QMessageBox::warning(
          d->m_GUI->GetParent(),
          "",
          "Cannot determine the axis and index of the current slice.\n"
          "Make sure that a 2D render window is selected. Cannot continue.",
          QMessageBox::Ok);
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  mitk::Image* segmentationImage = this->GetWorkingImage();

  if (!segmentationImage || !segmentationNode)
  {
    return;
  }

  mitk::DataNode* regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(Tool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
  assert(regionGrowingNode);

  mitk::Image* regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
  assert(regionGrowingImage);

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  DrawTool* drawTool = this->GetToolByType<DrawTool>();
  assert(drawTool);

  mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
  mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
  std::vector<int> outputRegion;

  bool wasUpdating = d->m_IsUpdating;
  d->m_IsUpdating = true;

  try
  {
    AccessFixedTypeByItk_n(regionGrowingImage,
        ITKPreprocessingOfSeedsForChangingSlice,
        MITK_ACCESSBYITK_PIXEL_TYPES_SEQ MITK_ACCESSBYITK_COMPOSITE_PIXEL_TYPES_SEQ,
        (3),
        (seeds,
        sliceAxis,
        sliceIndex,
        sliceAxis,
        sliceIndex,
        optimiseSeeds,
        newSliceEmpty,
        copyOfInputSeeds,
        outputSeeds,
        outputRegion
        )
    );

    bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();

    if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<PolyTool>())
    {
      toolManager->ActivateTool(-1);
    }

    mitk::UndoStackItem::IncCurrObjectEventId();
    mitk::UndoStackItem::IncCurrGroupEventId();
    mitk::UndoStackItem::ExecuteIncrement();

    mitk::Operation* doOp;
    mitk::Operation* undoOp;
    mitk::OperationEvent* opEvent;

    std::string orientationName = GetOrientationName(orientation);
    QString message = tr("Apply threshold on %1 slice %2 (image axis: %3, slice: %4)")
        .arg(QString::fromStdString(orientationName)).arg(selectedSliceIndex)
        .arg(sliceAxis).arg(sliceIndex);
    OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
    doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, newCheckboxStatus);
    undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, isThresholdingOn);
    opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
    this->ExecuteOperation(doOp);

    message = tr("Propagate seeds on %1 slice %2 (image axis: %3, slice: %4)")
        .arg(QString::fromStdString(orientationName)).arg(selectedSliceIndex)
        .arg(sliceAxis).arg(sliceIndex);
    doOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, true, sliceAxis, sliceIndex, outputSeeds);
    undoOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, false, sliceAxis, sliceIndex, copyOfInputSeeds);
    opEvent = new mitk::OperationEvent(this, doOp, undoOp, message.toStdString());
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
    this->ExecuteOperation(doOp);

    drawTool->ClearWorkingData();

    bool updateRendering = false;
    this->UpdatePriorAndNext(updateRendering);
    this->UpdateRegionGrowing(updateRendering);
    this->UpdateCurrentSliceContours(updateRendering);

    this->RequestRenderWindowUpdate();
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Could not do threshold apply command: Caught mitk::AccessByItkException:" << e.what();
  }
  catch(const itk::ExceptionObject& e)
  {
    MITK_ERROR << "Could not do threshold apply command: Caught itk::ExceptionObject:" << e.what();
  }

  d->m_IsUpdating = wasUpdating;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnCleanButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();
  int sliceAxis = this->GetReferenceImageSliceAxis();
  int sliceIndex = this->GetReferenceImageSliceIndex();

  if (sliceIndex == -1 || sliceAxis == -1)
  {
    QMessageBox::warning(
          d->m_GUI->GetParent(),
          "",
          "Cannot determine the axis and index of the current slice.\n"
          "Make sure that a 2D render window is selected. Cannot continue.",
          QMessageBox::Ok);
    return;
  }

  if (!isThresholdingOn)
  {
    bool hasUnenclosedSeeds = this->DoesSliceHaveUnenclosedSeeds(isThresholdingOn, sliceAxis, sliceIndex);
    if (hasUnenclosedSeeds)
    {
      QMessageBox::StandardButton returnValue =
          QMessageBox::warning(
            d->m_GUI->GetParent(),
            tr("NiftyMIDAS"),
            tr("There are unenclosed seeds - slice will be wiped\n"
               "Are you sure?"),
            QMessageBox::Yes | QMessageBox::No);

      if (returnValue == QMessageBox::Yes)
      {
        this->DoWipe(0);
      }
      return;
    }
  }

  const mitk::Image* referenceImage = this->GetReferenceImage();
  if (!referenceImage)
  {
    return;
  }

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  mitk::Image* segmentationImage = this->GetWorkingImage();

  if (!segmentationImage || !segmentationNode)
  {
    return;
  }

  if (sliceAxis == -1 || sliceIndex == -1)
  {
    MITK_ERROR << "Could not do clean operation: Error sliceAxis=" << sliceAxis << ", sliceIndex=" << sliceIndex;
    return;
  }

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  PolyTool* polyTool = this->GetToolByType<PolyTool>();
  assert(polyTool);

  DrawTool* drawTool = this->GetToolByType<DrawTool>();
  assert(drawTool);

  mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();

  mitk::ContourModel* polyToolContour = polyTool->GetContour();
  if (polyToolContour && polyToolContour->GetNumberOfVertices() >= 2)
  {
    polyToolContours->AddContourModel(polyToolContour);
  }

  mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingNode(Tool::CONTOURS)->GetData());
  assert(segmentationContours);

  mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingNode(Tool::DRAW_CONTOURS)->GetData());
  assert(drawToolContours);

  mitk::DataNode* regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(Tool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
  assert(regionGrowingNode);

  mitk::Image* regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
  assert(regionGrowingImage);

  double lowerThreshold = d->m_GUI->GetLowerThreshold();
  double upperThreshold = d->m_GUI->GetUpperThreshold();

  mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
  mitk::ContourModelSet::Pointer outputContourSet = mitk::ContourModelSet::New();

  bool wasUpdating = d->m_IsUpdating;
  d->m_IsUpdating = true;

  try
  {
    // Calculate the region of interest for this slice.
    std::vector<int> outputRegion;
    AccessFixedDimensionByItk_n(segmentationImage,
        ITKCalculateSliceRegionAsVector, 3,
        (sliceAxis,
         sliceIndex,
         outputRegion
        )
      );

    if (isThresholdingOn)
    {
      bool useThresholdsWhenCalculatingEnclosedSeeds = false;

      this->DoThresholdApply(false, false, true);

      // Get seeds just on the current slice
      mitk::PointSet::Pointer seedsForCurrentSlice = mitk::PointSet::New();
      this->FilterSeedsToCurrentSlice(
          seeds,
          sliceAxis,
          sliceIndex,
          seedsForCurrentSlice
          );

      // Reduce the list just down to those that are fully enclosed.
      mitk::PointSet::Pointer enclosedSeeds = mitk::PointSet::New();
      this->FilterSeedsToEnclosedSeedsOnSlice(
          seedsForCurrentSlice,
          useThresholdsWhenCalculatingEnclosedSeeds,
          sliceAxis,
          sliceIndex,
          enclosedSeeds
          );

      // Do region growing, using only enclosed seeds.
      AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
          ITKUpdateRegionGrowing, 3,
          (false,
           segmentationImage,
           enclosedSeeds,
           segmentationContours,
           drawToolContours,
           polyToolContours,
           sliceAxis,
           sliceIndex,
           lowerThreshold,
           upperThreshold,
           regionGrowingImage  // This is the image we are writing to.
          )
      );

      // Copy to segmentation image.
      typedef itk::Image<unsigned char, 3> ImageType;
      typedef mitk::ImageToItk< ImageType > ImageToItkType;

      ImageToItkType::Pointer regionGrowingToItk = ImageToItkType::New();
      regionGrowingToItk->SetInput(regionGrowingImage);
      regionGrowingToItk->Update();

      ImageToItkType::Pointer outputToItk = ImageToItkType::New();
      outputToItk->SetInput(segmentationImage);
      outputToItk->Update();

      ITKCopyRegion<unsigned char, 3>(
          regionGrowingToItk->GetOutput(),
          sliceAxis,
          sliceIndex,
          outputToItk->GetOutput()
          );

      regionGrowingToItk = nullptr;
      outputToItk = nullptr;

      // Update the current slice contours, to regenerate cleaned orange contours
      // around just the regions of interest that have a valid seed.
      this->UpdateCurrentSliceContours();
    }
    else
    {
      // Here we are not thresholding.

      // However, we can assume that all seeds are enclosed.
      // If the seeds were not all enclosed, the user received warning earlier,
      // and either abandoned this method, or accepted the warning and wiped the slice.

      if (referenceImage->GetPixelType().GetNumberOfComponents() == 1)
      {
        AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
          ITKUpdateRegionGrowing, 3,
          (false,
          segmentationImage,
          seeds,
          segmentationContours,
          drawToolContours,
          polyToolContours,
          sliceAxis,
          sliceIndex,
          referenceImage->GetStatistics()->GetScalarValueMinNoRecompute(),
          referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute(),
          regionGrowingImage  // This is the image we are writing to.
          )
        );
      }
      else
      {
        // The reference image is the RGB image (read only).
        AccessFixedTypeByItk_n(referenceImage,
          ITKUpdateRegionGrowing,
          MITK_ACCESSBYITK_COMPOSITE_PIXEL_TYPES_SEQ,
          (3),
          (false,
          segmentationImage,
          seeds,
          segmentationContours,
          drawToolContours,
          polyToolContours,
          sliceAxis,
          sliceIndex,
          regionGrowingImage  // This is the image we are writing to.
          )
        );
      }

    }

    // Then create filtered contours for the current slice.
    // So, if we are thresholding, we fit them round the current region growing image,
    // which if we have just used enclosed seeds above, will not include regions defined
    // by a seed and a threshold, but that have not been "applied" yet.

    AccessFixedTypeByItk_n(referenceImage, // The reference image is the grey scale image (read only).
        ITKFilterContours,
        MITK_ACCESSBYITK_PIXEL_TYPES_SEQ MITK_ACCESSBYITK_COMPOSITE_PIXEL_TYPES_SEQ,
        (3),
        (segmentationImage,
          seeds,
          segmentationContours,
          drawToolContours,
          polyToolContours,
          sliceAxis,
          sliceIndex,
          lowerThreshold,
          upperThreshold,
          isThresholdingOn,
          copyOfInputContourSet,
          outputContourSet
        )
    );

    mitk::UndoStackItem::IncCurrObjectEventId();
    mitk::UndoStackItem::IncCurrGroupEventId();
    mitk::UndoStackItem::ExecuteIncrement();

    mitk::Operation* doOp;
    mitk::Operation* undoOp;
    mitk::OperationEvent* opEvent;

    doOp = new OpClean(OP_CLEAN, true, outputContourSet);
    undoOp = new OpClean(OP_CLEAN, false, copyOfInputContourSet);
    opEvent = new mitk::OperationEvent(this, doOp, undoOp, "Clean: Filtering contours");
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
    this->ExecuteOperation(doOp);

    // Then we update the region growing to get up-to-date contours.
    this->UpdateRegionGrowing();

    if (!isThresholdingOn)
    {
      // Then we "apply" this region growing.
      OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
      doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, false);
      undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, false);
      opEvent = new mitk::OperationEvent(this, doOp, undoOp, "Clean: Calculate new image");
      mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(opEvent);
      this->ExecuteOperation(doOp);

      // We should update the current slice contours, as the green contours
      // are the current segmentation that will be applied when we change slice.
      this->UpdateCurrentSliceContours();
    }

    drawTool->Clean(sliceIndex, sliceAxis);

    segmentationImage->Modified();
    segmentationNode->Modified();

    this->RequestRenderWindowUpdate();
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Could not do clean command: Caught mitk::AccessByItkException:" << e.what();
  }
  catch(const itk::ExceptionObject& e)
  {
    MITK_ERROR << "Could not do clean command: Caught itk::ExceptionObject:" << e.what();
  }

  d->m_IsUpdating = wasUpdating;
}


/******************************************************************
 * Start of ExecuteOperation - main method in Undo/Redo framework.
 *
 * Notes: In this method, we update items, using the given
 * operation. We do not know if this is a "Undo" or a "Redo"
 * type of operation. We can set the modified field.
 * But do not be tempted to put things like:
 *
 * this->RequestRenderWindowUpdate();
 *
 * or
 *
 * this->UpdateRegionGrowing() etc.
 *
 * as these methods may be called multiple times during one user
 * operation. So the methods creating the mitk::Operation objects
 * should also be the ones deciding when we update the display.
 ******************************************************************/

void GeneralSegmentorController::ExecuteOperation(mitk::Operation* operation)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasWorkingNodes())
  {
    return;
  }

  if (!operation)
  {
    return;
  }

  const mitk::Image* segmentationImage = this->GetWorkingImage();
  assert(segmentationImage);

  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  assert(segmentationNode);

  const mitk::Image* referenceImage = this->GetReferenceImage();
  assert(referenceImage);

  mitk::Image* regionGrowingImage = this->GetWorkingImage(Tool::REGION_GROWING);
  assert(regionGrowingImage);

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  mitk::DataNode* seedsNode = this->GetWorkingNode(Tool::SEEDS);
  assert(seedsNode);

  switch (operation->GetOperationType())
  {
  case OP_CHANGE_SLICE:
    {
      // Simply to make sure we can switch slice, and undo/redo it.
      OpChangeSliceCommand* op = dynamic_cast<OpChangeSliceCommand*>(operation);
      assert(op);

      mitk::Point3D beforePoint = op->GetBeforePoint();
      mitk::Point3D afterPoint = op->GetAfterPoint();

      mitk::Point3D selectedPoint;

      if (op->IsRedo())
      {
        selectedPoint = afterPoint;
      }
      else
      {
        selectedPoint = beforePoint;
      }

      bool wasChangingSlice = d->m_IsChangingSlice;
      d->m_IsChangingSlice = true;
      this->GetView()->SetSelectedPosition(selectedPoint);
      d->m_IsChangingSlice = wasChangingSlice;

      break;
    }
  case OP_PROPAGATE_SEEDS:
    {
      OpPropagateSeeds* op = dynamic_cast<OpPropagateSeeds*>(operation);
      assert(op);

      mitk::PointSet* newSeeds = op->GetSeeds();
      assert(newSeeds);

      niftk::CopyPointSets(*newSeeds, *seeds);

      seeds->Modified();
      seedsNode->Modified();

      break;
    }
  case OP_RETAIN_MARKS:
    {
      try
      {
        OpRetainMarks* op = dynamic_cast<OpRetainMarks*>(operation);
        assert(op);

        OpRetainMarks::ProcessorType::Pointer processor = op->GetProcessor();
        int fromSlice = op->GetFromSliceIndex();
        int toSlice = op->GetToSliceIndex();
        itk::Orientation orientation = op->GetOrientation();

        typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 3> BinaryImage3DType;
        typedef mitk::ImageToItk< BinaryImage3DType > SegmentationImageToItkType;
        SegmentationImageToItkType::Pointer targetImageToItk = SegmentationImageToItkType::New();
        targetImageToItk->SetInput(segmentationImage);
        targetImageToItk->Update();

        processor->SetSourceImage(targetImageToItk->GetOutput());
        processor->SetDestinationImage(targetImageToItk->GetOutput());
        processor->SetSlices(orientation, fromSlice, toSlice);

        if (op->IsRedo())
        {
          processor->Redo();
        }
        else
        {
          processor->Undo();
        }

        targetImageToItk = nullptr;

        mitk::Image::Pointer outputImage;
        mitk::CastToMitkImage(processor->GetDestinationImage(), outputImage);

        processor->SetSourceImage(nullptr);
        processor->SetDestinationImage(nullptr);

        segmentationNode->SetData(outputImage);
        segmentationNode->Modified();
      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do retain marks: Caught itk::ExceptionObject:" << e.what();
        return;
      }

      break;
    }
  case OP_THRESHOLD_APPLY:
    {
      OpThresholdApply* op = dynamic_cast<OpThresholdApply*>(operation);
      assert(op);

      try
      {
        AccessFixedTypeByItk_n(
            referenceImage,
            ITKPropagateToSegmentationImage,
            MITK_ACCESSBYITK_PIXEL_TYPES_SEQ MITK_ACCESSBYITK_COMPOSITE_PIXEL_TYPES_SEQ,
            (3),
            (segmentationImage,
             regionGrowingImage,
             op
            )
            );

        d->m_GUI->SetThresholdingCheckBoxChecked(op->GetThresholdFlag());
        d->m_GUI->SetThresholdingWidgetsEnabled(op->GetThresholdFlag());

        segmentationImage->Modified();
        segmentationNode->Modified();

        regionGrowingImage->Modified();
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do threshold: Caught mitk::AccessByItkException:" << e.what();
        return;
      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do threshold: Caught itk::ExceptionObject:" << e.what();
        return;
      }

      break;
    }
  case OP_CLEAN:
    {
      try
      {
        OpClean* op = dynamic_cast<OpClean*>(operation);
        assert(op);

        mitk::ContourModelSet* newContours = op->GetContourSet();
        assert(newContours);

        mitk::DataNode* contoursNode = this->GetWorkingNode(Tool::CONTOURS);

        mitk::ContourModelSet* contoursToReplace = dynamic_cast<mitk::ContourModelSet*>(contoursNode->GetData());
        assert(contoursToReplace);

        ContourTool::CopyContourSet(*newContours, *contoursToReplace);
        contoursToReplace->Modified();
        contoursNode->Modified();

        segmentationImage->Modified();
        segmentationNode->Modified();
      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do clean: Caught itk::ExceptionObject:" << e.what();
        return;
      }

      break;
    }
  case OP_WIPE:
    {
      OpWipe* op = dynamic_cast<OpWipe*>(operation);
      assert(op);

      try
      {
        mitk::Image* nonConstSegmentationImage = this->GetWorkingImage(Tool::SEGMENTATION);
        AccessFixedTypeByItk_n(nonConstSegmentationImage,
            ITKDoWipe,
            (unsigned char),
            (3),
              (
                seeds,
                op
              )
            );

        this->UpdateRegionGrowing();

        segmentationImage->Modified();
        segmentationNode->Modified();
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do wipe: Caught mitk::AccessByItkException:" << e.what();
        return;
      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do wipe: Caught itk::ExceptionObject:" << e.what();
        return;
      }

      break;
    }
  case OP_PROPAGATE:
    {
      OpPropagate* op = dynamic_cast<OpPropagate*>(operation);
      assert(op);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, ITKPropagateToSegmentationImage, 3,
              (
                segmentationImage,
                regionGrowingImage,
                op
              )
            );

        segmentationImage->Modified();
        segmentationNode->Modified();
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do propagation: Caught mitk::AccessByItkException:" << e.what();
        return;
      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do propagation: Caught itk::ExceptionObject:" << e.what();
        return;
      }
      break;
    }
  default:;
  }
}

/******************************************************************
 * End of ExecuteOperation - main method in Undo/Redo framework.
 ******************************************************************/

}
