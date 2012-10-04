/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "MIDASGeneralSegmentorView.h"

#include <QButtonGroup>
#include <QMessageBox>
#include <QGridLayout>

#include <mitkProperties.h>
#include <mitkStringProperty.h>
#include <mitkColorProperty.h>
#include <mitkExtractImageFilter.h>
#include <mitkDataNodeObject.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateNot.h>
#include <mitkProperties.h>
#include <mitkRenderingManager.h>
#include <mitkSegTool2D.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkPointSet.h>
#include <mitkGlobalInteraction.h>
#include <mitkTool.h>
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>
#include <mitkImageAccessByItk.h>
#include <mitkSlicedGeometry3D.h>
#include <mitkITKImageImport.h>
#include <mitkGeometry2D.h>
#include <mitkOperationEvent.h>
#include <mitkUndoController.h>
#include <mitkDataStorageUtils.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkContourSet.h>
#include <mitkFocusManager.h>
#include <mitkSegmentationObjectFactory.h>
#include <mitkSurface.h>
#include <itkCommand.h>
#include <itkContinuousIndex.h>

#include "MIDASGeneralSegmentorViewCommands.h"
#include "MIDASGeneralSegmentorViewHelper.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPosnTool.h"
#include "mitkMIDASSeedTool.h"
#include "mitkMIDASPolyTool.h"
#include "mitkMIDASDrawTool.h"
#include "mitkMIDASOrientationUtils.h"

const std::string MIDASGeneralSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor";

const mitk::OperationType MIDASGeneralSegmentorView::OP_PROPAGATE = 9320411;
const mitk::OperationType MIDASGeneralSegmentorView::OP_THRESHOLD_APPLY = 9320412;
const mitk::OperationType MIDASGeneralSegmentorView::OP_WIPE = 9320413;
const mitk::OperationType MIDASGeneralSegmentorView::OP_CLEAN = 9320414;
const mitk::OperationType MIDASGeneralSegmentorView::OP_CHANGE_SLICE = 9320415;
const mitk::OperationType MIDASGeneralSegmentorView::OP_RETAIN_MARKS = 9320416;

/**************************************************************
 * Start of Constructing/Destructing the View stuff.
 *************************************************************/

//-----------------------------------------------------------------------------
MIDASGeneralSegmentorView::MIDASGeneralSegmentorView()
: QmitkMIDASBaseSegmentationFunctionality()
, m_ToolKeyPressStateMachine(NULL)
, m_GeneralControls(NULL)
, m_Layout(NULL)
, m_ContainerForControlsWidget(NULL)
, m_SliceNavigationController(NULL)
, m_SliceNavigationControllerObserverTag(0)
, m_PreviousSliceNumber(0)
, m_FocusManagerObserverTag(0)
, m_IsUpdating(false)
{
  RegisterSegmentationObjectFactory();

  m_Interface = MIDASGeneralSegmentorViewEventInterface::New();
  m_Interface->SetMIDASGeneralSegmentorView(this);
}


//-----------------------------------------------------------------------------
MIDASGeneralSegmentorView::MIDASGeneralSegmentorView(
    const MIDASGeneralSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
MIDASGeneralSegmentorView::~MIDASGeneralSegmentorView()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::MIDASTool* seedTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASSeedTool>()));
  assert(seedTool);
  seedTool->NumberOfSeedsHasChanged -= mitk::MessageDelegate1<MIDASGeneralSegmentorView, int>( this, &MIDASGeneralSegmentorView::OnNumberOfSeedsChanged );

  mitk::MIDASTool* drawTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASDrawTool>()));
  assert(drawTool);
  drawTool->NumberOfSeedsHasChanged -= mitk::MessageDelegate1<MIDASGeneralSegmentorView, int>( this, &MIDASGeneralSegmentorView::OnNumberOfSeedsChanged );

  mitk::MIDASDrawTool* midasDrawTool = dynamic_cast<mitk::MIDASDrawTool*>(drawTool);
  midasDrawTool->ContoursHaveChanged -= mitk::MessageDelegate<MIDASGeneralSegmentorView>( this, &MIDASGeneralSegmentorView::OnContoursChanged );

  mitk::MIDASTool* polyTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
  assert(polyTool);
  polyTool->NumberOfSeedsHasChanged -= mitk::MessageDelegate1<MIDASGeneralSegmentorView, int>( this, &MIDASGeneralSegmentorView::OnNumberOfSeedsChanged );

  mitk::MIDASPolyTool* midasPolyTool = dynamic_cast<mitk::MIDASPolyTool*>(polyTool);
  midasPolyTool->ContoursHaveChanged -= mitk::MessageDelegate<MIDASGeneralSegmentorView>( this, &MIDASGeneralSegmentorView::OnContoursChanged );

  if (m_GeneralControls != NULL)
  {
    delete m_GeneralControls;
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::CreateQtPartControl(QWidget *parent)
{
  this->SetParent(parent);

  if (!m_GeneralControls)
  {
    m_Layout = new QGridLayout(parent);
    m_Layout->setContentsMargins(0,0,0,0);
    m_Layout->setSpacing(0);
    m_Layout->setRowStretch(0, 0);
    m_Layout->setRowStretch(1, 10);
    m_Layout->setRowStretch(2, 0);
    m_Layout->setRowStretch(3, 0);

    m_ContainerForControlsWidget = new QWidget(parent);

    m_GeneralControls = new MIDASGeneralSegmentorViewControlsWidget();
    m_GeneralControls->setupUi(m_ContainerForControlsWidget);

    QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(parent);

    m_Layout->addWidget(m_ContainerForSelectorWidget,         0, 0);
    m_Layout->addWidget(m_ContainerForSegmentationViewWidget, 1, 0);
    m_Layout->addWidget(m_ContainerForToolWidget,             2, 0);
    m_Layout->addWidget(m_ContainerForControlsWidget,         3, 0);

    m_GeneralControls->SetEnableThresholdingWidgets(false);
    m_GeneralControls->SetEnableThresholdingCheckbox(false);

    m_ToolSelector->m_ManualToolSelectionBox->SetDisplayedToolGroups("Seed Draw Poly");
    m_ToolSelector->m_ManualToolSelectionBox->SetLayoutColumns(3);
    m_ToolSelector->m_ManualToolSelectionBox->SetShowNames(true);
    m_ToolSelector->m_ManualToolSelectionBox->SetGenerateAccelerators(false);

    // Create/Connect the state machine which does the key-press shortcuts.
    m_ToolKeyPressStateMachine = mitk::MIDASToolKeyPressStateMachine::New("MIDASKeyPressStateMachine", this);
    mitk::GlobalInteraction::GetInstance()->AddListener( m_ToolKeyPressStateMachine );

    mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
    if (focusManager != NULL)
    {
      itk::SimpleMemberCommand<MIDASGeneralSegmentorView>::Pointer onFocusChangedCommand =
        itk::SimpleMemberCommand<MIDASGeneralSegmentorView>::New();
      onFocusChangedCommand->SetCallbackFunction( this, &MIDASGeneralSegmentorView::OnFocusChanged );

      m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
    }

    // Connect registered tools back to here, so we can do seed processing logic here.
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::MIDASTool* seedTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASSeedTool>()));
    assert(seedTool);
    seedTool->NumberOfSeedsHasChanged += mitk::MessageDelegate1<MIDASGeneralSegmentorView, int>( this, &MIDASGeneralSegmentorView::OnNumberOfSeedsChanged );

    mitk::MIDASTool* drawTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASDrawTool>()));
    assert(drawTool);
    drawTool->NumberOfSeedsHasChanged += mitk::MessageDelegate1<MIDASGeneralSegmentorView, int>( this, &MIDASGeneralSegmentorView::OnNumberOfSeedsChanged );

    mitk::MIDASDrawTool* midasDrawTool = dynamic_cast<mitk::MIDASDrawTool*>(drawTool);
    midasDrawTool->ContoursHaveChanged += mitk::MessageDelegate<MIDASGeneralSegmentorView>( this, &MIDASGeneralSegmentorView::OnContoursChanged );

    mitk::MIDASTool* polyTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
    assert(polyTool);
    polyTool->NumberOfSeedsHasChanged += mitk::MessageDelegate1<MIDASGeneralSegmentorView, int>( this, &MIDASGeneralSegmentorView::OnNumberOfSeedsChanged );

    mitk::MIDASPolyTool* midasPolyTool = dynamic_cast<mitk::MIDASPolyTool*>(polyTool);
    midasPolyTool->ContoursHaveChanged += mitk::MessageDelegate<MIDASGeneralSegmentorView>( this, &MIDASGeneralSegmentorView::OnContoursChanged );

    // Finally do Qt signals/slots.
    this->CreateConnections();
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::CreateConnections()
{
  QmitkMIDASBaseSegmentationFunctionality::CreateConnections();

  if ( m_GeneralControls )
  {
    connect(m_ToolSelector, SIGNAL(ToolSelected(int)), this, SLOT(OnToolSelected(int)));
    connect(m_GeneralControls->m_CleanButton, SIGNAL(pressed()), this, SLOT(OnCleanButtonPressed()) );
    connect(m_GeneralControls->m_WipeButton, SIGNAL(pressed()), this, SLOT(OnWipeButtonPressed()) );
    connect(m_GeneralControls->m_WipePlusButton, SIGNAL(pressed()), this, SLOT(OnWipePlusButtonPressed()) );
    connect(m_GeneralControls->m_WipeMinusButton, SIGNAL(pressed()), this, SLOT(OnWipeMinusButtonPressed()) );
    connect(m_GeneralControls->m_PropUpButton, SIGNAL(pressed()), this, SLOT(OnPropagateUpButtonPressed()) );
    connect(m_GeneralControls->m_PropDownButton, SIGNAL(pressed()), this, SLOT(OnPropagateDownButtonPressed()) );
    connect(m_GeneralControls->m_Prop3DButton, SIGNAL(pressed()), this, SLOT(OnPropagate3DButtonPressed()) );
    connect(m_GeneralControls->m_OKButton, SIGNAL(pressed()), this, SLOT(OnOKButtonPressed()) );
    connect(m_GeneralControls->m_ResetButton, SIGNAL(pressed()), this, SLOT(OnResetButtonPressed()) );
    connect(m_GeneralControls->m_CancelButton, SIGNAL(pressed()), this, SLOT(OnCancelButtonPressed()) );
    connect(m_GeneralControls->m_ThresholdApplyButton, SIGNAL(pressed()), this, SLOT(OnThresholdApplyButtonPressed()) );
    connect(m_GeneralControls->m_ThresholdCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnThresholdCheckBoxToggled(bool)));
    connect(m_GeneralControls->m_SeePriorCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnSeePriorCheckBoxToggled(bool)));
    connect(m_GeneralControls->m_SeeNextCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnSeeNextCheckBoxToggled(bool)));
    connect(m_GeneralControls->m_SeeImageCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnSeeImageCheckBoxPressed(bool)));
    connect(m_GeneralControls->m_ThresholdLowerSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(OnLowerThresholdValueChanged(double)));
    connect(m_GeneralControls->m_ThresholdUpperSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(OnUpperThresholdValueChanged(double)));
    connect(m_ImageAndSegmentationSelector->m_NewSegmentationButton, SIGNAL(clicked()), this, SLOT(OnCreateNewSegmentationButtonPressed()) );
  }
}

/**************************************************************
 * End of Constructing/Destructing the View stuff.
 *************************************************************/

/**************************************************************
 * Start of: Some base class functions we have to implement
 *************************************************************/

//-----------------------------------------------------------------------------
std::string MIDASGeneralSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  bool canRestart = false;

  if (node.IsNotNull() && mitk::IsNodeABinaryImage(node)
      )
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, false);
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
bool MIDASGeneralSegmentorView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  assert(node);
  bool result = false;

  if (IsNodeABinaryImage(node))
  {

    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      mitk::DataNode::Pointer seedsNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEED_POINT_SET_NAME.c_str(), node, true);
      mitk::DataNode::Pointer currentContoursNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::CURRENT_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer regionGrowingImageNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seePriorContoursNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::PRIOR_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seeNextContoursNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::NEXT_CONTOURS_NAME.c_str(), node, true);

      if (seedsNode.IsNotNull()
          && currentContoursNode.IsNotNull()
          && regionGrowingImageNode.IsNotNull()
          && seePriorContoursNode.IsNotNull()
          && seeNextContoursNode.IsNotNull()
          )
      {
        result = true;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType MIDASGeneralSegmentorView::GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  assert(node);
  mitk::ToolManager::DataVectorType result;

  if (IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      mitk::DataNode::Pointer seedsNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEED_POINT_SET_NAME.c_str(), node, true);
      mitk::DataNode::Pointer currentContoursNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::CURRENT_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer regionGrowingImageNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seePriorContoursNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::PRIOR_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seeNextContoursNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::NEXT_CONTOURS_NAME.c_str(), node, true);

      if (seedsNode.IsNotNull()
          && currentContoursNode.IsNotNull()
          && regionGrowingImageNode.IsNotNull()
          && seePriorContoursNode.IsNotNull()
          && seeNextContoursNode.IsNotNull()
          )
      {
        // The order of this list must match the order they were created in.
        result.push_back(node);
        result.push_back(seedsNode);
        result.push_back(currentContoursNode);
        result.push_back(regionGrowingImageNode);
        result.push_back(seePriorContoursNode);
        result.push_back(seeNextContoursNode);
      }
    }
  }
  return result;
}

/**************************************************************
 * End of: Some base class functions we have to implement
 *************************************************************/

/**************************************************************
 * Start of: Functions to create reference data (hidden nodes)
 *************************************************************/

//-----------------------------------------------------------------------------
mitk::DataNode::Pointer MIDASGeneralSegmentorView::CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  mitk::ColorProperty::Pointer col = mitk::ColorProperty::New();
  col->SetColor(r, g, b);

  mitk::ContourSet::Pointer contourSet = mitk::ContourSet::New();
  mitk::DataNode::Pointer contourSetNode = mitk::DataNode::New();

  contourSetNode->SetColor(col->GetColor());
  contourSetNode->SetProperty( "name", mitk::StringProperty::New( name ) );
  contourSetNode->SetBoolProperty("helper object", true);
  contourSetNode->SetBoolProperty("visible", visible);
  contourSetNode->SetProperty("layer", mitk::IntProperty::New(layer));
  contourSetNode->SetData(contourSet);

  this->GetDataStorage()->Add(contourSetNode, segmentationNode);

  return contourSetNode;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer MIDASGeneralSegmentorView::CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::Tool* drawTool = toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASDrawTool>());
  assert(drawTool);

  mitk::ColorProperty::Pointer col = mitk::ColorProperty::New();
  col->SetColor(r, g, b);
  mitk::DataNode::Pointer helperImageNode = drawTool->CreateEmptySegmentationNode( referenceImage, name, col->GetColor());
  helperImageNode->SetColor(col->GetColor());
  helperImageNode->SetProperty("binaryimage.selectedcolor", col);
  helperImageNode->SetBoolProperty("helper object", true);
  helperImageNode->SetBoolProperty("visible", visible);
  helperImageNode->SetProperty("layer", mitk::IntProperty::New(layer));

  this->ApplyDisplayOptions(helperImageNode);
  this->GetDataStorage()->Add(helperImageNode, segmentationNode);

  return helperImageNode;
}


//-----------------------------------------------------------------------------
mitk::DataNode* MIDASGeneralSegmentorView::OnCreateNewSegmentationButtonPressed()
{
  // Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation = NULL;
  bool isRestarting = false;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image* image = this->GetReferenceImageFromToolManager();
  if (image != NULL)
  {
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);


    if (mitk::IsNodeABinaryImage(m_SelectedNode)
        && this->CanStartSegmentationForBinaryNode(m_SelectedNode)
        && !this->IsNodeASegmentationImage(m_SelectedNode)
        )
    {
      newSegmentation =  m_SelectedNode;
      isRestarting = true;
    }
    else
    {
      newSegmentation = QmitkMIDASBaseSegmentationFunctionality::OnCreateNewSegmentationButtonPressed(m_DefaultSegmentationColor);

      // The above method returns NULL if the user exited the colour selection dialog box.
      if (newSegmentation.IsNull())
      {
        return NULL;
      }
    }

    this->WaitCursorOn();

    // Set initial properties.
    newSegmentation->SetProperty("layer", mitk::IntProperty::New(90));
    newSegmentation->SetBoolProperty(mitk::MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), false);

    // Make sure these are up to date, even though we don't use them right now.
    image->GetStatistics()->GetScalarValueMin();
    image->GetStatistics()->GetScalarValueMax();

    // Create the region growing image.
    mitk::DataNode::Pointer regionGrowingImageNode = this->CreateHelperImage(image, newSegmentation, 0,0,1, mitk::MIDASTool::REGION_GROWING_IMAGE_NAME, true, 95);

    // Create all the contours.
    mitk::DataNode::Pointer seeNextNode = this->CreateContourSet(newSegmentation, 0,1,1, mitk::MIDASTool::NEXT_CONTOURS_NAME, false, 96);
    mitk::DataNode::Pointer seePriorNode = this->CreateContourSet(newSegmentation, 1,0,1, mitk::MIDASTool::PRIOR_CONTOURS_NAME, false, 97);
    mitk::DataNode::Pointer currentContours = this->CreateContourSet(newSegmentation, 1,0,0, mitk::MIDASTool::CURRENT_CONTOURS_NAME, true, 98);

    // This creates the point set for the seeds.
    mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
    mitk::DataNode::Pointer pointSetNode = mitk::DataNode::New();
    pointSetNode->SetData( pointSet );
    pointSetNode->SetProperty( "name", mitk::StringProperty::New( mitk::MIDASTool::SEED_POINT_SET_NAME ) );
    pointSetNode->SetProperty( "opacity", mitk::FloatProperty::New( 1 ) );
    pointSetNode->SetProperty( "point line width", mitk::IntProperty::New( 1 ) );
    pointSetNode->SetProperty( "point 2D size", mitk::IntProperty::New( 5 ) );
    pointSetNode->SetBoolProperty("helper object", true);
    pointSetNode->SetBoolProperty("show distant lines", false);
    pointSetNode->SetBoolProperty("show distant points", false);
    pointSetNode->SetBoolProperty("show distances", false);
    pointSetNode->SetProperty("layer", mitk::IntProperty::New(99));
    pointSetNode->SetColor( 1.0, 0, 0 );
    this->GetDataStorage()->Add(pointSetNode, newSegmentation);

    // Make sure these points and contours are not rendered in 3D, as there can be many of them if you "propagate",
    // and furthermore, there seem to be several seg faults rendering contour code in 3D. Haven't investigated yet.
    const mitk::RenderingManager::RenderWindowVector& renderWindows = mitk::RenderingManager::GetInstance()->GetAllRegisteredRenderWindows();
    for (mitk::RenderingManager::RenderWindowVector::const_iterator iter = renderWindows.begin(); iter != renderWindows.end(); ++iter)
    {
      if ( mitk::BaseRenderer::GetInstance((*iter))->GetMapperID() == mitk::BaseRenderer::Standard3D )
      {
        pointSetNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
        seePriorNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
        seeNextNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
        currentContours->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
      }
    }

    // Set working data. See header file, as the order here is critical, and should match the documented order.
    mitk::ToolManager::DataVectorType workingData;
    workingData.push_back(newSegmentation);
    workingData.push_back(pointSetNode);
    workingData.push_back(currentContours);
    workingData.push_back(regionGrowingImageNode);
    workingData.push_back(seePriorNode);
    workingData.push_back(seeNextNode);
    toolManager->SetWorkingData(workingData);

    if (isRestarting)
    {
      this->InitialiseSeedsForWholeVolume();
    }

    // Setup GUI.
    this->m_GeneralControls->SetEnableAllWidgets(true);
    this->m_GeneralControls->SetEnableThresholdingWidgets(false);
    this->m_GeneralControls->SetEnableThresholdingCheckbox(true);
    this->m_GeneralControls->m_ThresholdCheckBox->setChecked(false);
    this->m_GeneralControls->m_SeeImageCheckBox->blockSignals(true);
    this->m_GeneralControls->m_SeeImageCheckBox->setChecked(false);
    this->m_GeneralControls->m_SeeImageCheckBox->blockSignals(false);

    this->WaitCursorOff();

  } // end if we have a reference image

  this->RequestRenderWindowUpdate();

  // And... relax.
  return newSegmentation;
}

/**************************************************************
 * End of: Functions to create reference data (hidden nodes)
 *************************************************************/


/**************************************************************
 * Start of: Utility functions
 *************************************************************/

//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::EnableSegmentationWidgets(bool b)
{
  this->m_GeneralControls->SetEnableAllWidgets(b);
  bool thresholdingIsOn = this->m_GeneralControls->m_ThresholdCheckBox->isChecked();
  this->m_GeneralControls->SetEnableThresholdingWidgets(thresholdingIsOn);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::RecalculateMinAndMaxOfImage()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    double min = referenceImage->GetStatistics()->GetScalarValueMinNoRecompute();
    double max = referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute();
    this->m_GeneralControls->SetLowerAndUpperIntensityRanges(min, max);
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::RecalculateMinAndMaxOfSeedValues()
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
        AccessFixedDimensionByItk_n(referenceImage, ITKRecalculateMinAndMaxOfSeedValues, 3, (*(seeds.GetPointer()), axisNumber, sliceNumber, min, max));
        this->m_GeneralControls->SetSeedMinAndMaxValues(min, max);
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception, so abandoning recalculating min and max of seeds values, due to:" << e.what();
      }
    }
  }
}


//-----------------------------------------------------------------------------
mitk::PointSet* MIDASGeneralSegmentorView::GetSeeds()
{

  mitk::PointSet* result = NULL;

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer pointSetNode = toolManager->GetWorkingData(1);
  if (pointSetNode.IsNotNull())
  {
    result = static_cast<mitk::PointSet*>(pointSetNode->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::CopySeeds(const mitk::PointSet::Pointer inputPoints, mitk::PointSet::Pointer outputPoints)
{
  outputPoints->Clear();
  for (int i = 0; i < inputPoints->GetSize(); i++)
  {
    outputPoints->InsertPoint(i, inputPoints->GetPoint(i));
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::UpdateSegmentationImageVisibility(bool overrideToGlobal)
{
  mitk::ToolManager::DataVectorType nodes = GetWorkingNodesFromToolManager();
  if (nodes.size() > 0 && nodes[0] != NULL)
  {

    /*
     * Work in progress.
     *
     * I'm removing this because:
     *   If we are using the MIDAS editor, then we have renderer specific visibility flags.
     *   If we are using the MITK editor, then we only bother with global flags.
     *   So, at this stage, test with a red outline, leaving the current segmentation as a
     *   green outline and don't mess around with the visibility.
     */
//
//    mitk::DataNode::Pointer segmentationNode = nodes[0];
//
//    if (this->GetPreviouslyFocussedRenderer() != NULL)
//    {
//      mitk::PropertyList* list = segmentationNode->GetPropertyList(this->GetPreviouslyFocussedRenderer());
//      if (list != NULL)
//      {
//        list->DeleteProperty("visible");
//      }
//    }
//
//    if (this->GetCurrentlyFocussedRenderer() != NULL)
//    {
//      if (overrideToGlobal)
//      {
//        mitk::PropertyList* list = segmentationNode->GetPropertyList(GetCurrentlyFocussedRenderer());
//        if (list != NULL)
//        {
//          list->DeleteProperty("visible");
//        }
//      }
//      else
//      {
//        segmentationNode->SetVisibility(false, this->GetCurrentlyFocussedRenderer());
//      }
//    }
  }
}

//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::GenerateOutlineFromBinaryImage(mitk::Image::Pointer image,
    int axisNumber,
    int sliceNumber,
    int projectedSliceNumber,
    mitk::ContourSet::Pointer outputContourSet
    )
{
  try
  {
    AccessFixedTypeByItk_n(image,
        ITKGenerateOutlineFromBinaryImage,
        (unsigned char),
        (3),
        (axisNumber,
         sliceNumber,
         projectedSliceNumber,
         outputContourSet
        )
      );
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Failed in ITKGenerateOutlineFromBinaryImage due to:" << e.what();
    outputContourSet->Initialize();
  }
}


/**************************************************************
 * End of: Utility functions
 *************************************************************/

/**************************************************************
 * Start of: Functions for OK/Reset/Cancel/Close.
 * i.e. finishing a segmentation, and destroying stuff.
 *************************************************************/


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::DestroyPipeline()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk(referenceImage, ITKDestroyPipeline, 3);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning destroying the ITK pipeline, caused by:" << e.what();
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::RemoveWorkingData()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  mitk::ToolManager::DataVectorType workingData = this->GetToolManager()->GetWorkingData();
  if (workingData[0] != NULL)
  {
    for (unsigned int i = 1; i < workingData.size(); i++)
    {
      this->GetDataStorage()->Remove(workingData[i]);
    }

    mitk::ToolManager::DataVectorType emptyWorkingDataArray;
    toolManager->SetWorkingData(emptyWorkingDataArray);
    toolManager->ActivateTool(-1);
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::ClearWorkingData()
{
  mitk::DataNode::Pointer workingData = this->GetToolManager()->GetWorkingData(0);
  if (workingData.IsNotNull())
  {
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(workingData->GetData());
    assert(segmentationImage);

    try
    {
      AccessFixedDimensionByItk(segmentationImage.GetPointer(), ITKClearImage, 3);
      segmentationImage->Modified();
      workingData->Modified();

      mitk::PointSet::Pointer seeds = this->GetSeeds();
      seeds->Clear();

      // This will cause OnSliceNumberChanged to be called, forcing refresh of all contours.
      m_SliceNavigationController->SendSlice();
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception during ITKClearImage, caused by:" << e.what();
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnOKButtonPressed()
{
  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(0);
  assert(segmentationNode);

  this->DestroyPipeline();
  this->RemoveWorkingData();
  this->UpdateSegmentationImageVisibility(true);
  this->EnableSegmentationWidgets(false);
  this->SetReferenceImageSelected();

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::ClosePart()
{
  mitk::Image* segmentationImage = this->GetWorkingImageFromToolManager(0);
  if (segmentationImage != NULL)
  {
    this->OnCancelButtonPressed();
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnCancelButtonPressed()
{
  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(0);
  assert(segmentationNode);

  this->DestroyPipeline();
  this->RemoveWorkingData();
  this->GetDataStorage()->Remove(segmentationNode);
  this->EnableSegmentationWidgets(false);
  this->SetReferenceImageSelected();

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnResetButtonPressed()
{
  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(0);
  assert(segmentationNode);

  this->ClearWorkingData();
  this->UpdateRegionGrowing();
  this->UpdatePriorAndNext();
  this->UpdateCurrentSliceContours();
  this->RequestRenderWindowUpdate();
}


/**************************************************************
 * End of: Functions for OK/Reset/Cancel/Close.
 *************************************************************/

/**************************************************************
 * Start of: Functions for simply tool toggling
 *************************************************************/

//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnToolSelected(int id)
{
  QmitkMIDASBaseSegmentationFunctionality::OnToolSelected(id);
  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::ToggleTool(int toolId)
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
bool MIDASGeneralSegmentorView::SelectSeedTool()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  int toolId = toolManager->GetToolIdByToolType<mitk::MIDASSeedTool>();
  this->ToggleTool(toolId);
  return true;
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::SelectDrawTool()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  int toolId = toolManager->GetToolIdByToolType<mitk::MIDASDrawTool>();
  this->ToggleTool(toolId);
  return true;
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::SelectPolyTool()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  int toolId = toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>();
  this->ToggleTool(toolId);
  return true;
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::UnselectTools()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  toolManager->ActivateTool(-1);
  return true;
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::SelectViewMode()
{
  if (m_GeneralControls->m_SeeImageCheckBox->isChecked())
  {
    this->m_GeneralControls->m_SeeImageCheckBox->setChecked(false);
  }
  else if (!m_GeneralControls->m_SeeImageCheckBox->isChecked())
  {
    this->m_GeneralControls->m_SeeImageCheckBox->setChecked(true);
  }
  return true;
}


/**************************************************************
 * End of: Functions for simply tool toggling
 *************************************************************/

/**************************************************************
 * Start of: The main MIDAS business logic.
 *************************************************************/

//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::InitialiseSeedsForWholeVolume()
{
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

  mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
  mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

  if (workingImage.IsNotNull() && workingNode.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(workingImage,
          ITKInitialiseSeedsForVolume, 3,
          (*seeds,
           axis
          )
        );
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception during ITKInitialiseSeedsForVolume, so have not initialised seeds correctly, caused by:" << e.what();
    }

  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnFocusChanged()
{
  QmitkBaseView::OnFocusChanged();
  mitk::BaseRenderer* currentFocussedRenderer = this->GetCurrentlyFocussedRenderer();

  if (currentFocussedRenderer != NULL)
  {
    // For every new window we get the new windows slice navigation controller.
    if (m_SliceNavigationController.IsNotNull())
    {
      m_SliceNavigationController->RemoveObserver(m_SliceNavigationControllerObserverTag);
    }

    itk::ReceptorMemberCommand<MIDASGeneralSegmentorView>::Pointer onSliceChangedCommand =
      itk::ReceptorMemberCommand<MIDASGeneralSegmentorView>::New();
    onSliceChangedCommand->SetCallbackFunction( this, &MIDASGeneralSegmentorView::OnSliceChanged );
    m_SliceNavigationController = this->GetSliceNavigationController();

    if (m_SliceNavigationController.IsNotNull())
    {
      m_PreviousSliceNumber = -1;

      m_SliceNavigationControllerObserverTag =
          m_SliceNavigationController->AddObserver(
              mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSliceChangedCommand);

      m_SliceNavigationController->SendSlice();
    }

    this->UpdateSegmentationImageVisibility(false);
    this->UpdateCurrentSliceContours();
    this->UpdatePriorAndNext();
    this->OnThresholdCheckBoxToggled(false);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::DoesSliceHaveUnenclosedSeeds()
{
  // Compare this method with UpdateRegionGrowing. We use region growing without
  // threshold limits to work out if a seed can connect to the edge of the image.

  bool sliceDoesHaveUnenclosedSeeds = false;

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

    if (workingImage.IsNotNull() && workingNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      mitk::MIDASPolyTool *polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
      assert(polyTool);

      mitk::ContourSet::Pointer yellowContours = mitk::ContourSet::New();

      mitk::Contour* polyToolContour = polyTool->GetContour();
      if (polyToolContour != NULL && polyToolContour->GetPoints()->Size() > 0)
      {
        yellowContours->AddContour(0, polyToolContour);
      }

      // These contours, stored with the ToolManager represent all the green lines in MIDAS,
      // and come from DrawTool and the current segmentation. When PolyTool is deselected
      // it copies the PolyTool contours (yellow), into this data set, so they appear green.
      mitk::ContourSet* greenContours = static_cast<mitk::ContourSet*>((this->GetWorkingNodesFromToolManager()[2])->GetData());

      double lowerThreshold = this->m_GeneralControls->m_ThresholdLowerSliderWidget->value();
      double upperThreshold = this->m_GeneralControls->m_ThresholdUpperSliderWidget->value();
      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();

      if (axisNumber != -1 && sliceNumber != -1)
      {
        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKSliceDoesHaveUnEnclosedSeeds, 3,
              (*seeds,
               *greenContours,
               *yellowContours,
               lowerThreshold,
               upperThreshold,
               this->m_GeneralControls->m_ThresholdCheckBox->isChecked(),
               axisNumber,
               sliceNumber,
               sliceDoesHaveUnenclosedSeeds
              )
            );
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Caught exception during ITKSliceDoesHaveUnEnclosedSeeds, so will return false, caused by:" << e.what();
        }
      }
    }
  }

  return sliceDoesHaveUnenclosedSeeds;
}



//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::CleanSlice()
{
  this->OnCleanButtonPressed();
  return true;
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnSeePriorCheckBoxToggled(bool b)
{
  mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();
  if (workingNodes.size() > 0)
  {
    if (b)
    {
      this->UpdatePriorAndNext();
    }
    workingNodes[4]->SetVisibility(b);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnSeeNextCheckBoxToggled(bool b)
{
  mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();
  if (workingNodes.size() > 0)
  {
    if (b)
    {
      this->UpdatePriorAndNext();
    }
    workingNodes[5]->SetVisibility(b);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnSeeImageCheckBoxPressed(bool justImage)
{
  mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();
  if (workingNodes.size() > 0)
  {
    workingNodes[0]->SetVisibility(!justImage); // segmentation image
    workingNodes[1]->SetVisibility(!justImage); // seeds
    workingNodes[2]->SetVisibility(!justImage); // green contours from current segmentation
    workingNodes[3]->SetVisibility(!justImage && this->m_GeneralControls->m_ThresholdCheckBox->isChecked()); // region growing
    workingNodes[4]->SetVisibility(!justImage && this->m_GeneralControls->m_SeePriorCheckBox->isChecked()); // see prior
    workingNodes[5]->SetVisibility(!justImage && this->m_GeneralControls->m_SeeNextCheckBox->isChecked()); // see next

    // Also need to check if feedback contour from poly line is off/on.
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    mitk::MIDASPolyTool* polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
    assert(polyTool);
    polyTool->SetFeedbackContourVisible(!justImage);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::UpdatePriorAndNext()
{
  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  int axisNumber = this->GetViewAxis();

  mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
  if (workingImage.IsNotNull())
  {
    mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();

    if (this->m_GeneralControls->m_SeePriorCheckBox->isChecked())
    {
      mitk::ContourSet::Pointer contourSet = static_cast<mitk::ContourSet*>(workingNodes[4]->GetData());
      this->GenerateOutlineFromBinaryImage(workingImage, axisNumber, sliceNumber-1, sliceNumber, contourSet);

      if (contourSet->GetNumberOfContours() > 0)
      {
        workingNodes[4]->Modified();
      }
    }

    if (this->m_GeneralControls->m_SeeNextCheckBox->isChecked())
    {
      mitk::ContourSet::Pointer contourSet = static_cast<mitk::ContourSet*>(workingNodes[5]->GetData());
      this->GenerateOutlineFromBinaryImage(workingImage, axisNumber, sliceNumber+1, sliceNumber, contourSet);

      if (contourSet->GetNumberOfContours() > 0)
      {
        workingNodes[5]->Modified();
      }
    }
  }
} // end function


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::UpdateCurrentSliceContours()
{
  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  int axisNumber = this->GetViewAxis();
  bool updated = false;

  mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
  if (workingImage.IsNotNull())
  {
    mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();
    mitk::ContourSet::Pointer contourSet = static_cast<mitk::ContourSet*>(workingNodes[2]->GetData());
    this->GenerateOutlineFromBinaryImage(workingImage, axisNumber, sliceNumber, sliceNumber, contourSet);

    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::MIDASPolyTool* polyTool = dynamic_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
    assert(polyTool);

    polyTool->ClearData();

    if (contourSet->GetNumberOfContours() > 0)
    {
      workingNodes[2]->Modified();
      updated = true;
    }
  }
  if (updated)
  {
    this->RequestRenderWindowUpdate();
  }
} // end function


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnThresholdCheckBoxToggled(bool b)
{
  mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();
  if (workingNodes.size() > 0)
  {

    this->m_GeneralControls->SetEnableThresholdingWidgets(b);

    // Assuming that if we have ANY registered working data, we have the CORRECT
    // working data. See OnCreateNewSegmentationButtonPressed for the correct order.
    workingNodes[3]->SetVisibility(b);

    if (b)
    {
      this->RecalculateMinAndMaxOfImage();
      this->RecalculateMinAndMaxOfSeedValues();
      this->UpdateRegionGrowing();
    }
  }
  else
  {
    // So, if there is NO working data, we leave the widgets disabled regardless.
    this->m_GeneralControls->SetEnableThresholdingWidgets(false);
  }

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnLowerThresholdValueChanged(double d)
{
  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnUpperThresholdValueChanged(double d)
{
  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::UpdateRegionGrowing()
{

  if (!this->m_GeneralControls->m_ThresholdCheckBox->isChecked())
  {
    mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();
    if (workingNodes.size() >=4)
    {
      workingNodes[3]->SetVisibility(false);
      this->RequestRenderWindowUpdate();
    }
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

    if (workingImage.IsNotNull() && workingNode.IsNotNull())
    {

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      mitk::MIDASPolyTool *polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
      assert(polyTool);

      mitk::ContourSet::Pointer yellowContours = mitk::ContourSet::New();

      mitk::Contour* polyToolContour = polyTool->GetContour();
      if (polyToolContour != NULL && polyToolContour->GetPoints()->Size() > 0)
      {
        yellowContours->AddContour(0, polyToolContour);
      }

      // These contours, stored with the ToolManager represent all the green lines in MIDAS,
      // and come from DrawTool and the current segmentation. When PolyTool is deselected
      // it copies the PolyTool contours (yellow), into this data set, so they appear green.
      mitk::ContourSet* greenContours = static_cast<mitk::ContourSet*>((this->GetWorkingNodesFromToolManager()[2])->GetData());

      double lowerThreshold = this->m_GeneralControls->m_ThresholdLowerSliderWidget->value();
      double upperThreshold = this->m_GeneralControls->m_ThresholdUpperSliderWidget->value();
      bool skipUpdate = !(this->m_GeneralControls->m_ThresholdCheckBox->isChecked());
      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();
      MIDASOrientation tmpOrientation = this->GetOrientationAsEnum();
      itk::ORIENTATION_ENUM orientation = mitk::GetItkOrientation(tmpOrientation);

      if (axisNumber != -1 && sliceNumber != -1 && orientation != itk::ORIENTATION_UNKNOWN)
      {
        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKUpdateRegionGrowing, 3,
              (skipUpdate,
               *seeds,
               *greenContours,
               *yellowContours,
               orientation,
               sliceNumber,
               axisNumber,
               lowerThreshold,
               upperThreshold,
               regionGrowingNode,  // This is the node for the image we are writing to.
               regionGrowingImage  // This is the image we are writing to.
              )
            );

          regionGrowingImage->Modified();
          regionGrowingNode->Modified();
          this->RequestRenderWindowUpdate();
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do region growing: Caught exception, so abandoning ITK pipeline update:" << e.what();
        }
      }
      else
      {
        MITK_ERROR << "Could not do region growing: Error axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << ", orientation=" << orientation << std::endl;
      }
    } // end if working image
  } // end if reference image
} // end function


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnPropagate3DButtonPressed()
{
  bool didPropagation = this->DoPropagate(true, true, true);
  if (didPropagation)
  {
    this->DoPropagate(false, false, true);
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnPropagateUpButtonPressed()
{
  this->DoPropagate(true, true, false);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnPropagateDownButtonPressed()
{
  this->DoPropagate(true, false, false);
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::DoPropagate(bool showWarning, bool isUp, bool is3D)
{

  bool propagationWasPerformed = false;

  if (showWarning)
  {
    QString message("All slices posterior to present will be cleared");
    if (is3D)
    {
      message = "All slices will be cleared";
    }
    else if (isUp)
    {
      message = "All slices anterior to present will be cleared";
    }

    int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                     tr("%1.\n"
                                                        "Are you sure?").arg(message),
                                                     QMessageBox::Yes | QMessageBox::No);
    if (returnValue == QMessageBox::No)
    {
      return propagationWasPerformed;
    }
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {

    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
    if (workingImage.IsNotNull() && workingNode.IsNotNull())
    {

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      double lowerThreshold = this->m_GeneralControls->m_ThresholdLowerSliderWidget->value();
      double upperThreshold = this->m_GeneralControls->m_ThresholdUpperSliderWidget->value();
      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();
      MIDASOrientation tmpOrientation = this->GetOrientationAsEnum();
      itk::ORIENTATION_ENUM orientation = mitk::GetItkOrientation(tmpOrientation);
      int direction = this->GetUpDirection();
      if (!isUp)
      {
        direction *= -1;
      }

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (axisNumber != -1 && sliceNumber != -1 && orientation != itk::ORIENTATION_UNKNOWN)
      {

        m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKPropagateToRegionGrowingImage, 3,
              (*seeds,
               orientation,
               sliceNumber,
               axisNumber,
               direction,
               lowerThreshold,
               upperThreshold,
               *(copyOfInputSeeds.GetPointer()),
               *(outputSeeds.GetPointer()),
               outputRegion,
               regionGrowingNode,  // This is the node for the image we are writing to.
               regionGrowingImage  // This is the image we are writing to.
              )
            );

          // The output of ITKPropagate is a copy of the seeds and an updated region growing image.
          // We need to make the propagation undo-able. This involves adding data to a command, then
          // executing the command in the ExecuteOperationMethod.

          if (!is3D || (is3D && isUp))
          {
            mitk::UndoStackItem::IncCurrObjectEventId();
            mitk::UndoStackItem::IncCurrGroupEventId();
            mitk::UndoStackItem::ExecuteIncrement();
          }

          mitk::OpPropagate::ProcessorPointer processor = mitk::OpPropagate::ProcessorType::New();
          mitk::OpPropagate *doOp = new mitk::OpPropagate(OP_PROPAGATE, true, sliceNumber, axisNumber, outputRegion, outputSeeds, processor, true);
          mitk::OpPropagate *undoOp = new mitk::OpPropagate(OP_PROPAGATE, false, sliceNumber, axisNumber, outputRegion, copyOfInputSeeds, processor, true);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Propagate region growing");
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          ExecuteOperation(doOp);

          // Successful outcome.
          propagationWasPerformed = true;
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not propagate: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not propagate: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;
      }
      else
      {
        MITK_ERROR << "Could not propagate: Error axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << ", orientation=" << orientation << ", direction=" << direction << std::endl;
      }
    }
  }

  if (propagationWasPerformed)
  {
    this->RequestRenderWindowUpdate();
  }
  return propagationWasPerformed;
}



//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnWipeButtonPressed()
{
  this->DoWipe(0);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnWipePlusButtonPressed()
{

  int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                   tr("All slices anterior to present will be cleared.\n"
                                                      "Are you sure?"),
                                                   QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(1);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnWipeMinusButtonPressed()
{
  int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                   tr("All slices posterior to present will be cleared.\n"
                                                      "Are you sure?"),
                                                   QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(-1);
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::DoWipe(int direction)
{
  bool wipeWasPerformed = false;

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {

    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
    if (workingImage.IsNotNull() && workingNode.IsNotNull())
    {

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (axisNumber != -1 && sliceNumber != -1)
      {

        m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(workingImage, // The binary image = current segmentation
              ITKPreProcessingForWipe, 3,
              (*seeds,
               sliceNumber,
               axisNumber,
               direction,
               *(copyOfInputSeeds.GetPointer()),
               *(outputSeeds.GetPointer()),
               outputRegion
              )
            );

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          mitk::OpWipe::ProcessorPointer processor = mitk::OpWipe::ProcessorType::New();
          mitk::OpWipe *doOp = new mitk::OpWipe(OP_WIPE, true, sliceNumber, axisNumber, outputRegion, outputSeeds, processor);
          mitk::OpWipe *undoOp = new mitk::OpWipe(OP_WIPE, false, sliceNumber, axisNumber, outputRegion, copyOfInputSeeds, processor);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Wipe command");
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          ExecuteOperation(doOp);

          // Successful outcome.
          wipeWasPerformed = true;
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do wipe command: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not do wipe command: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;
      }
      else
      {
        MITK_ERROR << "Could not wipe: Error, axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << std::endl;
      }
    }
  }

  if (wipeWasPerformed)
  {
    this->RequestRenderWindowUpdate();
  }
  return wipeWasPerformed;
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnCleanButtonPressed()
{
  bool hasUnEnclosedPoints = this->DoesSliceHaveUnenclosedSeeds();
  if (hasUnEnclosedPoints)
  {
    int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                     tr("There are unenclosed points - slice will be wiped\n"
                                                        "Are you sure?"),
                                                     QMessageBox::Yes | QMessageBox::No);
    if (returnValue == QMessageBox::Yes)
    {
      this->DoWipe(0);
    }
    else if (returnValue == QMessageBox::No)
    {
      return;
    }
  }

  bool cleanWasPerformed = false;

  // If not wiping slice, we are doing proper clean.
  // Take all seeds, region grow without intensity limits = binary image of current regions.
  // If thresholding on = region growing with intensity limits
  // Take red contours, and filter by comparing contours with the output of above two pipelines.

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

    if (workingImage.IsNotNull() && workingNode.IsNotNull())
    {

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      mitk::MIDASPolyTool *polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
      assert(polyTool);

      mitk::ContourSet::Pointer yellowContours = mitk::ContourSet::New();

      mitk::Contour* polyToolContour = polyTool->GetContour();
      if (polyToolContour != NULL && polyToolContour->GetPoints()->Size() > 0)
      {
        yellowContours->AddContour(0, polyToolContour);
      }

      mitk::ContourSet* greenContours = static_cast<mitk::ContourSet*>((this->GetWorkingNodesFromToolManager()[2])->GetData());
      assert(greenContours);

      double lowerThreshold = this->m_GeneralControls->m_ThresholdLowerSliderWidget->value();
      double upperThreshold = this->m_GeneralControls->m_ThresholdUpperSliderWidget->value();
      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();

      mitk::ContourSet::Pointer copyOfInputContourSet = mitk::ContourSet::New();
      mitk::MIDASContourTool::CopyContourSet(*(greenContours), *(copyOfInputContourSet.GetPointer()));

      mitk::ContourSet::Pointer outputContourSet = mitk::ContourSet::New();

      if (axisNumber != -1 && sliceNumber != -1)
      {
        m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKFilterContours, 3,
              (
               *seeds,
               *greenContours,
               *yellowContours,
               axisNumber,
               sliceNumber,
               lowerThreshold,
               upperThreshold,
               *(outputContourSet.GetPointer())
              )
            );

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          mitk::OpClean *doOp = new mitk::OpClean(OP_CLEAN, true, outputContourSet);
          mitk::OpClean *undoOp = new mitk::OpClean(OP_CLEAN, false, copyOfInputContourSet);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Clean");
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          ExecuteOperation(doOp);

          // Successful outcome.
          cleanWasPerformed = true;
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do clean command: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not do clean command: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;

      }
      else
      {
        MITK_ERROR << "Could not do clean operation: Error axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << std::endl;
      }
    }
  }

  if (cleanWasPerformed)
  {
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  // To stop repeated updates triggering from data storage updates.
  if (m_IsUpdating)
  {
    return;
  }

  mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodesFromToolManager();
  if (workingNodes.size() > 0)
  {
    bool relevantNodesWereChanged(false);
    for (unsigned int i = 0; i < workingNodes.size(); i++)
    {
      if (workingNodes[i] != NULL && workingNodes[i] == node)
      {
        relevantNodesWereChanged = true;
        break;
      }
    }
    if (!relevantNodesWereChanged)
    {
      return;
    }

    mitk::DataNode::Pointer segmentationImageNode = workingNodes[0];
    if (segmentationImageNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      if (seeds != NULL && seeds->GetSize() > 0)
      {

        bool contourIsBeingEdited(false);
        if (segmentationImageNode.GetPointer() == node)
        {
          segmentationImageNode->GetBoolProperty(mitk::MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), contourIsBeingEdited);
        }

        if (!contourIsBeingEdited)
        {

          mitk::ToolManager *toolManager = this->GetToolManager();
          assert(toolManager);

          mitk::MIDASPolyTool *polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
          assert(polyTool);

          mitk::Contour* polyToolContour = polyTool->GetContour();

          if (relevantNodesWereChanged || polyToolContour->GetPoints()->Size() > 0)
          {
            this->RecalculateMinAndMaxOfSeedValues();
            this->RecalculateMinAndMaxOfImage();
            this->UpdateRegionGrowing();
          }
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnThresholdApplyButtonPressed()
{
  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();

  // We are calling the DoThresholdApply with the "current" slice number as we do not want it to change slice.
  this->DoThresholdApply(sliceNumber, sliceNumber);
}


//-----------------------------------------------------------------------------
bool MIDASGeneralSegmentorView::DoThresholdApply(int oldSliceNumber, int newSliceNumber)
{
  bool updateWasApplied = false;

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {

    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
    if (workingImage.IsNotNull() && workingNode.IsNotNull())
    {

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      int axisNumber = this->GetViewAxis();

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (axisNumber != -1 && oldSliceNumber != -1)
      {
        m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(regionGrowingImage, // The binary image = current segmentation
              ITKPreProcessingOfSeedsForChangingSlice, 3,
              (*seeds,
               oldSliceNumber,
               axisNumber,
               newSliceNumber,
               *(copyOfInputSeeds.GetPointer()),
               *(outputSeeds.GetPointer()),
               outputRegion
              )
            );

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          bool leaveThresholdingOn(false);
          if (newSliceNumber != oldSliceNumber)
          {
            leaveThresholdingOn = true;
          }

          mitk::OpThresholdApply::ProcessorPointer processor = mitk::OpThresholdApply::ProcessorType::New();
          mitk::OpThresholdApply *doOp = new mitk::OpThresholdApply(OP_THRESHOLD_APPLY, true, oldSliceNumber, axisNumber, outputRegion, outputSeeds, processor, false, leaveThresholdingOn, newSliceNumber);
          mitk::OpThresholdApply *undoOp = new mitk::OpThresholdApply(OP_THRESHOLD_APPLY, false, oldSliceNumber, axisNumber, outputRegion, copyOfInputSeeds, processor, false, true, oldSliceNumber);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Apply threshold");
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          ExecuteOperation(doOp);

          // Successful outcome.
          updateWasApplied = true;

        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do threshold apply command: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not do threshold apply command: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;

      } // end if we have valid axis / slice
    } // end if we have working data
  }// end if we have a reference image

  if (updateWasApplied)
  {
    this->RequestRenderWindowUpdate();
  }
  return updateWasApplied;
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnNumberOfSeedsChanged(int numberOfSeeds)
{
  this->DoUpdateCurrentSlice();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnContoursChanged()
{
  this->DoUpdateCurrentSlice();
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  int previousSlice = m_PreviousSliceNumber;
  int currentSlice = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();

  if (previousSlice == -1)
  {
    previousSlice = currentSlice;
  }
  this->OnSliceNumberChanged(previousSlice, currentSlice);

  m_PreviousSliceNumber = currentSlice;
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::OnSliceNumberChanged(int beforeSliceNumber, int afterSliceNumber)
{
  if (abs(beforeSliceNumber - afterSliceNumber) != 1)
  {
    m_PreviousSliceNumber = afterSliceNumber;
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

    if (workingNode.IsNotNull() && workingImage.IsNotNull())
    {
      int axisNumber = this->GetViewAxis();
      MIDASOrientation tmpOrientation = this->GetOrientationAsEnum();
      itk::ORIENTATION_ENUM orientation = mitk::GetItkOrientation(tmpOrientation);

      if (axisNumber != -1 && beforeSliceNumber != -1 && afterSliceNumber != -1)
      {

        if (this->m_GeneralControls->m_ThresholdCheckBox->isChecked())
        {
          // Compared with OnThresholdApplyButtonPressed, we want to threshold the before slice, and move to the after slice.
          this->DoThresholdApply(beforeSliceNumber, afterSliceNumber);
        }
        else
        {
          std::vector<int> outputRegion;
          mitk::PointSet::Pointer copyOfCurrentSeeds = mitk::PointSet::New();
          mitk::PointSet::Pointer propagatedSeeds = mitk::PointSet::New();

          mitk::PointSet* seeds = this->GetSeeds();
          assert(seeds);

          bool sliceIsEmpty(true);

          m_IsUpdating = true;

          try
          {
            AccessFixedDimensionByItk_n(workingImage,
                ITKPreProcessingOfSeedsForChangingSlice, 3,
                (*seeds,
                 beforeSliceNumber,
                 axisNumber,
                 afterSliceNumber,
                 *(copyOfCurrentSeeds.GetPointer()),
                 *(propagatedSeeds.GetPointer()),
                 outputRegion
                )
              );

            AccessFixedDimensionByItk_n(workingImage,
                ITKSliceIsEmpty, 3,
                (axisNumber,
                 afterSliceNumber,
                 sliceIsEmpty
                )
              );

            if (this->m_GeneralControls->m_RetainMarksCheckBox->isChecked())
            {
              if (!sliceIsEmpty)
              {
                int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                                 tr("The new slice is not empty - retain marks will overwrite slice.\n"
                                                                    "Are you sure?"),
                                                                 QMessageBox::Yes | QMessageBox::No);
                if (returnValue == QMessageBox::No)
                {
                  m_IsUpdating = false;
                  return;
                }
              }

              // Do retain marks
              mitk::OpRetainMarks::ProcessorPointer processor = mitk::OpRetainMarks::ProcessorType::New();
              mitk::OpRetainMarks *doOp = new mitk::OpRetainMarks(OP_RETAIN_MARKS, true, beforeSliceNumber, afterSliceNumber, axisNumber, orientation, outputRegion, propagatedSeeds, processor);
              mitk::OpRetainMarks *undoOp = new mitk::OpRetainMarks(OP_RETAIN_MARKS, false, beforeSliceNumber, afterSliceNumber, axisNumber, orientation, outputRegion, copyOfCurrentSeeds, processor);
              mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Change slice, propagate marks.");
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
              ExecuteOperation(doOp);
            }
            else
            {
              // Move to new slice.
              mitk::OpPropagateSeeds *doOp = new mitk::OpPropagateSeeds(OP_CHANGE_SLICE, true, afterSliceNumber, axisNumber, propagatedSeeds);
              mitk::OpPropagateSeeds *undoOp = new mitk::OpPropagateSeeds(OP_CHANGE_SLICE, false, beforeSliceNumber, axisNumber, copyOfCurrentSeeds);
              mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Change slice, propagate seeds.");
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
              ExecuteOperation(doOp);
            }

          }
          catch(const mitk::AccessByItkException& e)
          {
            MITK_ERROR << "Could not change slice: Caught mitk::AccessByItkException:" << e.what() << std::endl;
          }
          catch( itk::ExceptionObject &err )
          {
            MITK_ERROR << "Could not change slice: Caught itk::ExceptionObject:" << err.what() << std::endl;
          }

          m_IsUpdating = false;

        } // end else, not threshold apply

        this->UpdateRegionGrowing();
        this->UpdatePriorAndNext();
        this->UpdateCurrentSliceContours();
        this->RequestRenderWindowUpdate();

      } // end if, slice number, axis ok.
    } // end have working image
  } // end have reference image
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorView::DoUpdateCurrentSlice()
{
  // To stop repeated calls.
  if (m_IsUpdating)
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

    mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingNode, true);
    assert(regionGrowingNode);

    mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
    assert(regionGrowingImage);

    if (workingImage.IsNotNull() && workingNode.IsNotNull())
    {

      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();
      MIDASOrientation tmpOrientation = this->GetOrientationAsEnum();
      itk::ORIENTATION_ENUM orientation = mitk::GetItkOrientation(tmpOrientation);

      if (axisNumber != -1 && sliceNumber != -1)
      {

        mitk::PointSet* seeds = this->GetSeeds();
        assert(seeds);

        int numberOfSeeds = seeds->GetSize();

        if (numberOfSeeds > 0)
        {

          mitk::ContourSet* greenContours = static_cast<mitk::ContourSet*>((this->GetWorkingNodesFromToolManager()[2])->GetData());
          assert(greenContours);

          mitk::ToolManager *toolManager = this->GetToolManager();
          assert(toolManager);

          mitk::MIDASPolyTool *polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
          assert(polyTool);

          mitk::ContourSet::Pointer yellowContours = mitk::ContourSet::New();

          mitk::Contour* polyToolContour = polyTool->GetContour();
          if (polyToolContour != NULL && polyToolContour->GetPoints()->Size() > 0)
          {
            yellowContours->AddContour(0, polyToolContour);
          }

          bool hasUnEnclosedSeeds = this->DoesSliceHaveUnenclosedSeeds();
          double lowerThreshold = this->m_GeneralControls->m_ThresholdLowerSliderWidget->value();
          double upperThreshold = this->m_GeneralControls->m_ThresholdUpperSliderWidget->value();

          try
          {
            if (hasUnEnclosedSeeds)
            {
              AccessFixedDimensionByItk_n(workingImage,
                  ITKClearSlice, 3,
                  (axisNumber,
                   sliceNumber
                  )
                );
            }
            else if (!hasUnEnclosedSeeds
                     && ((greenContours->GetNumberOfContours() > 0) || (yellowContours->GetNumberOfContours() > 0))
                     && !this->m_GeneralControls->m_ThresholdCheckBox->isChecked()
                    )
            {

              // We do region growing unlimited by thresholds.
              AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
                  ITKUpdateRegionGrowing, 3,
                  (false,
                   *seeds,
                   *greenContours,
                   *yellowContours,
                   orientation,
                   sliceNumber,
                   axisNumber,
                   this->m_GeneralControls->m_ThresholdLowerSliderWidget->minimum(),  // i.e. min and max possible values.
                   this->m_GeneralControls->m_ThresholdLowerSliderWidget->maximum(),
                   workingNode,
                   workingImage
                  )
                );
            }
            else if (this->m_GeneralControls->m_ThresholdCheckBox->isChecked())
            {
              AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
                  ITKUpdateRegionGrowing, 3,
                  (false,
                   *seeds,
                   *greenContours,
                   *yellowContours,
                   orientation,
                   sliceNumber,
                   axisNumber,
                   lowerThreshold,
                   upperThreshold,
                   workingNode,
                   workingImage
                  )
                );

            } // end else if
          }
          catch(const mitk::AccessByItkException& e)
          {
            MITK_ERROR << "Could not update slice: Caught mitk::AccessByItkException:" << e.what() << std::endl;
          }
          catch( itk::ExceptionObject &err )
          {
            MITK_ERROR << "Could not update slice: Caught itk::ExceptionObject:" << err.what() << std::endl;
          }
        } // end if number seeds > 0

        // Make sure we are rendering the latest.
        workingImage->Modified();
        workingNode->Modified();
        this->RequestRenderWindowUpdate();
      }
      else
      {
        MITK_ERROR << "Could not do UpdateCurrentSlice: Error axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << std::endl;
      }
    }
  }
}

/**************************************************************
 * End of: The main MIDAS business logic.
 *************************************************************/

/******************************************************************
 * Start of ExecuteOperation - main method in Undo/Redo framework.
 ******************************************************************/

void MIDASGeneralSegmentorView::ExecuteOperation(mitk::Operation* operation)
{
  if (!operation) return;

  mitk::Image::Pointer segmentedImage = this->GetWorkingImageFromToolManager(0);
  assert(segmentedImage);

  mitk::DataNode::Pointer segmentedNode = this->GetWorkingNodesFromToolManager()[0];
  assert(segmentedNode);

  mitk::Image* referenceImage = this->GetReferenceImageFromToolManager();
  assert(referenceImage);

  mitk::Image* regionGrowingImage = this->GetWorkingImageFromToolManager(3);
  assert(regionGrowingImage);

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  switch (operation->GetOperationType())
  {
  case OP_PROPAGATE:
    {
      mitk::OpPropagate *op = dynamic_cast<mitk::OpPropagate*>(operation);
      assert(op);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, ITKPropagateToSegmentationImage, 3,
              (
                segmentedImage,
                regionGrowingImage,
                seeds,
                op
              )
            );
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do propagation: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        return;
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do propagation: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }
      break;
    }
  case OP_THRESHOLD_APPLY:
    {
      mitk::OpThresholdApply *op = dynamic_cast<mitk::OpThresholdApply*>(operation);
      assert(op);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, ITKPropagateToSegmentationImage, 3,
              (
                segmentedImage,
                regionGrowingImage,
                seeds,
                op
              )
            );

        this->m_GeneralControls->m_ThresholdCheckBox->blockSignals(true);
        this->m_GeneralControls->m_ThresholdCheckBox->setChecked(op->GetThresholdFlag());
        this->m_GeneralControls->m_ThresholdCheckBox->blockSignals(false);

      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do threshold: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        return;
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do threshold: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }
      break;
    }
  case OP_WIPE:
    {
      mitk::OpWipe *op = dynamic_cast<mitk::OpWipe*>(operation);
      assert(op);

      try
      {
        AccessFixedTypeByItk_n(segmentedImage,
            ITKDoWipe,
            (unsigned char),
            (3),
              (
                seeds,
                op
              )
            );
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do wipe: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        return;
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do wipe: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }
      break;
    }
  case OP_CLEAN:
    {
      try
      {
        mitk::OpClean *op = dynamic_cast<mitk::OpClean*>(operation);
        assert(op);

        mitk::ContourSet* newContours = op->GetContourSet();
        assert(newContours);

        mitk::ContourSet* contoursToReplace = static_cast<mitk::ContourSet*>((this->GetWorkingNodesFromToolManager()[2])->GetData());
        assert(contoursToReplace);

        mitk::MIDASContourTool::CopyContourSet(*newContours, *contoursToReplace);
        contoursToReplace->Modified();
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do clean: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }
      break;
    }
  case OP_CHANGE_SLICE:
    {

      mitk::OpPropagateSeeds *op = dynamic_cast<mitk::OpPropagateSeeds*>(operation);
      assert(op);

      mitk::PointSet* newSeeds = op->GetSeeds();
      assert(seeds);

      this->CopySeeds(newSeeds, seeds);

      break;
    }
  case OP_RETAIN_MARKS:
    {
      try
      {
        mitk::OpRetainMarks *op = static_cast<mitk::OpRetainMarks*>(operation);
        assert(op);

        mitk::OpRetainMarks::ProcessorType::Pointer processor = op->GetProcessor();
        bool redo = op->IsRedo();
        int fromSlice = op->GetFromSlice();
        int toSlice = op->GetToSlice();
        itk::ORIENTATION_ENUM orientation = op->GetOrientation();

        typedef mitk::ImageToItk< BinaryImage3DType > SegmentationImageToItkType;
        SegmentationImageToItkType::Pointer targetImageToItk = SegmentationImageToItkType::New();
        targetImageToItk->SetInput(segmentedImage);
        targetImageToItk->Update();

        processor->SetSourceImage(targetImageToItk->GetOutput());
        processor->SetDestinationImage(targetImageToItk->GetOutput());
        processor->SetSlices(orientation, fromSlice, toSlice);

        if (redo)
        {
          processor->Redo();
        }
        else
        {
          processor->Undo();
        }

        mitk::Image::Pointer outputImage = mitk::ImportItkImage( processor->GetDestinationImage());

        segmentedNode->SetData(outputImage);
        segmentedNode->Modified();

        mitk::PointSet* newSeeds = op->GetSeeds();
        assert(seeds);

        this->CopySeeds(newSeeds, seeds);

      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do retain marks: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }
      break;
    }
  default:;
  }

  mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodes();
  assert(workingNodes.size() == 6);

  segmentedImage->Modified();
  seeds->Modified();

  for (unsigned int i = 0; i < workingNodes.size(); i++)
  {
    workingNodes[i]->Modified();
  }
  this->RecalculateMinAndMaxOfSeedValues();
  this->UpdateRegionGrowing();
  this->UpdateCurrentSliceContours();
  this->RequestRenderWindowUpdate();
}

/******************************************************************
 * End of ExecuteOperation - main method in Undo/Redo framework.
 ******************************************************************/

/**************************************************************
 * Start of ITK stuff.
 *************************************************************/

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKFillRegion(
    itk::Image<TPixel, VImageDimension>* itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType &region,
    TPixel fillValue
    )
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  itk::ImageRegionIterator<ImageType> iter(itkImage, region);

  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
  {
    iter.Set(fillValue);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKClearImage(itk::Image<TPixel, VImageDimension>* itkImage)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType largestPossibleRegion = itkImage->GetLargestPossibleRegion();
  this->ITKFillRegion(itkImage, largestPossibleRegion, (TPixel)0);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void MIDASGeneralSegmentorView::ITKCopyImage(
    itk::Image<TPixel, VImageDimension>* input,
    itk::Image<TPixel, VImageDimension>* output
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  itk::ImageRegionIterator<ImageType> inputIterator(input, input->GetLargestPossibleRegion());
  itk::ImageRegionIterator<ImageType> outputIterator(output, output->GetLargestPossibleRegion());

  for (inputIterator.GoToBegin(), outputIterator.GoToBegin();
      !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd();
      ++inputIterator, ++outputIterator
      )
  {
    outputIterator.Set(inputIterator.Get());
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKCalculateSliceRegion(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice,
    typename itk::Image<TPixel, VImageDimension>::RegionType &outputRegion
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::RegionType RegionType;

  RegionType region = itkImage->GetLargestPossibleRegion();
  SizeType regionSize = region.GetSize();
  IndexType regionIndex = region.GetIndex();

  regionSize[axis] = 1;
  regionIndex[axis] = slice;

  outputRegion.SetSize(regionSize);
  outputRegion.SetIndex(regionIndex);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKClearSlice(itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice
    )
{

  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType sliceRegion;
  TPixel pixelValue = 0;

  this->ITKCalculateSliceRegion(itkImage, axis, slice, sliceRegion);
  this->ITKFillRegion(itkImage, sliceRegion, pixelValue);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void MIDASGeneralSegmentorView
::ITKFilterSeedsToCurrentSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int axis,
    int slice,
    mitk::PointSet &outputSeeds
    )
{
  outputSeeds.Clear();

  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  PointType voxelIndexInMillimetres;
  IndexType voxelIndex;

  int pointCounter = 0;
  for (int i = 0; i < inputSeeds.GetSize(); i++)
  {
    voxelIndexInMillimetres = inputSeeds.GetPoint(i);
    itkImage->TransformPhysicalPointToIndex(voxelIndexInMillimetres, voxelIndex);

    if (voxelIndex[axis] == slice)
    {
      outputSeeds.InsertPoint(pointCounter, inputSeeds.GetPoint(i));
      pointCounter++;
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKFilterInputPointSetToExcludeRegionOfInterest(
    itk::Image<TPixel, VImageDimension> *itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType regionOfInterest,
    mitk::PointSet &inputSeeds,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeedsNotInRegionOfInterest
    )
{
  // Copy inputSeeds to outputCopyOfInputSeeds seeds, so that they can be passed on to
  // Redo/Undo framework for Undo purposes. Additionally, copy any input seed that is not
  // within the regionOfInterest. Seed locations are all in millimetres.

  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  PointType voxelIndexInMillimetres;
  IndexType voxelIndex;

  int pointCounter = 0;
  for (int i = 0; i < inputSeeds.GetSize(); i++)
  {
    // Copy every point to outputCopyOfInputSeeds.
    outputCopyOfInputSeeds.InsertPoint(i, inputSeeds.GetPoint(i));

    // Only copy points outside of ROI.
    voxelIndexInMillimetres = inputSeeds.GetPoint(i);
    itkImage->TransformPhysicalPointToIndex(voxelIndexInMillimetres, voxelIndex);

    if (!regionOfInterest.IsInside(voxelIndex))
    {
      outputNewSeedsNotInRegionOfInterest.InsertPoint(pointCounter, inputSeeds.GetPoint(i));
      pointCounter++;
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKRecalculateMinAndMaxOfSeedValues(
    itk::Image<TPixel, VImageDimension>* itkImage,
    mitk::PointSet &inputSeeds,
    int axis,
    int slice,
    double &min,
    double &max
    )
{
  if (inputSeeds.GetSize() == 0)
  {
    min = 0;
    max = 0;
  }
  else
  {
    typedef itk::Image<TPixel, VImageDimension> ImageType;
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::IndexType IndexType;

    PointType millimetreCoordinate;
    IndexType voxelCoordinate;

    mitk::PointSet::Pointer filteredSeeds = mitk::PointSet::New();
    this->ITKFilterSeedsToCurrentSlice(itkImage, inputSeeds, axis, slice, *(filteredSeeds.GetPointer()));

    if (filteredSeeds->GetSize() == 0)
    {
      min = 0;
      max = 0;
    }
    else
    {
      min = std::numeric_limits<double>::max();
      max = std::numeric_limits<double>::min();

      // Iterate through each point, get voxel value, keep running total of min/max.
      for (int i = 0; i < filteredSeeds->GetSize(); i++)
      {
        mitk::PointSet::PointType point = filteredSeeds->GetPoint(i);

        millimetreCoordinate[0] = point[0];
        millimetreCoordinate[1] = point[1];
        millimetreCoordinate[2] = point[2];

        if (itkImage->TransformPhysicalPointToIndex(millimetreCoordinate, voxelCoordinate))
        {
          TPixel voxelValue = itkImage->GetPixel(voxelCoordinate);
          if (voxelValue < min)
          {
            min = voxelValue;
          }
          if (voxelValue > max)
          {
            max = voxelValue;
          }
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
bool MIDASGeneralSegmentorView
::ITKSliceDoesHaveSeeds(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* seeds,
    int axis,
    int slice
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  PointType voxelIndexInMillimetres;
  IndexType voxelIndex;

  bool hasSeeds = false;
  for (int i = 0; i < seeds->GetSize(); i++)
  {
    voxelIndexInMillimetres = seeds->GetPoint(i);
    itkImage->TransformPhysicalPointToIndex(voxelIndexInMillimetres, voxelIndex);

    if (voxelIndex[axis] ==  slice)
    {
      hasSeeds = true;
      break;
    }
  }

  return hasSeeds;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
bool
MIDASGeneralSegmentorView
::ITKSliceIsEmpty(
    itk::Image<TPixel, VImageDimension> *itkImage,
    int axis,
    int slice,
    bool &outputSliceIsEmpty
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexType IndexType;

  RegionType region = itkImage->GetLargestPossibleRegion();
  SizeType regionSize = region.GetSize();
  IndexType regionIndex = region.GetIndex();

  regionSize[axis] = 1;
  regionIndex[axis] = slice;
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  outputSliceIsEmpty = true;

  itk::ImageRegionConstIterator<ImageType> iterator(itkImage, region);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    if (iterator.Get() != 0)
    {
      outputSliceIsEmpty = false;
      break;
    }
  }

  return outputSliceIsEmpty;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKUpdateRegionGrowing(
    itk::Image<TPixel, VImageDimension>* itkImage,  // Grey scale image (read only).
    bool skipUpdate,
    mitk::PointSet &seeds,
    mitk::ContourSet &greenContours,
    mitk::ContourSet &yellowContours,
    itk::ORIENTATION_ENUM orientation,
    int sliceNumber,
    int axisNumber,
    double lowerThreshold,
    double upperThreshold,
    mitk::DataNode::Pointer &outputRegionGrowingNode,
    mitk::Image::Pointer &outputRegionGrowingImage
    )
{

  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  GeneralSegmentorPipeline<TPixel, VImageDimension>* pipeline = NULL;
  GeneralSegmentorPipelineInterface* myPipeline = NULL;

  std::map<std::string, GeneralSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  if (iter == m_TypeToPipelineMap.end())
  {
    pipeline = new GeneralSegmentorPipeline<TPixel, VImageDimension>();
    myPipeline = pipeline;
    m_TypeToPipelineMap.insert(StringAndPipelineInterfacePair(key.str(), myPipeline));
    pipeline->m_ExtractRegionOfInterestFilter->SetInput(itkImage);
  }
  else
  {
    myPipeline = iter->second;
    pipeline = static_cast<GeneralSegmentorPipeline<TPixel, VImageDimension>*>(myPipeline);
  }

  typename ImageToItkType::Pointer regionGrowingToItk = ImageToItkType::New();
  regionGrowingToItk->SetInput(outputRegionGrowingImage);
  regionGrowingToItk->Update();

  GeneralSegmentorPipelineParams params;
  params.m_SliceNumber = sliceNumber;
  params.m_AxisNumber = axisNumber;
  params.m_LowerThreshold = (TPixel)lowerThreshold;
  params.m_UpperThreshold = (TPixel)upperThreshold;
  params.m_Seeds = &seeds;
  params.m_GreenContours = &greenContours;
  params.m_YellowContours = &yellowContours;

  // Update pipeline.
  if (!skipUpdate)
  {
    pipeline->SetParam(params);

    // Setting the pointer to the output image, then calling update on the pipeline
    // will mean that the pipeline will copy its data to the output image.
    pipeline->m_OutputImage = regionGrowingToItk->GetOutput();
    pipeline->Update(params);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKPropagateToRegionGrowingImage
 (itk::Image<TPixel, VImageDimension>* itkImage,
  mitk::PointSet& inputSeeds,
  itk::ORIENTATION_ENUM orientation,
  int sliceNumber,
  int axisNumber,
  int direction,
  double lowerThreshold,
  double upperThreshold,
  mitk::PointSet &outputCopyOfInputSeeds,
  mitk::PointSet &outputNewSeeds,
  std::vector<int> &outputRegion,
  mitk::DataNode::Pointer &outputRegionGrowingNode,
  mitk::Image::Pointer &outputRegionGrowingImage
 )
{
  typedef typename itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef typename itk::Image<unsigned char, VImageDimension> BinaryImageType;
  typedef typename BinaryImageType::PointType BinaryPointType;
  typedef typename itk::Image<unsigned int, VImageDimension>  IntegerImageType;
  typedef typename itk::MIDASRegionGrowingImageFilter<GreyScaleImageType, BinaryImageType, PointSetType> RegionGrowingFilterType;

  // Work out the region of interest that will be affected.
  // We want the region upstream/downstream of the slice of interest.
  typename GreyScaleImageType::RegionType region = itkImage->GetLargestPossibleRegion();
  typename GreyScaleImageType::SizeType regionSize = region.GetSize();
  typename GreyScaleImageType::IndexType regionIndex = region.GetIndex();

  if (direction == 1)
  {
    regionSize[axisNumber] = regionSize[axisNumber] - sliceNumber - 1;
    regionIndex[axisNumber] = sliceNumber + 1;
  }
  else if (direction == -1)
  {
    regionSize[axisNumber] = sliceNumber;
    regionIndex[axisNumber] = 0;
  }
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  outputRegion.push_back(regionIndex[0]);
  outputRegion.push_back(regionIndex[1]);
  outputRegion.push_back(regionIndex[2]);
  outputRegion.push_back(regionSize[0]);
  outputRegion.push_back(regionSize[1]);
  outputRegion.push_back(regionSize[2]);

  // This copies all inputSeeds to outputCopyOfInputSeeds
  // and copies inputSeeds to outputNewSeeds if they are not in region.
  this->ITKFilterInputPointSetToExcludeRegionOfInterest(
      itkImage,
      region,
      inputSeeds,
      outputCopyOfInputSeeds,
      outputNewSeeds
      );

  // Copy MITK seeds to ITK seeds for region growing.
  // Use outputNewSeeds as this list is smaller.
  PointSetPointer itkSeeds = PointSetType::New();
  ConvertMITKSeedsAndAppendToITKSeeds(&outputNewSeeds, itkSeeds);

  // Perform 3D region growing.
  typename RegionGrowingFilterType::Pointer regionGrowingFilter = RegionGrowingFilterType::New();
  regionGrowingFilter->SetInput(itkImage);
  regionGrowingFilter->SetRegionOfInterest(region);
  regionGrowingFilter->SetUseRegionOfInterest(true);
  regionGrowingFilter->SetProjectSeedsIntoRegion(true);
  regionGrowingFilter->SetForegroundValue(1);
  regionGrowingFilter->SetBackgroundValue(0);
  regionGrowingFilter->SetLowerThreshold(lowerThreshold);
  regionGrowingFilter->SetUpperThreshold(upperThreshold);
  regionGrowingFilter->SetSeedPoints(*(itkSeeds.GetPointer()));
  regionGrowingFilter->Update();

  // For each slice in the region growing output, this will calculate
  // new seeds on a per slice basis.
  this->ITKAddNewSeedsToPointSet(
      regionGrowingFilter->GetOutput(),
      region,
      sliceNumber,
      axisNumber,
      outputNewSeeds
      );

  // Write output of region growing directly back to the region growing image
  typedef mitk::ImageToItk< BinaryImageType > ImageToItkType;
  typename ImageToItkType::Pointer outputToItk = ImageToItkType::New();
  outputToItk->SetInput(outputRegionGrowingImage);
  outputToItk->Update();

  typename itk::ImageRegionIterator< BinaryImageType > outputIter(outputToItk->GetOutput(), region);
  typename itk::ImageRegionConstIterator< BinaryImageType > regionGrowingIter(regionGrowingFilter->GetOutput(), region);

  for (outputIter.GoToBegin(), regionGrowingIter.GoToBegin(); !outputIter.IsAtEnd(); ++outputIter, ++regionGrowingIter)
  {
    outputIter.Set(regionGrowingIter.Get());
  }
}


//-----------------------------------------------------------------------------
template <typename TGreyScalePixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKPropagateToSegmentationImage(
    itk::Image<TGreyScalePixel, VImageDimension>* referenceGreyScaleImage,
    mitk::Image* segmentedImage,
    mitk::Image* regionGrowingImage,
    mitk::PointSet* currentSeeds,
    mitk::OpPropagate *op)
{
  typedef typename itk::Image<TGreyScalePixel, VImageDimension> GreyScaleImageType;
  typedef typename itk::Image<unsigned char, VImageDimension> BinaryImageType;

  typedef mitk::ImageToItk< BinaryImageType > ImageToItkType;
  typename ImageToItkType::Pointer segmentedImageToItk = ImageToItkType::New();
  segmentedImageToItk->SetInput(segmentedImage);
  segmentedImageToItk->Update();

  typename ImageToItkType::Pointer regionGrowingImageToItk = ImageToItkType::New();
  regionGrowingImageToItk->SetInput(regionGrowingImage);
  regionGrowingImageToItk->Update();

  mitk::OpPropagate::ProcessorPointer processor = op->GetProcessor();
  std::vector<int> region = op->GetRegion();
  bool redo = op->IsRedo();

  processor->SetSourceImage(regionGrowingImageToItk->GetOutput());
  processor->SetDestinationImage(segmentedImageToItk->GetOutput());
  processor->SetSourceRegionOfInterest(region);
  processor->SetDestinationRegionOfInterest(region);

  mitk::PointSet* outputSeeds = op->GetSeeds();

  if (redo)
  {
    processor->Redo();
  }
  else
  {
    processor->Undo();
  }

  // Update the current point set.
  currentSeeds->Clear();
  for (int i = 0; i < outputSeeds->GetSize(); i++)
  {
    currentSeeds->InsertPoint(i, outputSeeds->GetPoint(i));
  }

  // Clear the region growing image, as this was only used for temporary space.
  typename BinaryImageType::RegionType regionOfInterest = processor->GetSourceRegionOfInterest();
  typename itk::ImageRegionIterator<BinaryImageType> iter(regionGrowingImageToItk->GetOutput(), regionOfInterest);
  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
  {
    iter.Set(0);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKGenerateOutlineFromBinaryImage(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axisNumber,
    int sliceNumber,
    int projectedSliceNumber,
    mitk::ContourSet::Pointer outputContourSet
    )
{
  // NOTE: This function is only meant to be called on binary images,
  // so we are assuming that TPixel is only ever unsigned char.

  // Initialise contour set i.e. clear it.
  outputContourSet->Initialize();

  // Get the largest possible region of the input 3D image.
  Region3DType region = itkImage->GetLargestPossibleRegion();
  Size3DType regionSize = region.GetSize();
  Index3DType regionIndex = region.GetIndex();
  Index3DType projectedRegionIndex = region.GetIndex();

  // Collapse this 3D region down to 2D. So along the specified axis, the size=0.
  regionSize[axisNumber] = 0;
  regionIndex[axisNumber] = sliceNumber;
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  // Also, we setup an index for the "Projected" slice.
  // Here, the terminology "Projected" means which slice we are projecting the contour on to.
  // So, the input sliceNumber controls which slice of data we actually extract, but the "Projected"
  // slice determines the output coordinates of the contours. The contours are "projected" onto that slice.

  projectedRegionIndex[axisNumber] = projectedSliceNumber;

  // To convert 2D voxel coordinates, to 3D coordinates, we need to map the
  // X and Y axes of the 2D image into a 3D vector in the original 3D space.
  Index3DType axes[2];

  // From this point forward, in this method, by X axis we mean, the first axis that
  // is not the through plane direction in the 2D slice. Similarly for Y, the second axis.
  axes[0] = regionIndex;
  axes[1] = regionIndex;
  int axisCounter = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i != axisNumber)
    {
      axes[axisCounter][i] += 1;
      axisCounter++;
    }
  }

  // Calculate the 3D origin of the extracted slice and the projected slice,
  // and hence an offset that must be applied to each coordinate to project it.
  Point3DType originOfSlice;
  itkImage->TransformIndexToPhysicalPoint(regionIndex, originOfSlice);

  Point3DType originOfProjectedSlice;
  Point3DType offsetToProject;
  Point3DType axesInMillimetres[2];

  itkImage->TransformIndexToPhysicalPoint(projectedRegionIndex, originOfProjectedSlice);
  itkImage->TransformIndexToPhysicalPoint(axes[0], axesInMillimetres[0]);
  itkImage->TransformIndexToPhysicalPoint(axes[1], axesInMillimetres[1]);
  for (int i = 0; i < 3; i++)
  {
    axesInMillimetres[0][i] -= originOfSlice[i];
    axesInMillimetres[1][i] -= originOfSlice[i];
    offsetToProject[i] = originOfProjectedSlice[i] - originOfSlice[i];
  }

  // Extract 2D slice, and the contours, using ITK pipelines.
  typename ExtractSliceFilterType::Pointer extractSliceFilter = ExtractSliceFilterType::New();
  extractSliceFilter->SetInput(itkImage);
  extractSliceFilter->SetExtractionRegion(region);

  typename ExtractContoursFilterType::Pointer extractContoursFilter = ExtractContoursFilterType::New();
  extractContoursFilter->SetInput(extractSliceFilter->GetOutput());
  extractContoursFilter->SetContourValue(0.5);

  extractContoursFilter->Update();

  // Now extract the contours, and convert to millimetre coordinates.
  unsigned int numberOfContours = extractContoursFilter->GetNumberOfOutputs();
  for (unsigned int i = 0; i < numberOfContours; i++)
  {
    mitk::Contour::Pointer contour = mitk::Contour::New();
    contour->SetClosed(false);
    contour->SetSelected(false);
    contour->SetWidth(1);
    mitk::Point3D pointInMillimetres;

    typename PathType::Pointer path = extractContoursFilter->GetOutput(i);
    const typename PathType::VertexListType* list = path->GetVertexList();
    typename PathType::VertexType vertex;

    for (unsigned long int j = 0; j < list->Size(); j++)
    {
      vertex = list->ElementAt(j);

      pointInMillimetres[0] = originOfSlice[0] + (vertex[0] * axesInMillimetres[0][0]) + (vertex[1] * axesInMillimetres[1][0]) + offsetToProject[0];
      pointInMillimetres[1] = originOfSlice[1] + (vertex[0] * axesInMillimetres[0][1]) + (vertex[1] * axesInMillimetres[1][1]) + offsetToProject[1];
      pointInMillimetres[2] = originOfSlice[2] + (vertex[0] * axesInMillimetres[0][2]) + (vertex[1] * axesInMillimetres[1][2]) + offsetToProject[2];

      contour->AddVertex(pointInMillimetres);
    }
    outputContourSet->AddContour(i, contour);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void MIDASGeneralSegmentorView
::ITKGetLargestMinimumDistanceSeedLocation(
  itk::Image<TPixel, VImageDimension>* itkImage,
  typename itk::Image<TPixel, VImageDimension>::IndexType &outputSeedIndex,
  int &outputDistance)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::PixelType       PixelType;
  typedef typename ImageType::IndexType       IndexType;
  typedef typename ImageType::SizeType        SizeType;
  typedef typename ImageType::RegionType      RegionType;

  // For the given input image, will return the voxel location that has the
  // largest minimum distance (in x,y direction only) from the edge.
  // For each non-background pixel, we find the minimum distance to the edge for each of the
  // x,y axes in both directions. i.e. we iterate along +x, -x, +y, -y, and find the minimum
  // distance to the edge, and we do this for each non-background voxel, and return the voxel
  // with the largest minimum distance. The input is assumed to be binary ... or more specifically,
  // zero=background and anything else=foreground.

  // In MIDAS terms, this is only called on 2D images, so efficiency is not a problem.
  int workingDistance = -1;
  int minimumDistance = -1;
  int bestDistance = -1;
  IndexType bestIndex;
  IndexType workingIndex;
  IndexType currentIndex;
  PixelType currentPixel = 0;
  RegionType imageRegion = itkImage->GetLargestPossibleRegion();
  SizeType imageSize = imageRegion.GetSize();

  // Work out the largest number of steps we will need along each axis.
  int distanceLimitInVoxels = imageSize[0];
  for (unsigned int i = 1; i < IndexType::GetIndexDimension(); i++)
  {
    distanceLimitInVoxels = std::max((int)distanceLimitInVoxels, (int)imageSize[i]);
  }

  // Iterate through each pixel in image.
  itk::ImageRegionConstIteratorWithIndex<ImageType> imageIterator(itkImage, imageRegion);
  for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator)
  {
    // Check that the current pixel is not background.
    currentPixel = imageIterator.Get();
    if (currentPixel != 0)
    {
      currentIndex = imageIterator.GetIndex();
      minimumDistance = distanceLimitInVoxels;

      // If this is the first non-zero voxel, assume this is the best so far.
      if (bestDistance == -1)
      {
        bestDistance = 0;
        bestIndex = currentIndex;
      }

      // and for each of the image axes.
      for (unsigned int i = 0; i < IndexType::GetIndexDimension(); i++)
      {
        // Only iterate over the x,y,z, axis if the size of the axis is > 1
        if (imageSize[i] > 1)
        {
          // For each direction +/-
          for (int j = -1; j <= 1; j+=2)
          {
            // Reset the workingIndex to the current position.
            workingIndex = currentIndex;
            workingDistance = 0;
            do
            {
              // Calculate an offset.
              workingDistance++;
              workingIndex[i] = currentIndex[i] + j*workingDistance;

            } // And check we are still in the image on non-background.
            while (imageRegion.IsInside(workingIndex)
                   && itkImage->GetPixel(workingIndex) > 0
                   && workingDistance < minimumDistance);

            minimumDistance = workingDistance;
          } // end for j
        } // end if image size > 1.
      } // end for i

      // If this voxel has a larger minimum distance, than the bestDistance so far, we chose this one.
      if (minimumDistance > bestDistance)
      {
        bestIndex = currentIndex;
        bestDistance = minimumDistance;
      }
    }
  }
  // Output the largest minimumDistance and the corresponding voxel location.
  outputSeedIndex = bestIndex;
  outputDistance = bestDistance;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKAddNewSeedsToPointSet(
    itk::Image<TPixel, VImageDimension> *itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType region,
    int sliceNumber,
    int axisNumber,
    mitk::PointSet &outputNewSeeds
    )
{
  typedef typename itk::Image<TPixel, VImageDimension>        BinaryImageType;
  typedef typename BinaryImageType::PointType                 BinaryPointType;
  typedef typename BinaryImageType::IndexType                 BinaryIndexType;
  typedef typename itk::Image<unsigned int, VImageDimension>  IntegerImageType;
  typedef typename itk::ExtractImageFilter<BinaryImageType, BinaryImageType> ExtractImageFilterType;
  typedef typename itk::ConnectedComponentImageFilter<BinaryImageType, IntegerImageType> ConnectedComponentFilterType;
  typedef typename itk::BinaryThresholdImageFilter<IntegerImageType, BinaryImageType> BinaryThresholdFilterType;

  // Some working data.
  typename IntegerImageType::PixelType voxelValue = 0;
  BinaryIndexType voxelIndex;
  BinaryPointType voxelIndexInMillimetres;
  Size3DType regionSize = region.GetSize();
  Index3DType regionIndex = region.GetIndex();

  // We are going to repeatedly extract each slice, and calculate new seeds on a per slice basis.
  typename ExtractImageFilterType::Pointer extractSliceFilter = ExtractImageFilterType::New();
  extractSliceFilter->SetInput(itkImage);

  typename ConnectedComponentFilterType::Pointer connectedComponentsFilter = ConnectedComponentFilterType::New();
  connectedComponentsFilter->SetInput(extractSliceFilter->GetOutput());
  connectedComponentsFilter->SetBackgroundValue(0);
  connectedComponentsFilter->SetFullyConnected(true);

  typename BinaryThresholdFilterType::Pointer binaryThresholdFilter = BinaryThresholdFilterType::New();
  binaryThresholdFilter->SetInput(connectedComponentsFilter->GetOutput());
  binaryThresholdFilter->SetInsideValue(1);
  binaryThresholdFilter->SetOutsideValue(0);

  for (unsigned int i = 0; i < regionSize[axisNumber]; i++)
  {
    typename BinaryImageType::RegionType perSliceRegion;
    typename BinaryImageType::SizeType   perSliceRegionSize;
    typename BinaryImageType::IndexType  perSliceRegionStartIndex;

    perSliceRegionSize = regionSize;
    perSliceRegionStartIndex = regionIndex;

    perSliceRegionSize[axisNumber] = 1;
    perSliceRegionStartIndex[axisNumber] = regionIndex[axisNumber] + i;

    perSliceRegion.SetSize(perSliceRegionSize);
    perSliceRegion.SetIndex(perSliceRegionStartIndex);

    // Extract slice, and get connected components.
    extractSliceFilter->SetExtractionRegion(perSliceRegion);
    connectedComponentsFilter->UpdateLargestPossibleRegion();

    // For each distinct region in 2D, we calculate a new seed.
    typename IntegerImageType::Pointer ccImage = connectedComponentsFilter->GetOutput();
    typename itk::ImageRegionConstIteratorWithIndex<IntegerImageType> ccImageIterator(ccImage, ccImage->GetLargestPossibleRegion());
    std::set<typename IntegerImageType::PixelType> setOfLabels;

    int notUsed;
    mitk::PointSet::PointType point;
    int numberOfPoints = outputNewSeeds.GetSize();

    for (ccImageIterator.GoToBegin(); !ccImageIterator.IsAtEnd(); ++ccImageIterator)
    {
      voxelValue = ccImageIterator.Get();

      if (voxelValue != 0 && setOfLabels.find(voxelValue) == setOfLabels.end())
      {
        setOfLabels.insert(voxelValue);

        // Use the threshold filter to extract just the current label.
        binaryThresholdFilter->SetLowerThreshold(voxelValue);
        binaryThresholdFilter->SetUpperThreshold(voxelValue);
        binaryThresholdFilter->UpdateLargestPossibleRegion();

        // Now, with that binary image, calculate a new seed position.
        this->ITKGetLargestMinimumDistanceSeedLocation(binaryThresholdFilter->GetOutput(), voxelIndex, notUsed);

        // And convert that seed position to a 3D point.
        itkImage->TransformIndexToPhysicalPoint(voxelIndex, point);
        outputNewSeeds.InsertPoint(numberOfPoints, point);
        numberOfPoints++;
      } // end if new label
    } // end for each label
  } // end for each slice
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKPreProcessingOfSeedsForChangingSlice(
    itk::Image<TPixel, VImageDimension> *itkImage, // Note: the itkImage input should be the binary region growing image.
    mitk::PointSet &inputSeeds,
    int sliceNumber,
    int axisNumber,
    int newSliceNumber,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeeds,
    std::vector<int> &outputRegion
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> BinaryImageType;

  // Work out the region of interest that will be affected.
  // In contrast to PropagateToRegionGrowingImageUsingITK, this is JUST the current slice.
  typename BinaryImageType::RegionType region = itkImage->GetLargestPossibleRegion();
  typename BinaryImageType::SizeType regionSize = region.GetSize();
  typename BinaryImageType::IndexType regionIndex = region.GetIndex();
  regionSize[axisNumber] = 1;
  regionIndex[axisNumber] = sliceNumber;
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  outputRegion.push_back(regionIndex[0]);
  outputRegion.push_back(regionIndex[1]);
  outputRegion.push_back(regionIndex[2]);
  outputRegion.push_back(regionSize[0]);
  outputRegion.push_back(regionSize[1]);
  outputRegion.push_back(regionSize[2]);

  if (sliceNumber != newSliceNumber)
  {

    // If newSliceNumber != sliceNumber we are moving to a new slice.
    //
    // If that is the case, we propagate seeds forward, without calculating
    // new seed positions using connected components. i.e. they change slice,
    // and don't jump to a new position that looks like the centre of mass.

    this->CopySeeds(
        &inputSeeds,
        &outputCopyOfInputSeeds
        );

    this->ITKPropagateSeedsToNewSlice(
        itkImage,
        &inputSeeds,
        &outputNewSeeds,
        axisNumber,
        sliceNumber,
        newSliceNumber
        );
  }
  else
  {
    // This is the default "Threshold Apply".

    // We take a complete copy of the input seeds, and copy any seeds not in the current slice
    // as these seeds in the current slice will be overwritten in AddNewSeedsToPointSet.
    this->ITKFilterInputPointSetToExcludeRegionOfInterest(
        itkImage,
        region,
        inputSeeds,
        outputCopyOfInputSeeds,
        outputNewSeeds
        );

    // Here we calculate new seeds based on the connected component analysis - i.e. 1 seed per region.
    this->ITKAddNewSeedsToPointSet(
        itkImage,
        region,
        sliceNumber,
        axisNumber,
        outputNewSeeds
        );

  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKPreProcessingForWipe(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int sliceNumber,
    int axisNumber,
    int direction,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeeds,
    std::vector<int> &outputRegion
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;

  // Work out the region of interest that will be affected.
  typename ImageType::RegionType region = itkImage->GetLargestPossibleRegion();
  typename ImageType::SizeType regionSize = region.GetSize();
  typename ImageType::IndexType regionIndex = region.GetIndex();

  if (direction == 0)
  {
    // Single slice
    regionSize[axisNumber] = 1;
    regionIndex[axisNumber] = sliceNumber;
  }
  else if (direction == 1)
  {
    // All anterior
    regionSize[axisNumber] = regionSize[axisNumber] - sliceNumber - 1;
    regionIndex[axisNumber] = sliceNumber + 1;
  }
  else if (direction == -1)
  {
    // All posterior
    regionSize[axisNumber] = sliceNumber;
    regionIndex[axisNumber] = 0;
  }
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  outputRegion.push_back(regionIndex[0]);
  outputRegion.push_back(regionIndex[1]);
  outputRegion.push_back(regionIndex[2]);
  outputRegion.push_back(regionSize[0]);
  outputRegion.push_back(regionSize[1]);
  outputRegion.push_back(regionSize[2]);

  // We take a complete copy of the input seeds, and copy any seeds not in the current slice
  // as these seeds in the current slice will be overwritten in AddNewSeedsToPointSet.
  this->ITKFilterInputPointSetToExcludeRegionOfInterest(
      itkImage,
      region,
      inputSeeds,
      outputCopyOfInputSeeds,
      outputNewSeeds
      );
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKDoWipe(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* currentSeeds,
    mitk::OpWipe *op
    )
{
  // Assuming we are only called for the unsigned char, current segmentation image.
  typedef typename itk::Image<TPixel, VImageDimension> BinaryImageType;

  mitk::OpWipe::ProcessorPointer processor = op->GetProcessor();
  std::vector<int> region = op->GetRegion();
  bool redo = op->IsRedo();

  processor->SetWipeValue(0);
  processor->SetDestinationImage(itkImage);
  processor->SetDestinationRegionOfInterest(region);

  mitk::PointSet* outputSeeds = op->GetSeeds();

  if (redo)
  {
    processor->Redo();
  }
  else
  {
    processor->Undo();
  }

  // Update the current point set.
  currentSeeds->Clear();
  for (int i = 0; i < outputSeeds->GetSize(); i++)
  {
    currentSeeds->InsertPoint(i, outputSeeds->GetPoint(i));
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void MIDASGeneralSegmentorView
::ITKSliceDoesHaveUnEnclosedSeeds(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &seeds,
    mitk::ContourSet &greenContours,
    mitk::ContourSet &yellowContours,
    double lowerThreshold,
    double upperThreshold,
    bool doRegionGrowing,
    int axis,
    int slice,
    bool &sliceDoesHaveUnenclosedSeeds
    )
{
  sliceDoesHaveUnenclosedSeeds = false;

  // Note input image should be 3D grey scale.
  typedef itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, VImageDimension> BinaryImageType;

  // Filter seeds to only use ones on current slice.
  mitk::PointSet::Pointer seedsForThisSlice = mitk::PointSet::New();
  this->ITKFilterSeedsToCurrentSlice(itkImage, seeds, axis, slice, *(seedsForThisSlice.GetPointer()));

  // First we run region growing without limits to check if we have seeds inside non-enclosing green contours.
  GeneralSegmentorPipelineParams params;
  params.m_SliceNumber = slice;
  params.m_AxisNumber = axis;
  params.m_LowerThreshold = std::numeric_limits<TPixel>::min();
  params.m_UpperThreshold = std::numeric_limits<TPixel>::max();
  params.m_Seeds = seedsForThisSlice;
  params.m_GreenContours = &greenContours;
  params.m_YellowContours = &yellowContours;

  GeneralSegmentorPipeline<TPixel, VImageDimension> pipeline = GeneralSegmentorPipeline<TPixel, VImageDimension>();
  pipeline.m_UseOutput = false;  // don't export the output of this pipeline to an output image, as we are not providing one.
  pipeline.m_ExtractRegionOfInterestFilter->SetInput(itkImage);
  pipeline.SetParam(params);
  pipeline.Update(params);

  // Check the output, to see if we have seeds inside non-enclosing green contours.
  sliceDoesHaveUnenclosedSeeds = this->ITKImageHasNonZeroEdgePixels<
      mitk::Tool::DefaultSegmentationDataType, VImageDimension>
      (pipeline.m_RegionGrowingFilter->GetOutput());

  // If the thresholding checkbox is on we should check the region growing output as well.
  if (doRegionGrowing && !sliceDoesHaveUnenclosedSeeds)
  {
    params.m_LowerThreshold = lowerThreshold;
    params.m_UpperThreshold = upperThreshold;
    pipeline.SetParam(params);
    pipeline.Update(params);

    sliceDoesHaveUnenclosedSeeds = this->ITKImageHasNonZeroEdgePixels<
      mitk::Tool::DefaultSegmentationDataType, VImageDimension>
      (pipeline.m_RegionGrowingFilter->GetOutput());
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void MIDASGeneralSegmentorView
::ITKFilterContours(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &seeds,
    mitk::ContourSet &greenContours,
    mitk::ContourSet &yellowContours,
    int axis,
    int slice,
    double lowerThreshold,
    double upperThreshold,
    mitk::ContourSet &outputContours
    )
{
  // Input contour set could be empty, so nothing to filter.
  if (greenContours.GetNumberOfContours() == 0)
  {
    return;
  }

  // Note input image should be 3D grey scale.
  typedef itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, VImageDimension> BinaryImageType;
  typedef itk::ContinuousIndex<double, VImageDimension> ContinuousIndexType;
  typedef typename BinaryImageType::IndexType IndexType;
  typedef typename BinaryImageType::SizeType SizeType;
  typedef typename BinaryImageType::RegionType RegionType;
  typedef typename BinaryImageType::PointType PointType;

  // Filter seeds to only use ones on current slice.
  mitk::PointSet::Pointer seedsForThisSlice = mitk::PointSet::New();
  this->ITKFilterSeedsToCurrentSlice(itkImage, seeds, axis, slice, *(seedsForThisSlice.GetPointer()));

  // First we run region growing without limits to get regions within green contours.
  GeneralSegmentorPipelineParams greenParams;
  greenParams.m_SliceNumber = slice;
  greenParams.m_AxisNumber = axis;
  greenParams.m_LowerThreshold = std::numeric_limits<TPixel>::min();
  greenParams.m_UpperThreshold = std::numeric_limits<TPixel>::max();
  greenParams.m_Seeds = seedsForThisSlice;
  greenParams.m_GreenContours = &greenContours;
  greenParams.m_YellowContours = &yellowContours;

  GeneralSegmentorPipeline<TPixel, VImageDimension> greenPipeline = GeneralSegmentorPipeline<TPixel, VImageDimension>();
  greenPipeline.m_UseOutput = false;  // don't export the output of this pipeline to an output image, as we are not providing one.
  greenPipeline.m_ExtractRegionOfInterestFilter->SetInput(itkImage);
  greenPipeline.SetParam(greenParams);
  greenPipeline.Update(greenParams);

  // Then we run region growing with limits to get regions within blue contours.
  GeneralSegmentorPipelineParams blueParams;
  blueParams.m_SliceNumber = slice;
  blueParams.m_AxisNumber = axis;
  blueParams.m_LowerThreshold = lowerThreshold;
  blueParams.m_UpperThreshold = upperThreshold;
  blueParams.m_Seeds = seedsForThisSlice;
  blueParams.m_GreenContours = &greenContours;
  blueParams.m_YellowContours = &yellowContours;

  GeneralSegmentorPipeline<TPixel, VImageDimension> bluePipeline = GeneralSegmentorPipeline<TPixel, VImageDimension>();
  bluePipeline.m_UseOutput = false;  // don't export the output of this pipeline to an output image, as we are not providing one.
  bluePipeline.m_ExtractRegionOfInterestFilter->SetInput(itkImage);
  bluePipeline.SetParam(blueParams);
  bluePipeline.Update(blueParams);

  // Now calculate filtered contours, we want to get rid of any contours that are not near a region.
  // NOTE: Poly line contours (yellow) contours are not cleaned.
  unsigned int size = 0;
  mitk::Point3D pointInContour;
  int contourNumber = 0;
  PointType pointInMillimetres;
  ContinuousIndexType continuousVoxelIndex;
  IndexType voxelIndex;

  mitk::ContourSet::ContourVectorType contourVec = greenContours.GetContours();
  mitk::ContourSet::ContourIterator contourIt = contourVec.begin();
  mitk::Contour::Pointer firstContour = (*contourIt).second;

  outputContours.Initialize();
  mitk::Contour::Pointer outputContour = mitk::Contour::New();
  mitk::MIDASContourTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));

  while ( contourIt != contourVec.end() )
  {
    mitk::Contour::Pointer nextContour = (mitk::Contour::Pointer) (*contourIt).second;
    mitk::Contour::PointsContainerPointer nextPoints = nextContour->GetPoints();

    size = nextContour->GetNumberOfPoints();
    for (unsigned int i = 0; i < size; i++)
    {
      pointInContour = nextPoints->GetElement(i);
      for (unsigned int j = 0; j < SizeType::GetSizeDimension(); j++)
      {
        pointInMillimetres[j] = pointInContour[j];
      }

      itkImage->TransformPhysicalPointToContinuousIndex(pointInMillimetres, continuousVoxelIndex);

      for (unsigned int j = 0; j < SizeType::GetSizeDimension(); j++)
      {
        voxelIndex[j] = continuousVoxelIndex[j];
      }

      SizeType neighbourhoodSize;
      neighbourhoodSize.Fill(2);
      neighbourhoodSize[axis] = 1;

      IndexType neighbourhoodIndex = voxelIndex;
      RegionType neighbourhoodRegion;
      neighbourhoodRegion.SetSize(neighbourhoodSize);
      neighbourhoodRegion.SetIndex(neighbourhoodIndex);

      bool isNearRegion = false;
      itk::ImageRegionIterator<BinaryImageType> iteratorGreen(bluePipeline.m_RegionGrowingFilter->GetOutput(), neighbourhoodRegion);
      for (iteratorGreen.GoToBegin(); !iteratorGreen.IsAtEnd(); ++iteratorGreen)
      {
        if (iteratorGreen.Get() > 0)
        {
          isNearRegion = true;
        }
      }

      itk::ImageRegionIterator<BinaryImageType> iteratorBlue(greenPipeline.m_RegionGrowingFilter->GetOutput(), neighbourhoodRegion);
      for (iteratorBlue.GoToBegin(); !iteratorBlue.IsAtEnd(); ++iteratorBlue)
      {
        if (iteratorBlue.Get() > 0)
        {
          isNearRegion = true;
        }
      }

      if (isNearRegion)
      {
        outputContour->AddVertex(pointInContour);
      }
      else if (!isNearRegion && outputContour->GetNumberOfPoints() > 0)
      {
        outputContours.AddContour(contourNumber, outputContour);
        outputContour = mitk::Contour::New();
        mitk::MIDASContourTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
        contourNumber++;
      }
    }
    if (outputContour->GetNumberOfPoints() > 0)
    {
      outputContours.AddContour(contourNumber, outputContour);
      outputContour = mitk::Contour::New();
      mitk::MIDASContourTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
      contourNumber++;
    }
    contourIt++;
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
bool MIDASGeneralSegmentorView
::ITKImageHasNonZeroEdgePixels(
    itk::Image<TPixel, VImageDimension> *itkImage
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::SizeType SizeType;

  RegionType region = itkImage->GetLargestPossibleRegion();
  SizeType regionSize = region.GetSize();
  IndexType voxelIndex;

  for (unsigned int i = 0; i < IndexType::GetIndexDimension(); i++)
  {
    regionSize[i] -= 1;
  }

  itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(itkImage, region);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    voxelIndex = iterator.GetIndex();
    bool isEdge(false);
    for (unsigned int i = 0; i < IndexType::GetIndexDimension(); i++)
    {
      if ((int)voxelIndex[i] == 0 || (int)voxelIndex[i] == (int)regionSize[i])
      {
        isEdge = true;
      }
    }
    if (isEdge && itkImage->GetPixel(voxelIndex) > 0)
    {
      return true;
    }
  }
  return false;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void MIDASGeneralSegmentorView
::ITKPropagateSeedsToNewSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* currentSeeds,
    mitk::PointSet* newSeeds,
    int axis,
    int oldSliceNumber,
    int newSliceNumber
    )
{

  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  PointType voxelIndexInMillimetres;
  IndexType voxelIndex;
  IndexType newVoxelIndex;

  bool newSliceHasSeeds = this->ITKSliceDoesHaveSeeds(itkImage, currentSeeds, axis, newSliceNumber);

  newSeeds->Clear();

  unsigned int pointCounter = 0;
  for (int i = 0; i < currentSeeds->GetSize(); i++)
  {
    newSeeds->InsertPoint(pointCounter++, currentSeeds->GetPoint(i));

    // Don't overwrite any existing seeds on new slice.
    if (!newSliceHasSeeds)
    {
      voxelIndexInMillimetres = currentSeeds->GetPoint(i);
      itkImage->TransformPhysicalPointToIndex(voxelIndexInMillimetres, voxelIndex);

      if (voxelIndex[axis] == oldSliceNumber)
      {
        newVoxelIndex = voxelIndex;
        newVoxelIndex[axis] = newSliceNumber;
        itkImage->TransformIndexToPhysicalPoint(newVoxelIndex, voxelIndexInMillimetres);

        newSeeds->InsertPoint(pointCounter++, voxelIndexInMillimetres);
      }
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKDestroyPipeline(itk::Image<TPixel, VImageDimension>* itkImage)
{

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  std::map<std::string, GeneralSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  GeneralSegmentorPipeline<TPixel, VImageDimension> *pipeline = dynamic_cast<GeneralSegmentorPipeline<TPixel, VImageDimension>*>(iter->second);
  if (pipeline != NULL)
  {
    delete pipeline;
  }
  else
  {
    MITK_ERROR << "MIDASGeneralSegmentorView::DestroyITKPipeline(..), failed to delete pipeline, as it was already NULL????" << std::endl;
  }
  m_TypeToPipelineMap.clear();
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ITKInitialiseSeedsForVolume(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet& seeds,
    int axis
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType region = itkImage->GetLargestPossibleRegion();

  this->ITKAddNewSeedsToPointSet(
      itkImage,
      region,
      0,
      axis,
      seeds
      );
}

/**************************************************************
 * End of ITK stuff.
 *************************************************************/
