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

#include "mitkExtractImageFilter.h"
#include "mitkDataNodeObject.h"
#include "mitkNodePredicateDataType.h"
#include "mitkNodePredicateProperty.h"
#include "mitkNodePredicateAnd.h"
#include "mitkNodePredicateNot.h"
#include "mitkProperties.h"
#include "mitkRenderingManager.h"
#include "mitkSegTool2D.h"
#include "mitkVtkResliceInterpolationProperty.h"
#include "mitkPointSet.h"
#include "mitkGlobalInteraction.h"
#include "mitkTool.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPosnTool.h"
#include "mitkNodePredicateDataType.h"
#include "mitkPointSet.h"
#include "mitkImageAccessByItk.h"
#include "mitkSlicedGeometry3D.h"
#include "mitkITKImageImport.h"
#include "mitkGeometry2D.h"
#include "mitkOperationEvent.h"
#include "mitkUndoController.h"
#include "mitkDataStorageUtils.h"
#include "QmitkStdMultiWidget.h"
#include "QmitkRenderWindow.h"

#include "vtkImageData.h"

#include "MIDASGeneralSegmentorCommands.h"
#include "itkMIDASHelper.h"
#include "itkMIDASPropagateUpProcessor.h"
#include "itkMIDASPropagateDownProcessor.h"
#include "itkMIDASPropagateProcessor.h"

const std::string MIDASGeneralSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor";
const mitk::OperationType MIDASGeneralSegmentorView::OP_THRESHOLD_APPLY = 9320411;
const mitk::OperationType MIDASGeneralSegmentorView::OP_PROPAGATE_UP = 9320412;
const mitk::OperationType MIDASGeneralSegmentorView::OP_PROPAGATE_DOWN = 9320413;
const mitk::OperationType MIDASGeneralSegmentorView::OP_WIPE_SLICE = 9320414;
const mitk::OperationType MIDASGeneralSegmentorView::OP_WIPE_PLUS = 9320415;
const mitk::OperationType MIDASGeneralSegmentorView::OP_WIPE_MINUS = 9320416;
const mitk::OperationType MIDASGeneralSegmentorView::OP_RETAIN_MARKS = 9320417;

MIDASGeneralSegmentorView::MIDASGeneralSegmentorView()
: QmitkMIDASBaseSegmentationFunctionality()
, m_GeneralControls(NULL)
, m_Layout(NULL)
, m_ContainerForSelectorWidget(NULL)
, m_ContainerForControlsWidget(NULL)
, m_OrientationButtons(NULL)
{
  m_Interface = new MIDASGeneralSegmentorViewEventInterface();
  m_Interface->SetMIDASGeneralSegmentorView(this);
}

MIDASGeneralSegmentorView::MIDASGeneralSegmentorView(
    const MIDASGeneralSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

MIDASGeneralSegmentorView::~MIDASGeneralSegmentorView()
{
  if (m_GeneralControls != NULL)
  {
    delete m_GeneralControls;
  }

  if (m_OrientationButtons != NULL)
  {
    delete m_OrientationButtons;
  }
}

std::string MIDASGeneralSegmentorView::GetViewID() const
{
  return VIEW_ID;
}

void MIDASGeneralSegmentorView::CreateQtPartControl(QWidget *parent)
{
  m_Parent = parent;

  if (!m_GeneralControls)
  {

    m_Layout = new QGridLayout(parent);

    m_ContainerForSelectorWidget = new QWidget(parent);
    m_ContainerForControlsWidget = new QWidget(parent);

    m_GeneralControls = new MIDASGeneralSegmentorViewControlsWidget();
    m_GeneralControls->setupUi(m_ContainerForControlsWidget);

    QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(m_ContainerForSelectorWidget);

    m_Layout->addWidget(m_ContainerForSelectorWidget, 0, 0);
    m_Layout->addWidget(m_ContainerForControlsWidget, 1, 0);

    m_OrientationButtons = new QButtonGroup();
    m_OrientationButtons->setExclusive(true);
    m_OrientationButtons->addButton(m_GeneralControls->m_AxialRadioButton);
    m_OrientationButtons->addButton(m_GeneralControls->m_SagittalRadioButton);
    m_OrientationButtons->addButton(m_GeneralControls->m_CoronalRadioButton);

    m_GeneralControls->m_ManualToolSelectionBox->SetGenerateAccelerators(true);
    m_GeneralControls->m_ManualToolSelectionBox->SetLayoutColumns(3);
    m_GeneralControls->m_ManualToolSelectionBox->SetToolGUIArea( m_GeneralControls->m_ManualToolGUIContainer );
    m_GeneralControls->m_ManualToolSelectionBox->SetDisplayedToolGroups("Seed Draw Poly");
    m_GeneralControls->m_ManualToolSelectionBox->SetEnabledMode( QmitkToolSelectionBox::EnabledWithReferenceAndWorkingData );
    m_GeneralControls->SetEnableThresholdingWidgets(false);
    m_GeneralControls->SetEnableThresholdingCheckbox(false);
  }

  this->CreateConnections();
}

void MIDASGeneralSegmentorView::CreateConnections()
{
  if ( m_GeneralControls )
  {
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
    connect(m_GeneralControls->m_RetainMarksCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnRetainMarksCheckBoxToggled(bool)));
    connect(m_GeneralControls->m_ThresholdCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnThresholdCheckBoxToggled(bool)));
    connect(m_GeneralControls->m_SeePriorCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnSeePriorCheckBoxToggled(bool)));
    connect(m_GeneralControls->m_SeeNextCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnSeeNextCheckBoxToggled(bool)));
    connect(m_GeneralControls->m_ManualToolSelectionBox, SIGNAL(ToolSelected(int)), this, SLOT(OnManualToolSelected(int)) );
    connect(m_GeneralControls->m_ThresholdLowerSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(OnLowerThresholdValueChanged(double)));
    connect(m_GeneralControls->m_ThresholdUpperSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(OnUpperThresholdValueChanged(double)));
    connect(m_GeneralControls->m_AxialRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnOrientationAxialToggled(bool)));
    connect(m_GeneralControls->m_CoronalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnOrientationCoronalToggled(bool)));
    connect(m_GeneralControls->m_SagittalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnOrientationSagittalToggled(bool)));
    connect(m_GeneralControls->m_SliceSelectionWidget, SIGNAL(SliceNumberChanged(int, int)), this, SLOT(OnSliceNumberChanged(int, int)));
    connect(m_ImageAndSegmentationSelector->m_NewSegmentationButton, SIGNAL(clicked()), this, SLOT(OnCreateNewSegmentationButtonPressed()) );
  }
}

void MIDASGeneralSegmentorView::Activated()
{
  QmitkMIDASBaseSegmentationFunctionality::Activated();
}

void MIDASGeneralSegmentorView::Deactivated()
{
  QmitkMIDASBaseSegmentationFunctionality::Deactivated();
}

void MIDASGeneralSegmentorView::OnManualToolSelected(int id)
{
  // disable crosshair movement when a manual drawing tool is active (otherwise too much visual noise)
  if (m_MultiWidget)
  {
    /* If we do want a Posn button, put this code back in, and insert Posn into list of tools.
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::Tool* tool = toolManager->GetToolById(id);
    if (tool != NULL && strcmp(tool->GetName(),"Posn") != 0)
    {
      m_MultiWidget->DisableNavigationControllerEventListening();
      m_MultiWidget->SetWidgetPlaneMode(0);
    }
    else
    {
      m_MultiWidget->EnableNavigationControllerEventListening();
    }
    */

    // here, if we selected a tool, we stop navigation controls.
    if (id != -1)
    {
      m_MultiWidget->DisableNavigationControllerEventListening();
      m_MultiWidget->SetWidgetPlaneMode(0);

      // ToDo: Create a user preference
      if (!this->m_GeneralControls->m_AxialRadioButton->isChecked()
          && !this->m_GeneralControls->m_SagittalRadioButton->isChecked()
          && !this->m_GeneralControls->m_CoronalRadioButton->isChecked())
      {
        this->m_GeneralControls->m_CoronalRadioButton->setChecked(true);
        this->OnOrientationCoronalToggled(true);
      }
    }
    else
    {
      m_MultiWidget->EnableNavigationControllerEventListening();
    }
  }
  this->UpdatePriorAndNext();
  this->UpdateRegionGrowing();
}

bool MIDASGeneralSegmentorView::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  bool canRestart = false;

  if (node.IsNotNull()
      && mitk::IsNodeABinaryImage(node)
      )
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDefaultDataStorage(), node, false);
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

bool MIDASGeneralSegmentorView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  assert(node);
  bool result = false;

  if (IsNodeABinaryImage(node))
  {

    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDefaultDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      mitk::DataNode::Pointer regionGrowingNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seePriorNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seeNextNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_NEXT_IMAGE_NAME.c_str(), node, true);

      if (regionGrowingNode.IsNotNull() && seePriorNode.IsNotNull() && seeNextNode.IsNotNull())
      {
        result = true;
      }
    }
  }
  return result;
}

void MIDASGeneralSegmentorView::EnableSegmentationWidgets(bool b)
{
  bool thresholdingIsOn = this->m_GeneralControls->m_ThresholdCheckBox->isChecked();
  if(b)
  {
    if (thresholdingIsOn)
    {
      this->m_GeneralControls->SetEnableAllWidgets(false);
    }
    else
    {
      this->m_GeneralControls->SetEnableAllWidgets(true);
      this->m_GeneralControls->SetEnableThresholdingWidgets(false);
    }
  }
  else
  {
    this->m_GeneralControls->SetEnableAllWidgets(false);
  }
}

mitk::ToolManager* MIDASGeneralSegmentorView::GetToolManager()
{
  return m_GeneralControls->m_ManualToolSelectionBox->GetToolManager();
}

mitk::DataNode::Pointer MIDASGeneralSegmentorView::CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name)
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
  helperImageNode->SetBoolProperty("helper object", false);

  this->ApplyDisplayOptions(helperImageNode);
  this->GetDefaultDataStorage()->Add(helperImageNode, segmentationNode);

  return helperImageNode;
}

template<typename TPixel, unsigned int VImageDimension>
void MIDASGeneralSegmentorView::CopyImage(
    itk::Image<TPixel, VImageDimension>* input,
    itk::Image<TPixel, VImageDimension>* output
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  itk::ImageRegionIterator<ImageType> inputIterator(input, input->GetLargestPossibleRegion());
  itk::ImageRegionIterator<ImageType> outputIterator(output, output->GetLargestPossibleRegion());

  for (inputIterator.GoToBegin(), outputIterator.GoToBegin();
      !inputIterator.IsAtEnd();
      ++inputIterator, ++outputIterator
      )
  {
    outputIterator.Set(inputIterator.Get());
  }
}

mitk::DataNode* MIDASGeneralSegmentorView::OnCreateNewSegmentationButtonPressed()
{
  // This creates the "final output image"... i.e. the segmentation result.
  mitk::DataNode::Pointer emptySegmentation = QmitkMIDASBaseSegmentationFunctionality::OnCreateNewSegmentationButtonPressed();
  assert(emptySegmentation);

  // Set some initial properties, to make sure they are initialised properly.
  emptySegmentation->SetBoolProperty(mitk::MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), false);

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image* image = this->GetReferenceImageFromToolManager();
  if (image != NULL)
  {
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);

    // Make sure these are up to date, even though we don't use them right now.
    image->GetScalarValueMin();
    image->GetScalarValueMax();

    // This creates the point set for the seeds.
    mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
    mitk::DataNode::Pointer pointSetNode = mitk::DataNode::New();
    pointSetNode->SetData( pointSet );
    pointSetNode->SetProperty( "name", mitk::StringProperty::New( mitk::MIDASTool::SEED_POINT_SET_NAME ) );
    pointSetNode->SetProperty( "opacity", mitk::FloatProperty::New( 1 ) );
    pointSetNode->SetProperty( "point line width", mitk::IntProperty::New( 1 ) );
    pointSetNode->SetProperty( "point 2D size", mitk::IntProperty::New( 3 ) );
    pointSetNode->SetBoolProperty("helper object", true);
    pointSetNode->SetColor( 1.0, 0.75, 0.8 );
    this->GetDefaultDataStorage()->Add(pointSetNode, emptySegmentation);

    // In addition, (and this may need re-visiting), we create 3 volumes for working data
    // even though, in all 3 cases, we only ever display 1 slice at once. It will enable
    // us, if necessary to do the whole thing in 3D, or to at least view the 2D segmentation
    // easily in 3D, without worrying too much about geometry and memory management.
    //
    // 1. For holding the current region growing output
    // 2. For holding a copy of the prior slice (for "see prior").
    // 3. For holding a copy of the next slice (for "see next")
    mitk::DataNode::Pointer regionGrowingOutputNode = this->CreateHelperImage(image, emptySegmentation, 0,0,1, mitk::MIDASTool::REGION_GROWING_IMAGE_NAME);
    mitk::DataNode::Pointer seePriorOutputNode = this->CreateHelperImage(image, emptySegmentation, 1,0,1, mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME);
    mitk::DataNode::Pointer seeNextOutputNode = this->CreateHelperImage(image, emptySegmentation, 0,1,1, mitk::MIDASTool::SEE_NEXT_IMAGE_NAME);

    // Set working data. Compare with MIDASMorphologicalSegmentorView.
    // In practice, this editor, doesn't actually use any tools to write to the data.
    // The output is always produced by the result of region growing.
    mitk::ToolManager::DataVectorType workingData;
    workingData.push_back(emptySegmentation);
    toolManager->SetWorkingData(workingData);

    mitk::Image::Pointer emptySegmentationImage = dynamic_cast<mitk::Image*>(emptySegmentation->GetData());

    // If we are restarting a segmentation, we need to copy the segmentation from m_SelectedNode.
    if (m_SelectedNode.IsNotNull()
        && m_SelectedImage.IsNotNull()
        && emptySegmentationImage.IsNotNull()
        && mitk::IsNodeABinaryImage(m_SelectedNode)
        && m_SelectedNode != emptySegmentation
        && CanStartSegmentationForBinaryNode(m_SelectedNode)
        )
    {
      try
      {
        typedef itk::Image<unsigned char, 3> SegmentationImageType;
        typedef mitk::ImageToItk< SegmentationImageType > SegmentationImageToItkType;

        SegmentationImageToItkType::Pointer previousSegmentationToItk = SegmentationImageToItkType::New();
        previousSegmentationToItk->SetInput(m_SelectedImage);
        previousSegmentationToItk->Update();


        SegmentationImageToItkType::Pointer newSegmentationToItk = SegmentationImageToItkType::New();
        newSegmentationToItk->SetInput(emptySegmentationImage);
        newSegmentationToItk->Update();

        this->CopyImage(previousSegmentationToItk->GetOutput(), newSegmentationToItk->GetOutput());
      }
      catch(const itk::ExceptionObject &err)
      {
        MITK_ERROR << "Caught exception, so abandoning see prior/next update" << err.what();
      }
    }

    // Setup widgets.
    this->m_GeneralControls->SetEnableAllWidgets(true);
    this->m_GeneralControls->SetEnableThresholdingWidgets(false);
    this->m_GeneralControls->SetEnableThresholdingCheckbox(true);
    this->m_GeneralControls->m_ThresholdCheckBox->setChecked(false);
    this->m_GeneralControls->SetEnableManualToolSelectionBox(true);
    this->SelectNode(emptySegmentation);
  }
  return emptySegmentation.GetPointer();
}

int MIDASGeneralSegmentorView::GetSliceNumber()
{
  return this->m_GeneralControls->m_SliceSelectionWidget->GetValue();
}

int MIDASGeneralSegmentorView::GetAxis()
{
  int axisNumber = -1;

  if (this->m_GeneralControls->m_AxialRadioButton->isChecked())
  {
    axisNumber = QmitkMIDASBaseSegmentationFunctionality::GetAxis(QmitkMIDASBaseSegmentationFunctionality::AXIAL);
  }
  else if (this->m_GeneralControls->m_SagittalRadioButton->isChecked())
  {
    axisNumber = QmitkMIDASBaseSegmentationFunctionality::GetAxis(QmitkMIDASBaseSegmentationFunctionality::SAGITTAL);
  }
  else if (this->m_GeneralControls->m_CoronalRadioButton->isChecked())
  {
    axisNumber = QmitkMIDASBaseSegmentationFunctionality::GetAxis(QmitkMIDASBaseSegmentationFunctionality::CORONAL);
  }

  return axisNumber;
}

itk::ORIENTATION_ENUM MIDASGeneralSegmentorView::GetOrientationAsEnum()
{
  itk::ORIENTATION_ENUM orientation = itk::ORIENTATION_UNKNOWN;

  if (this->m_GeneralControls->m_AxialRadioButton->isChecked())
  {
    orientation = itk::ORIENTATION_AXIAL;
  }
  else if (this->m_GeneralControls->m_SagittalRadioButton->isChecked())
  {
    orientation = itk::ORIENTATION_SAGITTAL;
  }
  else if (this->m_GeneralControls->m_CoronalRadioButton->isChecked())
  {
    orientation = itk::ORIENTATION_CORONAL;
  }

  return orientation;
}

void MIDASGeneralSegmentorView::OnOrientationSelected(itk::ORIENTATION_ENUM orientation)
{
  mitk::Image* image = this->GetReferenceImageFromToolManager();
  if (image != NULL && m_MultiWidget != NULL)
  {

    // If we switch orientation, we need to recalculate the number of slices in that direction,
    // so we can setup the slice select widget, and we should switch to the correct slice
    // as determined by the intersection point of the ortho viewer.

    int axis = this->GetAxis();
    vtkImageData* vtkImage = image->GetVtkImageData(0);
    int *extents = vtkImage->GetWholeExtent();
    int size = extents[2*axis + 1] + 1;
    mitk::Index3D crossPositionInVoxels;
    mitk::Point3D crossPositionInMillimetres = m_MultiWidget->GetCrossPosition();
    image->GetGeometry()->WorldToIndex(crossPositionInMillimetres, crossPositionInVoxels);
    int sliceNumber = -1;
    int numberFromSlider = -1;

    if (this->m_GeneralControls->m_AxialRadioButton->isChecked())
    {
      numberFromSlider = m_MultiWidget->mitkWidget1->GetSliceNavigationController()->GetSlice()->GetPos();
    }
    else if (this->m_GeneralControls->m_SagittalRadioButton->isChecked())
    {
      numberFromSlider = m_MultiWidget->mitkWidget2->GetSliceNavigationController()->GetSlice()->GetPos();
    }
    else if (this->m_GeneralControls->m_CoronalRadioButton->isChecked())
    {
      numberFromSlider = m_MultiWidget->mitkWidget3->GetSliceNavigationController()->GetSlice()->GetPos();
    }

    // Hack:
    // It appears that sometimes the slider in the SliceNavigationController returns the value at the opposite end of the range.
    // Additionally, there is rounding going on, so converting mm to vox can differ by 1 slice from the SliceNavigationController.
    if (abs(numberFromSlider - crossPositionInVoxels[axis]) > 1)
    {
      sliceNumber = crossPositionInVoxels[axis];
    }
    else
    {
      sliceNumber = numberFromSlider;
    }
    this->m_GeneralControls->m_SliceSelectionWidget->SetMinimum(0);
    this->m_GeneralControls->m_SliceSelectionWidget->SetMaximum(size - 1);
    this->m_GeneralControls->m_SliceSelectionWidget->SetValue(sliceNumber);
    this->OnSliceNumberChanged(sliceNumber, sliceNumber);
    this->UpdatePriorAndNext();
    this->UpdateRegionGrowing();
  }
}

void MIDASGeneralSegmentorView::OnOrientationAxialToggled(bool b)
{
  if (this->m_MultiWidget != NULL)
  {
    this->m_MultiWidget->changeLayoutToWidget1();
  }
  this->OnOrientationSelected(itk::ORIENTATION_AXIAL);
}

void MIDASGeneralSegmentorView::OnOrientationSagittalToggled(bool b)
{
  if (this->m_MultiWidget != NULL)
  {
    this->m_MultiWidget->changeLayoutToWidget2();
  }
  this->OnOrientationSelected(itk::ORIENTATION_SAGITTAL);
}

void MIDASGeneralSegmentorView::OnOrientationCoronalToggled(bool b)
{
  if (this->m_MultiWidget != NULL)
  {
    this->m_MultiWidget->changeLayoutToWidget3();
  }
  this->OnOrientationSelected(itk::ORIENTATION_CORONAL);
}

void MIDASGeneralSegmentorView::RecalculateMinAndMaxOfImage()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    double min = referenceImage->GetScalarValueMinNoRecompute();
    double max = referenceImage->GetScalarValueMaxNoRecompute();
    this->m_GeneralControls->SetLowerAndUpperIntensityRanges(min, max);
  }
}

mitk::PointSet* MIDASGeneralSegmentorView::GetSeeds()
{
  mitk::PointSet* result = NULL;
  mitk::Image *workingImage = this->GetWorkingImageFromToolManager(0);

  if (workingImage != NULL)
  {
    mitk::TNodePredicateDataType<mitk::PointSet>::Pointer isPointSet = mitk::TNodePredicateDataType<mitk::PointSet>::New();
    mitk::DataStorage::SetOfObjects::ConstPointer allPointSets = this->GetDefaultDataStorage()->GetSubset( isPointSet );
    for ( mitk::DataStorage::SetOfObjects::const_iterator iter = allPointSets->begin();
          iter != allPointSets->end();
          ++iter)
    {
      mitk::DataNode* node = *iter;
      if (node != NULL && node->GetName() == mitk::MIDASTool::SEED_POINT_SET_NAME)
      {
        result = static_cast<mitk::PointSet*>((*iter)->GetData());
      }
    }
  }
  return result;
}

void MIDASGeneralSegmentorView::RecalculateMinAndMaxOfSeedValues()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  mitk::PointSet::Pointer seeds = this->GetSeeds();

  if (referenceImage.IsNotNull() && seeds.IsNotNull())
  {
    double min = -1;
    double max = -1;

    try
    {
      AccessFixedDimensionByItk_n(referenceImage, RecalculateMinAndMaxOfSeedValuesUsingITK, 3, (seeds.GetPointer(), min, max));
      this->m_GeneralControls->SetSeedMinAndMaxValues(min, max);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning recalculating min and max of seeds values:" << e.what();
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::RecalculateMinAndMaxOfSeedValuesUsingITK(
    itk::Image<TPixel, VImageDimension>* itkImage,
    mitk::PointSet* points,
    double &min,
    double &max
    )
{
  if (points->GetSize() == 0)
  {
    min = 0;
    max = 0;
  }
  else
  {
    min = std::numeric_limits<double>::max();
    max = std::numeric_limits<double>::min();

    typedef itk::Image<TPixel, VImageDimension> ImageType;
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::IndexType IndexType;

    PointType millimetreCoordinate;
    IndexType voxelCoordinate;

    // Iterate through each point, get voxel value, keep running total of min/max.
    for (int i = 0; i < points->GetSize(); i++)
    {
      mitk::PointSet::PointType point = points->GetPoint(i);

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

void MIDASGeneralSegmentorView::SetVisiblityOnDerivedImage(std::string name, bool visibility)
{
  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer workingData = toolManager->GetWorkingData(0);
  if (workingData.IsNotNull())
  {
    mitk::DataNode::Pointer node = this->GetDefaultDataStorage()->GetNamedDerivedNode(name.c_str(), workingData, true);
    if (node.IsNotNull())
    {
      node->SetVisibility(visibility);
      this->UpdatePriorAndNext();
    }
  }

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASGeneralSegmentorView::OnSeePriorCheckBoxToggled(bool b)
{
  this->SetVisiblityOnDerivedImage(mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME, b);
}

void MIDASGeneralSegmentorView::OnSeeNextCheckBoxToggled(bool b)
{
  this->SetVisiblityOnDerivedImage(mitk::MIDASTool::SEE_NEXT_IMAGE_NAME, b);
}

void MIDASGeneralSegmentorView::UpdatePriorAndNext()
{
  if (!this->m_GeneralControls->m_SeePriorCheckBox->isChecked() && !this->m_GeneralControls->m_SeeNextCheckBox->isChecked())
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();

  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

    if (workingImage.IsNotNull())
    {
      mitk::DataNode::Pointer priorNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME.c_str(), workingNode, true);
      assert(priorNode);

      mitk::Image::Pointer priorImage = dynamic_cast<mitk::Image*>(priorNode->GetData());
      assert(priorImage);

      mitk::DataNode::Pointer nextNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_NEXT_IMAGE_NAME.c_str(), workingNode, true);
      assert(nextNode);

      mitk::Image::Pointer nextImage = dynamic_cast<mitk::Image*>(nextNode->GetData());
      assert(nextImage);

      int sliceNumber = this->GetSliceNumber();
      int axisNumber = this->GetAxis();
      itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();

      if (axisNumber != -1 && sliceNumber != -1 && orientation != itk::ORIENTATION_UNKNOWN)
      {
        try
        {
          typedef itk::Image<unsigned char, 3> SegmentationImageType;
          typedef mitk::ImageToItk< SegmentationImageType > SegmentationImageToItkType;

          SegmentationImageToItkType::Pointer priorImageToItk = SegmentationImageToItkType::New();
          priorImageToItk->SetInput(priorImage);
          priorImageToItk->Update();

          SegmentationImageToItkType::Pointer nextImageToItk = SegmentationImageToItkType::New();
          nextImageToItk->SetInput(nextImage);
          nextImageToItk->Update();

          SegmentationImageToItkType::Pointer workingImageToItk = SegmentationImageToItkType::New();
          workingImageToItk->SetInput(workingImage);
          workingImageToItk->Update();

          itk::MIDASRegionOfInterestCalculator<unsigned char, 3>::Pointer calculator = itk::MIDASRegionOfInterestCalculator<unsigned char, 3>::New();
          int direction = calculator->GetPlusOrUpDirection(workingImageToItk->GetOutput(), orientation);

          if (this->m_GeneralControls->m_SeePriorCheckBox->isChecked())
          {
            this->CopySlice(
                workingImageToItk->GetOutput(),
                priorImageToItk->GetOutput(),
                orientation,
                sliceNumber,
                axisNumber,
                direction
                );
          }

          if (this->m_GeneralControls->m_SeeNextCheckBox)
          {
            this->CopySlice(
                workingImageToItk->GetOutput(),
                nextImageToItk->GetOutput(),
                orientation,
                sliceNumber,
                axisNumber,
                -1*direction
                );
          }

        }
        catch(const itk::ExceptionObject &err)
        {
          MITK_ERROR << "Caught exception, so abandoning see prior/next update" << err.what();
        }
      } // end if check axis etc.
    } // end if reference image
  } // end if working image
} // end function

template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::CopySlice(
      itk::Image<TPixel, VImageDimension>* sourceImage,
      itk::Image<TPixel, VImageDimension>* destinationImage,
      itk::ORIENTATION_ENUM orientation,
      int sliceNumber,
      int axis,
      int directionOffset
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexType IndexType;

  const RegionType originalSourceRegion = sourceImage->GetLargestPossibleRegion();
  const SizeType originalSourceSize = originalSourceRegion.GetSize();
  const IndexType originalSourceIndex = originalSourceRegion.GetIndex();

  RegionType sourceRegion = originalSourceRegion;
  SizeType sourceSize = originalSourceSize;
  IndexType sourceIndex = originalSourceIndex;

  RegionType destinationRegion = originalSourceRegion;
  SizeType destinationSize = originalSourceSize;
  IndexType destinationIndex = originalSourceIndex;

  if (sliceNumber+directionOffset < 0 || sliceNumber+directionOffset >= (int)originalSourceSize[axis])
  {
    directionOffset = 0;
  }

  // The destination is the current slice.
  destinationSize[axis] = 1;
  destinationIndex[axis] = sliceNumber;
  destinationRegion.SetSize(destinationSize);
  destinationRegion.SetIndex(destinationIndex);

  // The source data, comes from either the slice above or below.
  sourceSize[axis] = 1;
  sourceIndex[axis] = sliceNumber + directionOffset;
  sourceRegion.SetSize(sourceSize);
  sourceRegion.SetIndex(sourceIndex);

  // Now copy data
  itk::ImageRegionConstIterator<ImageType> sourceIterator(sourceImage, sourceRegion);
  itk::ImageRegionIterator<ImageType> destinationIterator(destinationImage, destinationRegion);

  for (sourceIterator.GoToBegin(), destinationIterator.GoToBegin();
      !sourceIterator.IsAtEnd();
      ++sourceIterator, ++destinationIterator)
  {
    TPixel pixel = sourceIterator.Get();
    destinationIterator.Set(pixel);
  }
}

void MIDASGeneralSegmentorView::OnThresholdCheckBoxToggled(bool b)
{
  this->m_GeneralControls->SetEnableThresholdingWidgets(b);
  if (b)
  {
    this->RecalculateMinAndMaxOfImage();
    this->RecalculateMinAndMaxOfSeedValues();
    this->UpdateRegionGrowing();
  }
}

void MIDASGeneralSegmentorView::OnLowerThresholdValueChanged(double d)
{
  this->UpdateRegionGrowing();
}

void MIDASGeneralSegmentorView::OnUpperThresholdValueChanged(double d)
{
  this->UpdateRegionGrowing();
}

void MIDASGeneralSegmentorView::UpdateRegionGrowing()
{
  if (!this->m_GeneralControls->m_ThresholdCheckBox->isChecked())
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();

  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer workingNode = this->GetWorkingNodesFromToolManager()[0];
    mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);

    if (workingImage.IsNotNull())
    {
      mitk::DataNode::Pointer regionGrowingNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      mitk::MIDASDrawTool *drawTool = static_cast<mitk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASDrawTool>()));
      assert(drawTool);

      mitk::MIDASPolyTool *polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
      assert(polyTool);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      double lowerThreshold = this->m_GeneralControls->m_ThresholdLowerSliderWidget->value();
      double upperThreshold = this->m_GeneralControls->m_ThresholdUpperSliderWidget->value();
      bool skipUpdate = !(this->m_GeneralControls->m_ThresholdCheckBox->isChecked());
      int sliceNumber = this->GetSliceNumber();
      int axisNumber = this->GetAxis();
      itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();

      if (axisNumber != -1 && sliceNumber != -1 && orientation != itk::ORIENTATION_UNKNOWN)
      {
        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              InvokeITKRegionGrowingPipeline, 3,
              (skipUpdate,
               *seeds,
               *drawTool,
               *polyTool,
               orientation,
               sliceNumber,
               axisNumber,
               lowerThreshold,
               upperThreshold,
               regionGrowingNode,  // This is the node for the image we are writing to.
               regionGrowingImage  // This is the image we are writing to.
              )
            );
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Caught exception, so abandoning pipeline update:" << e.what();
        }
      } // end if check axis etc.
    } // end if reference image
  } // end if working image
} // end function

template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::InvokeITKRegionGrowingPipeline(
    itk::Image<TPixel, VImageDimension>* itkImage,  // Grey scale image (read only).
    bool skipUpdate,
    mitk::PointSet& seeds,
    mitk::MIDASDrawTool& drawTool,
    mitk::MIDASPolyTool& polyTool,
    itk::ORIENTATION_ENUM orientation,
    int sliceNumber,
    int axisNumber,
    double lowerThreshold,
    double upperThreshold,
    mitk::DataNode::Pointer &outputRegionGrowingNode,
    mitk::Image::Pointer& outputRegionGrowingImage
    )
{

  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer regionGrowingToItk = ImageToItkType::New();
  regionGrowingToItk->SetInput(outputRegionGrowingImage);
  regionGrowingToItk->Update();

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
    pipeline->m_RegionGrowingProcessor->SetGreyScaleImage(itkImage);
    pipeline->m_RegionGrowingProcessor->SetDestinationImage(regionGrowingToItk->GetOutput());
  }
  else
  {
    myPipeline = iter->second;
    pipeline = static_cast<GeneralSegmentorPipeline<TPixel, VImageDimension>*>(myPipeline);
  }

  GeneralSegmentorPipelineParams params;
  params.m_SliceNumber = sliceNumber;
  params.m_AxisNumber = axisNumber;
  params.m_Orientation = orientation;
  params.m_LowerThreshold = lowerThreshold;
  params.m_UpperThreshold = upperThreshold;
  params.m_DrawTool = &drawTool;
  params.m_PolyTool = &polyTool;
  params.m_Seeds = &seeds;

  // Update pipeline.
  if (!skipUpdate)
  {
    pipeline->SetParam(params);
    pipeline->Update(params);

    // Get output into the output volume.
    //outputRegionGrowingImage->InitializeByItk< ImageType >(pipeline->m_PasteRegionFilter->GetOutput());
    //outputRegionGrowingImage->SetImportChannel(pipeline->m_PasteRegionFilter->GetOutput()->GetBufferPointer(), 0, mitk::Image::ReferenceMemory);
    outputRegionGrowingImage = mitk::ImportItkImage( pipeline->m_RegionGrowingProcessor->GetDestinationImage());
    outputRegionGrowingNode->SetData(outputRegionGrowingImage);
  }

  // Make sure all renderers update.
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASGeneralSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  mitk::ToolManager::DataVectorType workingNodes = this->GetWorkingNodesFromToolManager();
  if (workingNodes.size() > 0)
  {
    mitk::DataNode::Pointer workingDataNode = workingNodes[0];
    if (workingDataNode.IsNotNull())
    {
      if (workingDataNode.GetPointer() == node)
      {
        bool isContourBeingEdited(true);
        workingDataNode->GetBoolProperty(mitk::MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), isContourBeingEdited);
        if (!isContourBeingEdited)
        {
          this->RecalculateMinAndMaxOfSeedValues();
          this->RecalculateMinAndMaxOfImage();
          this->UpdateRegionGrowing();
        }
      }
    }
  }
}

void MIDASGeneralSegmentorView::DestroyPipeline()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk(referenceImage, DestroyITKPipeline, 3);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning clearing the segmentation image:" << e.what();
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::DestroyITKPipeline(itk::Image<TPixel, VImageDimension>* itkImage)
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
    MITK_ERROR << "MIDASGeneralSegmentorView::DestroyITKPipeline(..), failed to delete pipeline" << std::endl;
  }
  m_TypeToPipelineMap.clear();
}

void MIDASGeneralSegmentorView::ClearWorkingData()
{
  mitk::DataNode::Pointer workingData = this->GetToolManager()->GetWorkingData(0);
  if (workingData.IsNotNull())
  {
    mitk::DataNode::Pointer seePriorNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME.c_str(), workingData, true);
    mitk::DataNode::Pointer seeNextNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_NEXT_IMAGE_NAME.c_str(), workingData, true);
    mitk::DataNode::Pointer regionGrowingNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingData, true);

    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(workingData->GetData());
    mitk::Image::Pointer priorImage = dynamic_cast<mitk::Image*>(seePriorNode->GetData());
    mitk::Image::Pointer nextImage = dynamic_cast<mitk::Image*>(seeNextNode->GetData());
    mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());

    assert(seePriorNode);
    assert(seeNextNode);
    assert(regionGrowingNode);
    assert(segmentationImage);
    assert(priorImage);
    assert(nextImage);
    assert(regionGrowingImage);

    try
    {
      AccessFixedDimensionByItk(segmentationImage, ClearITKImage, 3);
      AccessFixedDimensionByItk(priorImage, ClearITKImage, 3);
      AccessFixedDimensionByItk(nextImage, ClearITKImage, 3);
      AccessFixedDimensionByItk(regionGrowingImage, ClearITKImage, 3);

      segmentationImage->Modified();
      priorImage->Modified();
      nextImage->Modified();
      regionGrowingImage->Modified();

      workingData->Modified();
      seePriorNode->Modified();
      seeNextNode->Modified();
      regionGrowingNode->Modified();
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning ClearWorkingData." << e.what();
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::ClearITKImage(itk::Image<TPixel, VImageDimension>* itkImage)
{
  itkImage->FillBuffer(0);
}

void MIDASGeneralSegmentorView::RemoveWorkingData()
{
  mitk::DataNode::Pointer workingData = this->GetToolManager()->GetWorkingData(0);
  if (workingData.IsNotNull())
  {
    mitk::DataNode::Pointer seePriorNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME.c_str(), workingData, true);
    assert(seePriorNode);

    mitk::DataNode::Pointer seeNextNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::SEE_NEXT_IMAGE_NAME.c_str(), workingData, true);
    assert(seeNextNode);

    mitk::DataNode::Pointer regionGrowingNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingData, true);
    assert(regionGrowingNode);

    this->GetDefaultDataStorage()->Remove(seePriorNode);
    this->GetDefaultDataStorage()->Remove(seeNextNode);
    this->GetDefaultDataStorage()->Remove(regionGrowingNode);

    mitk::ToolManager* toolManager = this->GetToolManager();
    mitk::ToolManager::DataVectorType emptyWorkingDataArray;
    toolManager->SetWorkingData(emptyWorkingDataArray);
    toolManager->ActivateTool(-1);
  }
}

void MIDASGeneralSegmentorView::OnResetButtonPressed()
{
  this->WipeTools();
  this->ClearWorkingData();
  this->UpdateRegionGrowing();
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASGeneralSegmentorView::OnOKButtonPressed()
{
  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(0);
  assert(segmentationNode);

  this->DestroyPipeline();
  this->RemoveWorkingData();
  this->UpdateVolumeProperty(segmentationNode);
  this->SetReferenceImageSelected();
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASGeneralSegmentorView::OnCancelButtonPressed()
{
  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(0);
  assert(segmentationNode);

  this->DestroyPipeline();
  this->RemoveWorkingData();
  this->GetDefaultDataStorage()->Remove(segmentationNode);
  this->SetReferenceImageSelected();
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASGeneralSegmentorView::ExecuteOperation(mitk::Operation* operation)
{
  if (!operation) return;

  typedef itk::Image<unsigned char, 3> SegmentationImageType;

  switch (operation->GetOperationType())
  {
  case OP_THRESHOLD_APPLY:
    {
      try
      {
        mitk::OpThresholdApply *op = static_cast<mitk::OpThresholdApply*>(operation);
        mitk::OpThresholdApply::ProcessorType::Pointer processor = op->GetProcessor();
        mitk::DataNode::Pointer targetNode = op->GetTargetNode();
        mitk::DataNode::Pointer sourceNode = op->GetSourceNode();
        bool redo = op->IsRedo();

        assert(targetNode);
        assert(sourceNode);

        mitk::Image::Pointer targetImage = dynamic_cast<mitk::Image*>(targetNode->GetData());
        mitk::Image::Pointer sourceImage = dynamic_cast<mitk::Image*>(sourceNode->GetData());

        assert(targetImage);
        assert(sourceImage);

        typedef mitk::ImageToItk< SegmentationImageType > SegmentationImageToItkType;
        SegmentationImageToItkType::Pointer targetImageToItk = SegmentationImageToItkType::New();
        targetImageToItk->SetInput(targetImage);
        targetImageToItk->Update();

        SegmentationImageToItkType::Pointer sourceImageToItk = SegmentationImageToItkType::New();
        sourceImageToItk->SetInput(sourceImage);
        sourceImageToItk->Update();

        processor->SetSourceImage(sourceImageToItk->GetOutput());
        processor->SetDestinationImage(targetImageToItk->GetOutput());
        processor->CalculateRegionOfInterest();

        if (redo)
        {
          processor->Redo();
        }
        else
        {
          processor->Undo();
        }

        targetImage = mitk::ImportItkImage( processor->GetDestinationImage());
        sourceImage = mitk::ImportItkImage( processor->GetSourceImage());

        targetNode->SetData(targetImage);
        targetNode->Modified();
        sourceNode->SetData(sourceImage);
        sourceNode->Modified();
      }
      catch( itk::ExceptionObject & err )
      {
        MITK_ERROR << "Failed to update due to: " << err << std::endl;
      }
      break;
    }
  case OP_PROPAGATE_UP:
  case OP_PROPAGATE_DOWN:
    {
      try
      {
        mitk::OpPropagate *op = static_cast<mitk::OpPropagate*>(operation);
        bool redo = op->IsRedo();
        int sliceNumber = op->GetSliceNumber();
        itk::ORIENTATION_ENUM orientation = op->GetOrientation();
        mitk::DataNode::Pointer regionGrowingNode = op->GetTargetNode();
        mitk::DataNode::Pointer referenceNode = op->GetSourceNode();

        assert(regionGrowingNode);
        assert(referenceNode);

        mitk::Image::Pointer referenceImage = dynamic_cast<mitk::Image*>(referenceNode->GetData());

        try
        {
          AccessFixedDimensionByItk_n(referenceImage, RunPropagateProcessor, 3,
                (
                  op,
                  regionGrowingNode,
                  redo,
                  sliceNumber,
                  orientation
                )
              );
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Caught exception, so abandoning RunPropagateProcessor." << e.what();
          return;
        }
      }
      catch( itk::ExceptionObject & err )
      {
        MITK_ERROR << "Failed to update due to: " << err << std::endl;
      }

      break;
    }
  case OP_WIPE_SLICE:
  case OP_WIPE_PLUS:
  case OP_WIPE_MINUS:
    {
      mitk::OpWipe *op = static_cast<mitk::OpWipe*>(operation);
      mitk::OpWipe::WipeProcessorType::Pointer processor = op->GetProcessor();

      bool redo = op->IsRedo();
      int sliceNumber = op->GetSliceNumber();
      itk::ORIENTATION_ENUM orientation = op->GetOrientation();

      mitk::DataNode::Pointer targetNode = op->GetTargetNode();
      assert(targetNode);

      mitk::Image::Pointer targetImage = dynamic_cast<mitk::Image*>(targetNode->GetData());
      assert(targetImage);

      typedef mitk::ImageToItk< SegmentationImageType > SegmentationImageToItkType;
      SegmentationImageToItkType::Pointer targetImageToItk = SegmentationImageToItkType::New();
      targetImageToItk->SetInput(targetImage);
      targetImageToItk->Update();

      processor->SetDestinationImage(targetImageToItk->GetOutput());
      processor->SetOrientationAndSlice(orientation, sliceNumber);
      processor->SetWipeValue(0);

      if (redo)
      {
        processor->Redo();
      }
      else
      {
        processor->Undo();
      }

      targetImage = mitk::ImportItkImage( processor->GetDestinationImage());
      targetNode->SetData(targetImage);
      targetNode->Modified();

      break;
    }
  case OP_RETAIN_MARKS:
    {
      try
      {
        mitk::OpRetainMarksNoThresholding *op = static_cast<mitk::OpRetainMarksNoThresholding*>(operation);
        mitk::OpRetainMarksNoThresholding::ProcessorType::Pointer processor = op->GetProcessor();
        bool redo = op->IsRedo();
        int sourceSlice = op->GetSourceSlice();
        int targetSlice = op->GetTargetSlice();
        itk::ORIENTATION_ENUM orientation = op->GetOrientation();

        mitk::DataNode::Pointer targetNode = op->GetTargetNode();
        assert(targetNode);

        mitk::Image::Pointer targetImage = dynamic_cast<mitk::Image*>(targetNode->GetData());
        assert(targetImage);

        typedef mitk::ImageToItk< SegmentationImageType > SegmentationImageToItkType;
        SegmentationImageToItkType::Pointer targetImageToItk = SegmentationImageToItkType::New();
        targetImageToItk->SetInput(targetImage);
        targetImageToItk->Update();

        processor->SetSourceImage(targetImageToItk->GetOutput());
        processor->SetDestinationImage(targetImageToItk->GetOutput());
        processor->SetSlices(orientation, sourceSlice, targetSlice);

        if (redo)
        {
          processor->Redo();
        }
        else
        {
          processor->Undo();
        }

        targetImage = mitk::ImportItkImage( processor->GetDestinationImage());

        targetNode->SetData(targetImage);
        targetNode->Modified();
      }
      catch( itk::ExceptionObject & err )
      {
        MITK_ERROR << "Failed to update due to: " << err << std::endl;
      }
      break;
    }
  default:;
  }
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASGeneralSegmentorView::OnThresholdApplyButtonPressed()
{
  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer workingData = toolManager->GetWorkingData(0);
  assert(workingData);

  mitk::DataNode::Pointer regionGrowingNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingData, true);
  assert(regionGrowingNode);

  this->m_GeneralControls->m_ThresholdCheckBox->setChecked(false);
  QmitkMIDASBaseSegmentationFunctionality::WipeTools();

  mitk::OpThresholdApply::ProcessorType::Pointer processor = mitk::OpThresholdApply::ProcessorType::New();
  mitk::OpThresholdApply *doOp = new mitk::OpThresholdApply(OP_THRESHOLD_APPLY, true, workingData, regionGrowingNode, processor);
  mitk::OpThresholdApply *undoOp = new mitk::OpThresholdApply(OP_THRESHOLD_APPLY, false, workingData, regionGrowingNode, processor);
  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Apply threshold");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

  ExecuteOperation(doOp);
}

void MIDASGeneralSegmentorView::OnWipeButtonPressed()
{
  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer workingData = toolManager->GetWorkingData(0);
  assert(workingData);

  this->m_GeneralControls->m_ThresholdCheckBox->setChecked(false);
  QmitkMIDASBaseSegmentationFunctionality::WipeTools();

  int sliceNumber = this->GetSliceNumber();
  itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();

  mitk::OpWipeSlice::WipeSliceProcessorType::Pointer processor = mitk::OpWipeSlice::WipeSliceProcessorType::New();
  mitk::OpWipeSlice *doOp = new mitk::OpWipeSlice(OP_WIPE_SLICE, true, workingData, sliceNumber, orientation, processor);
  mitk::OpWipeSlice *undoOp = new mitk::OpWipeSlice(OP_WIPE_SLICE, false, workingData, sliceNumber, orientation, processor);
  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Wipe Slice");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

  ExecuteOperation(doOp);
}

void MIDASGeneralSegmentorView::OnWipePlusButtonPressed()
{
  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer workingData = toolManager->GetWorkingData(0);
  assert(workingData);

  int sliceNumber = this->GetSliceNumber();
  itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();

  mitk::OpWipePlus::WipePlusProcessorType::Pointer processor = mitk::OpWipePlus::WipePlusProcessorType::New();
  mitk::OpWipePlus *doOp = new mitk::OpWipePlus(OP_WIPE_PLUS, true, workingData, sliceNumber, orientation, processor);
  mitk::OpWipePlus *undoOp = new mitk::OpWipePlus(OP_WIPE_PLUS, false, workingData, sliceNumber, orientation, processor);
  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Wipe Plus");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

  ExecuteOperation(doOp);
}

void MIDASGeneralSegmentorView::OnWipeMinusButtonPressed()
{
  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer workingData = toolManager->GetWorkingData(0);
  assert(workingData);

  int sliceNumber = this->GetSliceNumber();
  itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();

  mitk::OpWipeMinus::WipeMinusProcessorType::Pointer processor = mitk::OpWipeMinus::WipeMinusProcessorType::New();
  mitk::OpWipeMinus *doOp = new mitk::OpWipeMinus(OP_WIPE_MINUS, true, workingData, sliceNumber, orientation, processor);
  mitk::OpWipeMinus *undoOp = new mitk::OpWipeMinus(OP_WIPE_MINUS, false, workingData, sliceNumber, orientation, processor);
  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Wipe Minus");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

  ExecuteOperation(doOp);
}

void MIDASGeneralSegmentorView::OnPropagate3DButtonPressed()
{
  this->OnPropagateUpButtonPressed();
  this->OnPropagateDownButtonPressed();
}

void MIDASGeneralSegmentorView::OnPropagateUpButtonPressed()
{
  this->DoPropagate(true);
}

void MIDASGeneralSegmentorView::OnPropagateDownButtonPressed()
{
  this->DoPropagate(false);
}

void MIDASGeneralSegmentorView::DoPropagate(bool isUp)
{
  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::MIDASDrawTool *drawTool = static_cast<mitk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASDrawTool>()));
  assert(drawTool);

  mitk::MIDASPolyTool *polyTool = static_cast<mitk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
  assert(polyTool);

  mitk::DataNode::Pointer workingData = toolManager->GetWorkingData(0);
  assert(workingData);

  mitk::DataNode::Pointer regionGrowingNode = this->GetDefaultDataStorage()->GetNamedDerivedNode(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME.c_str(), workingData, true);
  assert(regionGrowingNode);

  mitk::DataNode::Pointer referenceNode = this->GetReferenceNodeFromToolManager();
  assert(referenceNode);

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  assert(referenceImage);

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  double lowerThreshold = this->m_GeneralControls->m_ThresholdLowerSliderWidget->value();
  double upperThreshold = this->m_GeneralControls->m_ThresholdUpperSliderWidget->value();
  int sliceNumber = this->GetSliceNumber();
  itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();

  mitk::OperationEvent* operationEvent = NULL;

  try
  {
    AccessFixedDimensionByItk_n(referenceImage, CreateAndPopulatePropagateProcessor, 3,
          (
            regionGrowingNode,
            referenceNode,
            *seeds,
            *drawTool,
            *polyTool,
            sliceNumber,
            orientation,
            isUp,
            lowerThreshold,
            upperThreshold,
            operationEvent
          )
        );
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception, so abandoning CreateAndPopulatePropagateProcessor." << e.what();
    return;
  }

  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
  ExecuteOperation(operationEvent->GetOperation());
}

template <typename TGreyScalePixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::CreateAndPopulatePropagateProcessor(
    itk::Image<TGreyScalePixel, VImageDimension>* greyScaleImage,
    mitk::DataNode* regionGrowingNode,
    mitk::DataNode* referenceNode,
    mitk::PointSet &seeds,
    mitk::MIDASDrawTool &drawTool,
    mitk::MIDASPolyTool &polyTool,
    int sliceNumber,
    itk::ORIENTATION_ENUM orientation,
    bool isUp,
    double lowerThreshold,
    double upperThreshold,
    mitk::OperationEvent*& eventOutput
    )
{
  try
  {

    typedef itk::MIDASPropagateProcessor<mitk::Tool::DefaultSegmentationDataType, TGreyScalePixel, PointSetPixelType, VImageDimension> PropagateProcessorType;
    typedef itk::MIDASPropagateUpProcessor<mitk::Tool::DefaultSegmentationDataType, TGreyScalePixel, PointSetPixelType, VImageDimension> PropagateUpProcessorType;
    typedef itk::MIDASPropagateDownProcessor<mitk::Tool::DefaultSegmentationDataType, TGreyScalePixel, PointSetPixelType, VImageDimension> PropagateDownProcessorType;
    typedef mitk::OpPropagateUp<mitk::Tool::DefaultSegmentationDataType, TGreyScalePixel, PointSetPixelType, VImageDimension> PropagateUpOperationType;
    typedef mitk::OpPropagateDown<mitk::Tool::DefaultSegmentationDataType, TGreyScalePixel, PointSetPixelType, VImageDimension> PropagateDownOperationType;

    PointSetPointer itkSeeds = PointSetType::New();
    ConvertMITKSeedsAndAppendToITKSeeds(&seeds, itkSeeds);

    PointSetPointer itkContours = PointSetType::New();
    ConvertMITKContoursFromOneToolAndAppendToITKPoints(&drawTool, itkContours);
    ConvertMITKContoursFromOneToolAndAppendToITKPoints(&polyTool, itkContours);

    if (isUp)
    {
      typename PropagateUpProcessorType::Pointer upProcessor = PropagateUpProcessorType::New();
      upProcessor->SetGreyScaleImage(greyScaleImage);
      upProcessor->SetSeeds(itkSeeds.GetPointer());
      upProcessor->SetContours(itkContours.GetPointer());
      upProcessor->SetLowerThreshold((TGreyScalePixel)lowerThreshold);
      upProcessor->SetUpperThreshold((TGreyScalePixel)upperThreshold);
      PropagateUpOperationType *doOp = new PropagateUpOperationType(OP_PROPAGATE_UP, true, regionGrowingNode, referenceNode, sliceNumber, orientation, upProcessor);
      PropagateUpOperationType *undoOp = new PropagateUpOperationType(OP_PROPAGATE_UP, false, regionGrowingNode, referenceNode, sliceNumber, orientation, upProcessor);
      eventOutput = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Propagate Up");
    }
    else // isDown :-)
    {
      typename PropagateDownProcessorType::Pointer downProcessor = PropagateDownProcessorType::New();
      downProcessor->SetGreyScaleImage(greyScaleImage);
      downProcessor->SetSeeds(itkSeeds.GetPointer());
      downProcessor->SetContours(itkContours.GetPointer());
      downProcessor->SetLowerThreshold((TGreyScalePixel)lowerThreshold);
      downProcessor->SetUpperThreshold((TGreyScalePixel)upperThreshold);
      PropagateDownOperationType *doOp = new PropagateDownOperationType(OP_PROPAGATE_DOWN, true, regionGrowingNode, referenceNode, sliceNumber, orientation, downProcessor);
      PropagateDownOperationType *undoOp = new PropagateDownOperationType(OP_PROPAGATE_DOWN, false, regionGrowingNode, referenceNode, sliceNumber, orientation, downProcessor);
      eventOutput = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Propagate Down");
    }
  }
  catch( itk::ExceptionObject & err )
  {
    MITK_ERROR << "Failed to populate processor due to: " << err << std::endl;
  }
}

template <typename TGreyScalePixel, unsigned int VImageDimension>
void
MIDASGeneralSegmentorView
::RunPropagateProcessor(
    itk::Image<TGreyScalePixel, VImageDimension>* greyScaleImage,
    mitk::OpPropagate *op,
    mitk::DataNode* regionGrowingNode,
    bool redo,
    int sliceNumber,
    itk::ORIENTATION_ENUM orientation
)
{
  typedef itk::MIDASRegionProcessor<mitk::Tool::DefaultSegmentationDataType, VImageDimension> RegionProcessorType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, VImageDimension> SegmentationImageType;
  typedef mitk::ImageToItk< SegmentationImageType > SegmentationImageToItkType;

  assert(regionGrowingNode);

  mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
  assert(regionGrowingImage);

  typename SegmentationImageToItkType::Pointer regionGrowingImageToItk = SegmentationImageToItkType::New();
  regionGrowingImageToItk->SetInput(regionGrowingImage);
  regionGrowingImageToItk->Update();

  typename RegionProcessorType::Pointer processor = op->GetProcessor();
  processor->SetDestinationImage(regionGrowingImageToItk->GetOutput());
  processor->SetOrientationAndSlice(orientation, sliceNumber);

  if (redo)
  {
    processor->Redo();
  }
  else
  {
    processor->Undo();
  }

  regionGrowingImage = mitk::ImportItkImage(processor->GetDestinationImage());
  regionGrowingNode->SetData(regionGrowingImage);
  regionGrowingNode->Modified();
}

void MIDASGeneralSegmentorView::OnSliceNumberChanged(int before, int after)
{
  if (this->m_GeneralControls->m_RetainMarksCheckBox->isChecked() && !this->m_GeneralControls->m_ThresholdCheckBox->isChecked() && before != after)
  {
    mitk::ToolManager *toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::DataNode::Pointer workingData = toolManager->GetWorkingData(0);
    assert(workingData);

    itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();

    mitk::OpRetainMarksNoThresholding::ProcessorType::Pointer processor = mitk::OpRetainMarksNoThresholding::ProcessorType::New();
    mitk::OpRetainMarksNoThresholding *doOp = new mitk::OpRetainMarksNoThresholding(OP_RETAIN_MARKS, true, workingData, after, before, orientation, processor);
    mitk::OpRetainMarksNoThresholding *undoOp = new mitk::OpRetainMarksNoThresholding(OP_RETAIN_MARKS, false, workingData, after, before, orientation, processor);
    mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Retain Marks");
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

    ExecuteOperation(doOp);
  }

  this->UpdateRegionGrowing();
  this->UpdatePriorAndNext();

  // Synch m_MultiWidget with this view.
  if (this->m_MultiWidget)
  {
    mitk::Image::Pointer referenceImage = this->GetReferenceImage();
    if (referenceImage.IsNotNull())
    {
      mitk::Index3D crossPositionInVoxels;
      mitk::Point3D crossPositionInMillimetres = m_MultiWidget->GetCrossPosition();
      referenceImage->GetGeometry()->WorldToIndex(crossPositionInMillimetres, crossPositionInVoxels);
      int axis = this->GetAxis();
      crossPositionInVoxels[axis] = this->m_GeneralControls->m_SliceSelectionWidget->GetValue();

      mitk::Point3D newVoxelPosition;
      for (int i = 0; i < 3; i++)
      {
        newVoxelPosition[i] = crossPositionInVoxels[i];
      }
      referenceImage->GetGeometry()->IndexToWorld(newVoxelPosition, crossPositionInMillimetres);
      this->m_MultiWidget->MoveCrossToPosition(crossPositionInMillimetres);
    }
  }
}

void MIDASGeneralSegmentorView::OnCleanButtonPressed()
{
  std::cerr << "TODO clean button pressed, which should simplify seeds/contours" << std::endl;
}

void MIDASGeneralSegmentorView::OnRetainMarksCheckBoxToggled(bool b)
{
  // Actually nothing to do until you move slice, then the current slice gets propagated.
}
