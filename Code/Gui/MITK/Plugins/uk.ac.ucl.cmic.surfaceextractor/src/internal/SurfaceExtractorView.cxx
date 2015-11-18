/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceExtractorView.h"
#include "SurfaceExtractorPreferencePage.h"

#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <berryIWorkbenchPart.h>
#include <berryIWorkbenchPage.h>
#include <berryISelection.h>
#include <berryISelectionProvider.h>
#include <berrySingleNodeSelection.h>

#include <itkEventObject.h>

#include <mitkImage.h>
#include <mitkNodePredicateBase.h>
#include <mitkNodePredicateDimension.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateOr.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateNot.h>
#include <mitkColorSequenceRainbow.h>
#include <mitkImage.h>
#include <mitkSurface.h>
#include <mitkManualSegmentationToSurfaceFilter.h>
#include <mitkLabeledImageToSurfaceFilter.h>
#include <mitkIDataStorageService.h>
#include <mitkNodePredicateDataType.h>

#include <mitkNifTKImageToSurfaceFilter.h>

#include <QButtonGroup>
#include <QSlider>
#include <QApplication>
#include <QMessageBox>
#include <QKeyEvent>

#include <cstdio>
#include <limits>

#include <berryISelectionListener.h>
#include <berryIStructuredSelection.h>

class NodePredicateLabelImage : public mitk::NodePredicateBase
{
public:
  mitkClassMacro(NodePredicateLabelImage, NodePredicateBase);
  itkNewMacro(NodePredicateLabelImage);

  //##Documentation
  //## @brief Standard Destructor
  virtual ~NodePredicateLabelImage() {}

  //##Documentation
  //## @brief Checks, if the nodes contains a property that is equal to m_ValidProperty
  virtual bool CheckNode(const mitk::DataNode* node) const
  {
    if (node == NULL)
    {
      throw std::invalid_argument("NodePredicateLabelImage: invalid node");
    }

    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (!image)
    {
      return false;
    }

    mitk::PixelType pixelType = image->GetPixelType();
    if (pixelType.GetPixelType() == itk::ImageIOBase::SCALAR
        && (pixelType.GetComponentType() == itk::ImageIOBase::CHAR || pixelType.GetComponentType() == itk::ImageIOBase::UCHAR))
    {
      return true;
    }

    return false;
  }
};


class SurfaceExtractorViewPrivate {
public:
  SurfaceExtractorViewPrivate();

  mitk::NodePredicateBase::Pointer has3dImage;
  mitk::NodePredicateBase::Pointer has3dOr4dImage;
  mitk::NodePredicateBase::Pointer has3dGrayScaleImage;
  mitk::NodePredicateBase::Pointer has3dOr4dGrayScaleImage;
  mitk::NodePredicateBase::Pointer has3dLabelImage;
  mitk::NodePredicateBase::Pointer has3dOr4dLabelImage;
  mitk::NodePredicateBase::Pointer has3dSurfaceImage;
  mitk::NodePredicateBase::Pointer has3dOr4dSurfaceImage;

  mitk::DataNode::Pointer                       m_ReferenceNode;
  mitk::DataStorage::SetOfObjects::ConstPointer m_SurfaceNodes;
  mitk::DataNode::Pointer                       m_SurfaceNode;
  mitk::ColorSequenceRainbow                    m_RainbowColor;

  bool   m_AlwaysCreateNewSurface;
  bool   m_Dirty;
  bool   m_IsVisible;
  bool   m_IsActivated;
  float  m_Threshold;
  
  mitk::NifTKImageToSurfaceFilter
    ::SurfaceExtractionMethod    m_SurfaceExtractionType;
  
  mitk::NifTKImageToSurfaceFilter
    ::InputSmoothingMethod       m_InputSmoothingType;
  int                            m_InputSmoothingIterations;
  float                          m_InputSmoothingRadius;

  mitk::NifTKImageToSurfaceFilter
    ::SurfaceSmoothingMethod     m_SurfaceSmoothingType;
  int                            m_SurfaceSmoothingIterations;
  float                          m_SurfaceSmoothingParameter;

  mitk::NifTKImageToSurfaceFilter
    ::SurfaceDecimationMethod    m_SurfaceDecimationType;
  double                         m_TargetReduction;

  bool                           m_PerformSurfaceCleaning;
  float                          m_SurfaceCleaningThreshold;

  double                         m_SamplingRatio;
};

SurfaceExtractorViewPrivate::SurfaceExtractorViewPrivate()
{
  mitk::NodePredicateDimension::Pointer has3dImage =
      mitk::NodePredicateDimension::New(3);

  mitk::NodePredicateDimension::Pointer has4dImage =
      mitk::NodePredicateDimension::New(4);

  mitk::NodePredicateOr::Pointer has3dOr4dImage =
      mitk::NodePredicateOr::New(has3dImage, has4dImage);

  mitk::TNodePredicateDataType<mitk::Image>::Pointer hasImage =
      mitk::TNodePredicateDataType<mitk::Image>::New();

  mitk::NodePredicateProperty::Pointer isBinary =
      mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));

  mitk::NodePredicateNot::Pointer isNotBinary =
      mitk::NodePredicateNot::New(isBinary);

  mitk::NodePredicateAnd::Pointer has3dNotBinaryImage =
      mitk::NodePredicateAnd::New(has3dImage, isNotBinary);

  mitk::NodePredicateAnd::Pointer has3dOr4dNotBinaryImage =
      mitk::NodePredicateAnd::New(has3dOr4dImage, isNotBinary);

  mitk::NodePredicateAnd::Pointer hasBinaryImage =
      mitk::NodePredicateAnd::New(hasImage, isBinary);

  mitk::NodePredicateAnd::Pointer has3dBinaryImage =
      mitk::NodePredicateAnd::New(hasBinaryImage, has3dImage);

  mitk::NodePredicateAnd::Pointer has3dOr4dBinaryImage =
      mitk::NodePredicateAnd::New(has3dOr4dImage, hasBinaryImage);

  // TODO incorrect definition:
//  mitk::NodePredicateAnd::Pointer hasLabelImage =
//      mitk::NodePredicateAnd::New(hasImage, isBinary);
  NodePredicateLabelImage::Pointer hasLabelImage =
      NodePredicateLabelImage::New();

  mitk::NodePredicateAnd::Pointer has3dLabelImage =
      mitk::NodePredicateAnd::New(has3dImage, hasLabelImage);

  mitk::NodePredicateAnd::Pointer has3dOr4dLabelImage =
      mitk::NodePredicateAnd::New(has3dOr4dImage, hasLabelImage);

  mitk::NodePredicateProperty::Pointer hasSurfaceImage =
      mitk::NodePredicateProperty::New("Surface", mitk::BoolProperty::New(true));

  this->has3dImage = has3dImage;
  this->has3dOr4dImage = has3dOr4dImage;
  this->has3dGrayScaleImage = has3dNotBinaryImage;
  this->has3dOr4dGrayScaleImage = has3dOr4dNotBinaryImage;
  this->has3dLabelImage = has3dLabelImage;
  this->has3dOr4dLabelImage = has3dOr4dLabelImage;
  this->has3dOr4dSurfaceImage = hasSurfaceImage;
}

const std::string SurfaceExtractorView::VIEW_ID = "uk.ac.ucl.cmic.SurfaceExtractor";

SurfaceExtractorView::SurfaceExtractorView()
: m_Controls(0)
, m_Parent(0)
, d_ptr(new SurfaceExtractorViewPrivate())
{
  Q_D(SurfaceExtractorView);

  d->m_AlwaysCreateNewSurface = false;
  d->m_Dirty = false;
  d->m_IsVisible = true;
  d->m_IsActivated = false;
  d->m_Threshold = 100;

  d->m_SurfaceExtractionType = mitk::NifTKImageToSurfaceFilter::StandardExtractor;

  d->m_InputSmoothingType = mitk::NifTKImageToSurfaceFilter::NoInputSmoothing;
  d->m_InputSmoothingIterations = 1;
  d->m_InputSmoothingRadius = 0.5;

  d->m_SurfaceSmoothingType = mitk::NifTKImageToSurfaceFilter::NoSurfaceSmoothing;
  d->m_SurfaceSmoothingIterations = 10;
  d->m_SurfaceSmoothingParameter = 0.5;

  d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::NoDecimation;
  d->m_TargetReduction = 0.1;

  d->m_PerformSurfaceCleaning = true;
  d->m_SurfaceCleaningThreshold = 1000;
  d->m_SamplingRatio = 0.75;
}

SurfaceExtractorView::~SurfaceExtractorView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}

void SurfaceExtractorView::RetrievePreferenceValues()
{
  Q_D(SurfaceExtractorView);
/*
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    d->gaussianSmooth = prefs->GetBool(SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_NAME, SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_DEFAULT);
    d->gaussianStdDev = prefs->GetDouble(SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_NAME, SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_DEFAULT);
    d->threshold = prefs->GetDouble(SurfaceExtractorPreferencePage::THRESHOLD_NAME, SurfaceExtractorPreferencePage::THRESHOLD_DEFAULT);
    d->targetReduction = prefs->GetDouble(SurfaceExtractorPreferencePage::TARGET_REDUCTION_NAME, SurfaceExtractorPreferencePage::TARGET_REDUCTION_DEFAULT);
    d->maxNumberOfPolygons = prefs->GetLong(SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_NAME, SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_DEFAULT);
  }
  else
  {
    MITK_INFO << "SurfaceExtractorView::RetrievePreferenceValues() no preferences found";
    d->gaussianSmooth = SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_DEFAULT;
    d->gaussianStdDev = SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_DEFAULT;
    d->threshold = SurfaceExtractorPreferencePage::THRESHOLD_DEFAULT;
    d->targetReduction = SurfaceExtractorPreferencePage::TARGET_REDUCTION_DEFAULT;
    d->maxNumberOfPolygons = SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_DEFAULT;
  }
*/
}

void SurfaceExtractorView::CreateQtPartControl(QWidget *parent)
{
  // setup the basic GUI of this view
  m_Parent = parent;

  if (!m_Controls)
  {
    // Create UI.
    m_Controls = new Ui::SurfaceExtractorViewControls();
    m_Controls->setupUi(parent);
    m_Controls->wgt_advancedControls->hide();

    connect(m_Controls->cbx_showAdvanced, SIGNAL(stateChanged(int )), this, SLOT(OnAdvancedFeaturesToggled(int )));
    connect(m_Controls->cmbx_extractionMethod, SIGNAL(currentIndexChanged(int )), this, SLOT(OnExtractionMethodChanged(int )));
    connect(m_Controls->btn_apply, SIGNAL(clicked( )), this, SLOT(OnApplyClicked()));

    // Retrieve and store preference values.
    RetrievePreferenceValues();

    UpdateFields();
  }
}

void SurfaceExtractorView::Activated()
{
  berry::IWorkbenchPart::Pointer nullPart;
  OnSelectionChanged(nullPart, this->GetDataManagerSelection());
}

bool SurfaceExtractorView::eventFilter(QObject *obj, QEvent *event)
{
  if (event->type() == QEvent::KeyPress)
  {
    QKeyEvent *keyEvent = dynamic_cast<QKeyEvent *>(event);
    switch (keyEvent->key())
    {
    case Qt::Key_Return:
      OnApplyClicked();
      return true;
    case Qt::Key_Escape:
      UpdateFields();
      Q_D(SurfaceExtractorView);
      mitk::Surface* surface = dynamic_cast<mitk::Surface*>(d->m_SurfaceNode->GetData());
      m_Controls->btn_apply->setEnabled(surface->IsEmpty());
      return true;
    }
  }
  // standard event processing
  return QObject::eventFilter(obj, event);
}

void SurfaceExtractorView::UpdateFields()
{
  Q_D(SurfaceExtractorView);

}

void SurfaceExtractorView::SetFocus()
{
}

void SurfaceExtractorView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  RetrievePreferenceValues();
}

void SurfaceExtractorView::EnableControls(bool b)
{
  m_Controls->cmbx_inSmoothMethod->setEnabled(b);
  m_Controls->spb_inSmoothRadius->setEnabled(b);
  m_Controls->spb_inSmoothIters->setEnabled(b);

  m_Controls->cmbx_extractionMethod->setEnabled(b);
  
  m_Controls->cmbx_decimationMethod->setEnabled(b);
  m_Controls->spb_targetReduction->setEnabled(b);
  
  m_Controls->cmbx_surfaceSmoothMethod->setEnabled(b);
  m_Controls->spb_surfaceSmoothIters->setEnabled(b);

  m_Controls->cbx_cleanSurface->setEnabled(b);
  m_Controls->spb_surfaceCleanThreshold->setEnabled(b);

  m_Controls->spb_threshold->setEnabled(b);
  m_Controls->btn_apply->setEnabled(b);

  m_Controls->dsbx_samplingRatio->setEnabled(b);

  OnExtractionMethodChanged(m_Controls->cmbx_extractionMethod->currentIndex() && b);
}

void SurfaceExtractorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  Q_D(SurfaceExtractorView);

  int numberOfSelectedNodes = nodes.size();
  if (numberOfSelectedNodes != 1)
  {
    DeselectNode();
    return;
  }

  mitk::DataNode::Pointer node = nodes[0];
  if (d->has3dOr4dImage->CheckNode(node))
  {
    SelectReferenceNode(node);
  }
  else if (d->has3dOr4dSurfaceImage->CheckNode(node))
  {
    SelectSurfaceNode(node);
  }
  else
  {
    DeselectNode();
  }
}

void SurfaceExtractorView::SelectReferenceNode(mitk::DataNode::Pointer node)
{
  Q_D(SurfaceExtractorView);
  d->m_ReferenceNode = node;
  d->m_SurfaceNodes = findSurfaceNodesOf(node);
  d->m_SurfaceNode = 0;

  mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
  if (image == 0)
  {
    EnableControls(false);
    return;
  }

  m_Controls->lblReferenceImage->setText(QString::fromStdString(node->GetName()));
  m_Controls->lblSurfaceImage->setText("No surface selected.");
  EnableControls(true);
}

void SurfaceExtractorView::SelectSurfaceNode(mitk::DataNode::Pointer node)
{
  Q_D(SurfaceExtractorView);
  d->m_SurfaceNode =  node;
  d->m_ReferenceNode = findReferenceNodeOf(node);
  d->m_SurfaceNodes =  findSurfaceNodesOf(d->m_ReferenceNode);

  LoadParameters();
  UpdateFields();

  m_Controls->lblReferenceImage->setText(QString::fromStdString(d->m_ReferenceNode->GetName()));
  m_Controls->lblSurfaceImage->setText(QString::fromStdString(d->m_SurfaceNode->GetName()));
  EnableControls(true);

  mitk::Surface* surface = dynamic_cast<mitk::Surface*>(d->m_SurfaceNode->GetData());
  m_Controls->btn_apply->setEnabled(surface->IsEmpty());
}

void SurfaceExtractorView::DeselectNode()
{
  Q_D(SurfaceExtractorView);
  m_Controls->lblReferenceImage->setText("No image selected.");
  m_Controls->lblSurfaceImage->setText("No surface selected.");
  d->m_ReferenceNode = 0;
  d->m_SurfaceNode = 0;
  d->m_SurfaceNodes = 0;
  EnableControls(false);

  m_Controls->btn_apply->setEnabled(false);
}

mitk::DataStorage::SetOfObjects::ConstPointer SurfaceExtractorView::findSurfaceNodesOf(mitk::DataNode::Pointer referenceNode)
{
  Q_D(SurfaceExtractorView);
  mitk::DataStorage::Pointer dataStorage = GetDataStorage();
  if (dataStorage.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::findSurfaceNodeOf(mitk::DataNode::Pointer referenceNode): No data storage.";
    return 0;
  }

  return dataStorage->GetDerivations(referenceNode, d->has3dOr4dSurfaceImage, true);
}

mitk::DataNode::Pointer SurfaceExtractorView::findReferenceNodeOf(mitk::DataNode::Pointer surfaceNode)
{
  Q_D(SurfaceExtractorView);
  mitk::DataStorage::Pointer dataStorage = GetDataStorage();
  mitk::DataStorage::SetOfObjects::ConstPointer referenceNodes = dataStorage->GetSources(surfaceNode, d->has3dOr4dImage, true);
  if (referenceNodes->Size() != 1)
  {
    return 0;
  }
  return referenceNodes->GetElement(0);
}

void SurfaceExtractorView::OnApplyClicked()
{
  Q_D(SurfaceExtractorView);
  d->m_Threshold = m_Controls->spb_threshold->value();
  d->m_InputSmoothingIterations = m_Controls->spb_inSmoothIters->value();
  d->m_InputSmoothingRadius = m_Controls->spb_inSmoothRadius->value();
  d->m_SurfaceSmoothingIterations = m_Controls->spb_surfaceSmoothIters->value();
  d->m_TargetReduction = m_Controls->spb_targetReduction->value();

  d->m_PerformSurfaceCleaning = m_Controls->cbx_cleanSurface->isChecked();
  d->m_SurfaceCleaningThreshold = m_Controls->spb_surfaceCleanThreshold->value();
  d->m_SamplingRatio = m_Controls->dsbx_samplingRatio->value();

  int index = m_Controls->cmbx_inSmoothMethod->currentIndex();

  switch (index)
  {
    case 0:
      d->m_InputSmoothingType = mitk::NifTKImageToSurfaceFilter::NoInputSmoothing;
      break;
    case 1:
      d->m_InputSmoothingType = mitk::NifTKImageToSurfaceFilter::MedianSmoothing;
      break;
    case 2:
      d->m_InputSmoothingType = mitk::NifTKImageToSurfaceFilter::GaussianSmoothing;
      break;
  }

  index = m_Controls->cmbx_extractionMethod->currentIndex();
  switch (index)
  {
    case 0:
      d->m_SurfaceExtractionType = mitk::NifTKImageToSurfaceFilter::StandardExtractor;
      break;
    case 1:
      d->m_SurfaceExtractionType = mitk::NifTKImageToSurfaceFilter::EnhancedCPUExtractor;
      break;
    case 2:
      d->m_SurfaceExtractionType = mitk::NifTKImageToSurfaceFilter::GPUExtractor;
      break;
  }

  index = m_Controls->cmbx_decimationMethod->currentIndex();
  switch (index)
  {
    case 0:
      d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::NoDecimation;
      break;
    case 1:
      d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::DecimatePro;
      break;
    case 2:
      d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::QuadricVTK;
      break;
    case 3:
      d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::Quadric;
      break;
    case 4:
      d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::QuadricTri;
      break;
    case 5:
      d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::Melax;
      break;
    case 6:
      d->m_SurfaceDecimationType = mitk::NifTKImageToSurfaceFilter::ShortestEdge;
      break;
  }
  
  index = m_Controls->cmbx_surfaceSmoothMethod->currentIndex();
  switch (index)
  {
    case 0:
      d->m_SurfaceSmoothingType = mitk::NifTKImageToSurfaceFilter::NoSurfaceSmoothing;
      break;
    case 1:
      d->m_SurfaceSmoothingType = mitk::NifTKImageToSurfaceFilter::StandardVTKSmoothing;
      break;
    case 2:
      d->m_SurfaceSmoothingType = mitk::NifTKImageToSurfaceFilter::WindowedSincSmoothing;
      break;
    case 3:
      d->m_SurfaceSmoothingType = mitk::NifTKImageToSurfaceFilter::TaubinSmoothing;
      break;
    case 4:
      d->m_SurfaceSmoothingType = mitk::NifTKImageToSurfaceFilter::CurvatureNormalSmooth;
      break;
    case 5:
      d->m_SurfaceSmoothingType = mitk::NifTKImageToSurfaceFilter::InverseEdgeLengthSmooth;
      break;
  }

  if (d->m_SurfaceNode.IsNull() || d->m_AlwaysCreateNewSurface)
    CreateSurfaceNode();

  UpdateSurfaceNode();
}

void SurfaceExtractorView::CreateSurfaceNode()
{
  Q_D(SurfaceExtractorView);

  if (d->m_ReferenceNode.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::CreateSurfaceNode(): No reference image. The button should be disabled.";
    return;
  }
  if (d->m_SurfaceNodes.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::CreateSurfaceNode(): No surface nodes. The button should be disabled.";
    return;
  }

  int maxSerial = 0;
  int surfaceNodeNumber = d->m_SurfaceNodes->Size();
  for (int i = 0; i < surfaceNodeNumber; ++i)
  {
    mitk::DataNode::Pointer currentSurfaceNode = d->m_SurfaceNodes->GetElement(i);
    std::string currentSurfaceName = currentSurfaceNode->GetName();
    int serial = 0;
    int ret = std::sscanf(currentSurfaceName.c_str(), "Surface %d", &serial);
    if (ret == 1 && serial > maxSerial)
    {
      maxSerial = serial;
    }
  }
  std::ostringstream newSurfaceName;
  newSurfaceName << "Surface " << (maxSerial + 1);

  d->m_SurfaceNode = mitk::DataNode::New();
  d->m_SurfaceNode->SetName(newSurfaceName.str());
  d->m_SurfaceNode->SetColor(d->m_RainbowColor.GetNextColor() );
  d->m_SurfaceNode->SetBoolProperty("Surface", true);
  d->m_SurfaceNode->SetVisibility(true);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  MITK_INFO << "create surface node - surface is empty: " << surface->IsEmpty();
  d->m_SurfaceNode->SetData(surface);

  RetrievePreferenceValues();
  UpdateFields();
  SaveParameters();

  GetDataStorage()->Add(d->m_SurfaceNode, d->m_ReferenceNode);
  d->m_SurfaceNodes = findSurfaceNodesOf(d->m_ReferenceNode);

  d->m_ReferenceNode->SetSelected(false);
  d->m_SurfaceNode->SetSelected(true);

  this->SetCurrentSelection(d->m_SurfaceNode);
}

void SurfaceExtractorView::UpdateSurfaceNode()
{
  Q_D(SurfaceExtractorView);

  if (d->m_ReferenceNode.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::UpdateSurfaceNode(): No reference image. The button should be disabled.";
    return;
  }

  if (d->m_SurfaceNode.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::UpdateSurfaceNode(): 12 no surface node is selected";
    return;
  }

  QApplication::setOverrideCursor( QCursor(Qt::WaitCursor) );

  mitk::Image* referenceImage = dynamic_cast<mitk::Image*>(d->m_ReferenceNode->GetData());

  if (!referenceImage)
  {
    MITK_INFO << "SurfaceExtractorView::createSurface(): No reference image. Should not arrive here.";
    return;
  }


  if (d->has3dOr4dLabelImage->CheckNode(d->m_ReferenceNode))
  {
    mitk::LabeledImageToSurfaceFilter::Pointer filter = mitk::LabeledImageToSurfaceFilter::New();
    filter->SetGaussianStandardDeviation(d->m_InputSmoothingRadius);
    if (d->m_InputSmoothingType == mitk::NifTKImageToSurfaceFilter::NoInputSmoothing)
      filter->SetSmooth(false);
    else
      filter->SetSmooth(true);
    
    filter->SetInput(referenceImage);
    //filter->SetThreshold(d->m_Threshold); // if( Gauss ) --> TH manipulated for vtkMarchingCube
    filter->SetTargetReduction(d->m_TargetReduction);
  
    // If the decimation value is non-zero we set the decimation type
    // Setting NoDecimation disables the whole processing
    if (d->m_SurfaceDecimationType == mitk::NifTKImageToSurfaceFilter::NoDecimation)
      filter->SetDecimate(mitk::ImageToSurfaceFilter::NoDecimation);
    else
      filter->SetDecimate(mitk::ImageToSurfaceFilter::DecimatePro);

    try
    {
      filter->Update();
    }
    catch (std::exception& exc)
    {
    }

    d->m_SurfaceNode->SetData(filter->GetOutput());
  }
  else
  {
    mitk::NifTKImageToSurfaceFilter::Pointer filter = mitk::NifTKImageToSurfaceFilter::New();
    filter->SetInput(referenceImage);
    filter->SetThreshold(d->m_Threshold);

    filter->SetSurfaceDecimationType(d->m_SurfaceDecimationType);
    filter->SetSurfaceExtractionType(d->m_SurfaceExtractionType);
    filter->SetSurfaceSmoothingType(d->m_SurfaceSmoothingType);
    filter->SetInputSmoothingType(d->m_InputSmoothingType);

    if (d->m_InputSmoothingType == mitk::NifTKImageToSurfaceFilter::NoInputSmoothing)
      filter->SetPerformInputSmoothing(false);
    else
      filter->SetPerformInputSmoothing(true);

    if (d->m_SurfaceSmoothingType == mitk::NifTKImageToSurfaceFilter::NoSurfaceSmoothing)
      filter->SetPerformSurfaceSmoothing(false);
    else
      filter->SetPerformSurfaceSmoothing(true);

    if (d->m_SurfaceDecimationType == mitk::NifTKImageToSurfaceFilter::NoDecimation)
      filter->SetPerformSurfaceDecimation(false);
    else
      filter->SetPerformSurfaceDecimation(true);

    filter->SetPerformSurfaceCleaning(d->m_PerformSurfaceCleaning);
    filter->SetSurfaceCleaningThreshold(d->m_SurfaceCleaningThreshold);
    filter->SetInputSmoothingIterations(d->m_InputSmoothingIterations);
    filter->SetInputSmoothingRadius(d->m_InputSmoothingRadius);
    filter->SetSurfaceSmoothingIterations(d->m_SurfaceSmoothingIterations);
    filter->SetSurfaceSmoothingRadius(d->m_SurfaceSmoothingParameter);
    filter->SetTargetReduction(d->m_TargetReduction);
    filter->SetSamplingRatio(d->m_SamplingRatio);

    try
    {
      filter->Update();
    }
    catch (std::exception& exc)
    {
    }

    d->m_SurfaceNode->SetData(filter->GetOutput());
  }

  int layer = 0;

  d->m_ReferenceNode->GetIntProperty("layer", layer);
  d->m_SurfaceNode->SetIntProperty("layer", layer + 1);
  d->m_SurfaceNode->SetProperty("Surface", mitk::BoolProperty::New(true));
  SaveParameters();

  RequestRenderWindowUpdate();

  d->m_Dirty = false;
  m_Controls->btn_apply->setEnabled(false);

  QApplication::restoreOverrideCursor();
}

void SurfaceExtractorView::SaveParameters()
{
  Q_D(SurfaceExtractorView);

/*
  d->m_SurfaceNode->SetBoolProperty(SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_NAME.c_str(), d->gaussianSmooth);
  d->m_SurfaceNode->SetFloatProperty(SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_NAME.c_str(), d->gaussianStdDev);
  d->m_SurfaceNode->SetFloatProperty(SurfaceExtractorPreferencePage::THRESHOLD_NAME.c_str(), d->threshold);
  d->m_SurfaceNode->SetFloatProperty(SurfaceExtractorPreferencePage::TARGET_REDUCTION_NAME.c_str(), d->targetReduction);
  d->m_SurfaceNode->SetIntProperty(SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_NAME.c_str(), d->maxNumberOfPolygons);
*/
}

void SurfaceExtractorView::LoadParameters()
{
  Q_D(SurfaceExtractorView);
/*
  bool gaussianSmooth;
  float gaussianStdDev;
  float threshold;
  float targetReduction;
  int maxNumberOfPolygons;

  d->m_SurfaceNode->GetBoolProperty(SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_NAME.c_str(), gaussianSmooth);
  d->m_SurfaceNode->GetFloatProperty(SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_NAME.c_str(), gaussianStdDev);
  d->m_SurfaceNode->GetFloatProperty(SurfaceExtractorPreferencePage::THRESHOLD_NAME.c_str(), threshold);
  d->m_SurfaceNode->GetFloatProperty(SurfaceExtractorPreferencePage::TARGET_REDUCTION_NAME.c_str(), targetReduction);
  d->m_SurfaceNode->GetIntProperty(SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_NAME.c_str(), maxNumberOfPolygons);

  d->gaussianSmooth = gaussianSmooth;
  d->gaussianStdDev = gaussianStdDev;
  d->threshold = threshold;
  d->targetReduction = targetReduction;
  d->maxNumberOfPolygons = maxNumberOfPolygons;
*/
}

void SurfaceExtractorView::OnAdvancedFeaturesToggled(int state)
{
  if (state != 2)
    m_Controls->wgt_advancedControls->setHidden(true);
  else
    m_Controls->wgt_advancedControls->setHidden(false);
}

void SurfaceExtractorView::OnExtractionMethodChanged(int which)
{
  if (which == 1) //the user is running CMC33
  {
    m_Controls->dsbx_samplingRatio->setEnabled(true);

    m_Controls->cmbx_surfaceSmoothMethod->setItemData(3, 33, Qt::UserRole - 1);
    m_Controls->cmbx_surfaceSmoothMethod->setItemData(4, 33, Qt::UserRole - 1);
    m_Controls->cmbx_surfaceSmoothMethod->setItemData(5, 33, Qt::UserRole - 1);
  }
  else
  {
    m_Controls->dsbx_samplingRatio->setEnabled(false);

    m_Controls->cmbx_surfaceSmoothMethod->setItemData(3, 0, Qt::UserRole - 1);
    m_Controls->cmbx_surfaceSmoothMethod->setItemData(4, 0, Qt::UserRole - 1);
    m_Controls->cmbx_surfaceSmoothMethod->setItemData(5, 0, Qt::UserRole - 1);
  }

}
