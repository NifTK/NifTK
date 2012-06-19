/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 17:52:47 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
    const std::type_info& pixelTypeId = pixelType.GetPixelTypeId();
    if (pixelTypeId == typeid(char) || pixelTypeId == typeid(unsigned char))
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

  mitk::DataNode::Pointer referenceNode;
  mitk::DataStorage::SetOfObjects::ConstPointer surfaceNodes;
  mitk::DataNode::Pointer surfaceNode;

  mitk::ColorSequenceRainbow rainbowColor;

  /// Threshold
  double threshold;

  /// Gaussian smoothing
  bool gaussianSmooth;

  /// Gaussian standard deviation
  double gaussianStdDev;

  // Value for DecimatePro
  float targetReduction;

  // Maximum number of polygons
  long maxNumberOfPolygons;

  bool dirty;

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

    m_Controls->spbThreshold->setMinimum(0);
    m_Controls->spbThreshold->setMaximum(std::numeric_limits<int>::max());
    m_Controls->spbMaxNumberOfPolygons->setMinimum(1);
    m_Controls->spbMaxNumberOfPolygons->setMaximum(std::numeric_limits<int>::max());
    m_Controls->spbTargetReduction->setMinimum(0.0);
    m_Controls->spbTargetReduction->setMaximum(1.0);
    m_Controls->spbTargetReduction->setSingleStep(0.05);

    int formRowHeight = m_Controls->spbThreshold->height();
    m_Controls->lblReferenceImage->setFixedHeight(formRowHeight);
    m_Controls->lblSurfaceImage->setFixedHeight(formRowHeight);
    m_Controls->cbxGaussianSmooth->setFixedHeight(formRowHeight);

    m_Controls->wgtControls->installEventFilter(this);

    connect(m_Controls->spbThreshold, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(m_Controls->cbxGaussianSmooth, SIGNAL(stateChanged(int)), this, SLOT(onValueChanged()));
    connect(m_Controls->spbGaussianStdDev, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(m_Controls->spbTargetReduction, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(m_Controls->spbMaxNumberOfPolygons, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged()));

    // Retrieve and store preference values.
    RetrievePreferenceValues();

    updateFields();

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
    CreateConnections();

    QList<mitk::DataNode::Pointer> selectedNodes = GetDataManagerSelection();
    berry::IWorkbenchPart::Pointer nullPart;
    OnSelectionChanged(nullPart, selectedNodes);
  }
}

void SurfaceExtractorView::onValueChanged()
{
  Q_D(SurfaceExtractorView);
  if (!d->dirty)
  {
    d->dirty = true;
    m_Controls->btnApply->setEnabled(true);
  }
}

bool SurfaceExtractorView::eventFilter(QObject *obj, QEvent *event)
{
  if (event->type() == QEvent::KeyPress)
  {
    QKeyEvent *keyEvent = dynamic_cast<QKeyEvent *>(event);
    switch (keyEvent->key())
    {
    case Qt::Key_Return:
      on_btnApply_clicked();
      return true;
    case Qt::Key_Escape:
      updateFields();
      Q_D(SurfaceExtractorView);
      mitk::Surface* surface = dynamic_cast<mitk::Surface*>(d->surfaceNode->GetData());
      m_Controls->btnApply->setEnabled(surface->IsEmpty());
      return true;
    }
  }
  // standard event processing
  return QObject::eventFilter(obj, event);
}

void SurfaceExtractorView::updateFields()
{
  Q_D(SurfaceExtractorView);
  m_Controls->cbxGaussianSmooth->setChecked(d->gaussianSmooth);
  m_Controls->spbGaussianStdDev->setValue(d->gaussianStdDev);
  m_Controls->spbThreshold->setValue(d->threshold);
  m_Controls->spbTargetReduction->setValue(d->targetReduction);
  m_Controls->spbMaxNumberOfPolygons->setValue(d->maxNumberOfPolygons);
}

void SurfaceExtractorView::CreateConnections()
{
  connect(m_Controls->cbxGaussianSmooth, SIGNAL(toggled(bool)), this, SLOT(on_cbxGaussianSmooth_toggled(bool)));
  connect(m_Controls->btnCreate, SIGNAL(clicked()), this, SLOT(on_btnCreate_clicked()));
  connect(m_Controls->btnApply, SIGNAL(clicked()), this, SLOT(on_btnApply_clicked()));
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
  m_Controls->cbxGaussianSmooth->setEnabled(b);
  m_Controls->spbGaussianStdDev->setEnabled(b && m_Controls->cbxGaussianSmooth->isChecked());
  m_Controls->spbThreshold->setEnabled(b);
  m_Controls->spbTargetReduction->setEnabled(b);
  m_Controls->spbMaxNumberOfPolygons->setEnabled(b);
}

void SurfaceExtractorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  Q_D(SurfaceExtractorView);

  int numberOfSelectedNodes = nodes.size();
  if (numberOfSelectedNodes != 1)
  {
    deselectNode();
    return;
  }

  mitk::DataNode::Pointer node = nodes[0];
  if (d->has3dOr4dImage->CheckNode(node))
  {
    selectReferenceNode(node);
  }
  else if (d->has3dOr4dSurfaceImage->CheckNode(node))
  {
    selectSurfaceNode(node);
  }
  else
  {
    deselectNode();
  }
}

void SurfaceExtractorView::selectReferenceNode(mitk::DataNode::Pointer node)
{
  Q_D(SurfaceExtractorView);
  d->referenceNode = node;
  d->surfaceNodes = findSurfaceNodesOf(node);
  d->surfaceNode = 0;

  m_Controls->lblReferenceImage->setText(QString::fromStdString(node->GetName()));
  m_Controls->lblSurfaceImage->setText("No surface selected.");
  EnableControls(false);
  m_Controls->btnCreate->setEnabled(true);
  m_Controls->btnApply->setEnabled(false);
}

void SurfaceExtractorView::selectSurfaceNode(mitk::DataNode::Pointer node)
{
  Q_D(SurfaceExtractorView);
  d->surfaceNode =  node;
  d->referenceNode = findReferenceNodeOf(node);
  d->surfaceNodes =  findSurfaceNodesOf(d->referenceNode);

  loadParameters();
  updateFields();

  m_Controls->lblReferenceImage->setText(QString::fromStdString(d->referenceNode->GetName()));
  m_Controls->lblSurfaceImage->setText(QString::fromStdString(d->surfaceNode->GetName()));
  EnableControls(true);
  m_Controls->btnCreate->setEnabled(true);
  mitk::Surface* surface = dynamic_cast<mitk::Surface*>(d->surfaceNode->GetData());
  m_Controls->btnApply->setEnabled(surface->IsEmpty());
}

void SurfaceExtractorView::deselectNode()
{
  Q_D(SurfaceExtractorView);
  m_Controls->lblReferenceImage->setText("No image selected.");
  m_Controls->lblSurfaceImage->setText("No surface selected.");
  d->referenceNode = 0;
  d->surfaceNode = 0;
  d->surfaceNodes = 0;
  EnableControls(false);
  m_Controls->btnCreate->setEnabled(false);
  m_Controls->btnApply->setEnabled(false);
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
  MITK_INFO << "SurfaceExtractorView::findReferenceNodeOf(mitk::DataNode::Pointer surfaceNode)";
  Q_D(SurfaceExtractorView);
  mitk::DataStorage::Pointer dataStorage = GetDataStorage();
  mitk::DataStorage::SetOfObjects::ConstPointer referenceNodes = dataStorage->GetSources(surfaceNode, d->has3dOr4dImage, true);
  if (referenceNodes->Size() != 1)
  {
    return 0;
  }
  return referenceNodes->GetElement(0);
}

void SurfaceExtractorView::on_btnCreate_clicked()
{
  createSurfaceNode();
}

void SurfaceExtractorView::on_btnApply_clicked()
{
  Q_D(SurfaceExtractorView);
  d->gaussianSmooth = m_Controls->cbxGaussianSmooth->isChecked();
  d->gaussianStdDev = m_Controls->spbGaussianStdDev->value();
  d->threshold = m_Controls->spbThreshold->value();
  d->targetReduction = m_Controls->spbTargetReduction->value();
  d->maxNumberOfPolygons = m_Controls->spbMaxNumberOfPolygons->value();
  updateSurfaceNode();
}

void SurfaceExtractorView::createSurfaceNode()
{
  Q_D(SurfaceExtractorView);

  if (d->referenceNode.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::createSurfaceNode(): No reference image. The button should be disabled.";
    return;
  }
  if (d->surfaceNodes.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::createSurfaceNode(): No surface nodes. The button should be disabled.";
    return;
  }

  int maxSerial = 0;
  int surfaceNodeNumber = d->surfaceNodes->Size();
  for (int i = 0; i < surfaceNodeNumber; ++i)
  {
    mitk::DataNode::Pointer currentSurfaceNode = d->surfaceNodes->GetElement(i);
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

  d->surfaceNode = mitk::DataNode::New();
  d->surfaceNode->SetName(newSurfaceName.str());
  d->surfaceNode->SetColor(d->rainbowColor.GetNextColor() );
  d->surfaceNode->SetBoolProperty("Surface", true);
  d->surfaceNode->SetVisibility(true);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  MITK_INFO << "create surface node - surface is empty: " << surface->IsEmpty();
  d->surfaceNode->SetData(surface);

  RetrievePreferenceValues();
  updateFields();
  saveParameters();

  GetDataStorage()->Add(d->surfaceNode, d->referenceNode);
  d->surfaceNodes = findSurfaceNodesOf(d->referenceNode);

  d->referenceNode->SetSelected(false);
  d->surfaceNode->SetSelected(true);

  QList<mitk::DataNode::Pointer> selectedNodes;
  selectedNodes.push_back(d->surfaceNode);
  berry::IWorkbenchPart::Pointer nullPart;
  OnSelectionChanged(nullPart, selectedNodes);
}

void SurfaceExtractorView::updateSurfaceNode()
{
  Q_D(SurfaceExtractorView);

  MITK_INFO << "SurfaceExtractorView::updateSurfaceNode()";
  if (d->referenceNode.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::updateSurfaceNode(): No reference image. The button should be disabled.";
    return;
  }

  if (d->surfaceNode.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::updateSurfaceNode(): 12 no surface node is selected";
    return;
  }

  QApplication::setOverrideCursor( QCursor(Qt::WaitCursor) );

  mitk::Image* referenceImage = dynamic_cast<mitk::Image*>(d->referenceNode->GetData());

  if (!referenceImage)
  {
    MITK_INFO << "SurfaceExtractorView::createSurface(): No reference image. Should not arrive here.";
    return;
  }

  mitk::ImageToSurfaceFilter::Pointer filter = createImageToSurfaceFilter();
  if (filter.IsNull())
  {
    MITK_INFO << "SurfaceExtractorView::updateSurface(): No filter created.";
    return;
  }

  filter->SetInput(referenceImage);
  filter->SetThreshold(d->threshold); // if( Gauss ) --> TH manipulated for vtkMarchingCube
  filter->SetTargetReduction(d->targetReduction);

  try
  {
    MITK_INFO << "SurfaceExtractorView::updateSurfaceNode(): 40";
    filter->Update();
    MITK_INFO << "SurfaceExtractorView::updateSurfaceNode(): 57";
  }
  catch (std::exception& exc)
  {
    MITK_INFO << "SurfaceExtractorView::updateSurfaceNode(): 58";
  }
  MITK_INFO << "SurfaceExtractorView::updateSurfaceNode(): 60";

  long long numOfPolys = filter->GetOutput()->GetVtkPolyData()->GetNumberOfPolys();
  if (numOfPolys > d->maxNumberOfPolygons)
  {
    QApplication::restoreOverrideCursor();
    QString title = "CAUTION!!!";
    QString text = QString("The number of polygons is greater than %1. "
        "If you continue, the program might crash. "
        "How do you want to go on?").arg(d->maxNumberOfPolygons);
    QString button0Text = "Proceed anyway!";
    QString button1Text = "Cancel immediately! (maybe you want to insert an other threshold)!";
    if (QMessageBox::question(NULL, title, text, button0Text, button1Text, QString::null, 0 ,1) == 1)
    {
      return;
    }
    QApplication::setOverrideCursor( QCursor(Qt::WaitCursor) );
  }
  MITK_INFO << "SurfaceExtractorView::updateSurfaceNode(): 30";

  d->surfaceNode->SetData(filter->GetOutput());

  int layer = 0;

  d->referenceNode->GetIntProperty("layer", layer);
  d->surfaceNode->SetIntProperty("layer", layer + 1);
  d->surfaceNode->SetProperty("Surface", mitk::BoolProperty::New(true));
  saveParameters();

  RequestRenderWindowUpdate();

  d->dirty = false;
  m_Controls->btnApply->setEnabled(false);

  QApplication::restoreOverrideCursor();
}

mitk::ImageToSurfaceFilter::Pointer SurfaceExtractorView::createImageToSurfaceFilter()
{
  Q_D(SurfaceExtractorView);

  if (d->has3dOr4dGrayScaleImage->CheckNode(d->referenceNode))
  {
    mitk::ManualSegmentationToSurfaceFilter::Pointer filter = mitk::ManualSegmentationToSurfaceFilter::New();
    filter->SetUseGaussianImageSmooth(d->gaussianSmooth);
    filter->SetGaussianStandardDeviation(d->gaussianStdDev);
    mitk::ImageToSurfaceFilter::Pointer f = filter.GetPointer();
    return f;
  }
  else if (d->has3dOr4dLabelImage->CheckNode(d->referenceNode))
  {
    mitk::LabeledImageToSurfaceFilter::Pointer filter = mitk::LabeledImageToSurfaceFilter::New();
    filter->SetGaussianStandardDeviation(d->gaussianStdDev);
    mitk::ImageToSurfaceFilter::Pointer f = filter.GetPointer();
    return f;
  }
  else
  {
    MITK_INFO << "SurfaceExtractorView::createImageToSurfaceFilter() unknown image type";
    return 0;
  }
}

void SurfaceExtractorView::saveParameters()
{
  Q_D(SurfaceExtractorView);
  d->surfaceNode->SetBoolProperty(SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_NAME.c_str(), d->gaussianSmooth);
  d->surfaceNode->SetFloatProperty(SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_NAME.c_str(), d->gaussianStdDev);
  d->surfaceNode->SetFloatProperty(SurfaceExtractorPreferencePage::THRESHOLD_NAME.c_str(), d->threshold);
  d->surfaceNode->SetFloatProperty(SurfaceExtractorPreferencePage::TARGET_REDUCTION_NAME.c_str(), d->targetReduction);
  d->surfaceNode->SetIntProperty(SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_NAME.c_str(), d->maxNumberOfPolygons);
}

void SurfaceExtractorView::loadParameters()
{
  Q_D(SurfaceExtractorView);

  bool gaussianSmooth;
  float gaussianStdDev;
  float threshold;
  float targetReduction;
  int maxNumberOfPolygons;

  d->surfaceNode->GetBoolProperty(SurfaceExtractorPreferencePage::GAUSSIAN_SMOOTH_NAME.c_str(), gaussianSmooth);
  d->surfaceNode->GetFloatProperty(SurfaceExtractorPreferencePage::GAUSSIAN_STDDEV_NAME.c_str(), gaussianStdDev);
  d->surfaceNode->GetFloatProperty(SurfaceExtractorPreferencePage::THRESHOLD_NAME.c_str(), threshold);
  d->surfaceNode->GetFloatProperty(SurfaceExtractorPreferencePage::TARGET_REDUCTION_NAME.c_str(), targetReduction);
  d->surfaceNode->GetIntProperty(SurfaceExtractorPreferencePage::MAX_NUMBER_OF_POLYGONS_NAME.c_str(), maxNumberOfPolygons);

  d->gaussianSmooth = gaussianSmooth;
  d->gaussianStdDev = gaussianStdDev;
  d->threshold = threshold;
  d->targetReduction = targetReduction;
  d->maxNumberOfPolygons = maxNumberOfPolygons;
}

void SurfaceExtractorView::on_cbxGaussianSmooth_toggled(bool checked)
{
  m_Controls->spbGaussianStdDev->setEnabled(checked);
}
