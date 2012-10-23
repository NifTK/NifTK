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

#include "ImageLookupTablesView.h"
#include <QButtonGroup>
#include <QSlider>
#include <QDebug>
#include <itkImage.h>
#include <itkCommand.h>
#include <itkStatisticsImageFilter.h>
#include <itkEventObject.h>
#include <vtkLookupTable.h>
#include <mitkImage.h>
#include <mitkImageAccessByItk.h>
#include <mitkLookupTable.h>
#include <mitkLookupTableProperty.h>
#include <mitkNamedLookupTableProperty.h>
#include <mitkRenderingManager.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include "QmitkImageLookupTablesPreferencePage.h"
#include "LookupTableManager.h"
#include "LookupTableContainer.h"

#include "mitkLevelWindowManager.h"
#include "mitkNodePredicateData.h"
#include "mitkNodePredicateDataType.h"
#include "mitkNodePredicateProperty.h"
#include "mitkNodePredicateAnd.h"
#include "mitkNodePredicateNot.h"

const std::string ImageLookupTablesView::VIEW_ID = "uk.ac.ucl.cmic.imagelookuptables";
const std::string ImageLookupTablesView::DATA_MIN("data min");
const std::string ImageLookupTablesView::DATA_MAX("data max");
const std::string ImageLookupTablesView::DATA_MEAN("data mean");
const std::string ImageLookupTablesView::DATA_STDDEV("data std dev");

//-----------------------------------------------------------------------------
ImageLookupTablesView::ImageLookupTablesView()
: m_Controls(0)
, m_LookupTableManager(0)
, m_CurrentNode(NULL)
, m_CurrentImage(NULL)
, m_InitialisationMethod(QmitkImageLookupTablesPreferencePage::INITIALISATION_MIDAS)
, m_PercentageOfRange(100)
, m_Precision(2)
, m_InUpdate(false)
, m_ThresholdForIntegerBehaviour(50)
, m_PropertyObserverTag(0)
{
  m_LookupTableManager = new LookupTableManager();
}


//-----------------------------------------------------------------------------
ImageLookupTablesView::ImageLookupTablesView(const ImageLookupTablesView& other)
: berry::Object()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
ImageLookupTablesView::~ImageLookupTablesView()
{
  this->Unregister();

  if (m_Controls != NULL)
  {
    delete m_Controls;
  }

  if (m_LookupTableManager != NULL)
  {
    delete m_LookupTableManager;
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::CreateQtPartControl(QWidget *parent)
{
  MITK_DEBUG << "ImageLookupTablesView::CreateQtPartControl() begin" << std::endl;

  if (!m_Controls)
  {
    // Create UI.
    m_Controls = new Ui::ImageLookupTablesViewControls();
    m_Controls->setupUi(parent);

    // Set defaults on controls
    this->EnableControls(false);

    // Decide which group boxes are open/closed.
    m_Controls->m_RangeGroupBox->setCollapsed(false);
    m_Controls->m_LimitsGroupBox->setCollapsed(true);

    // Populate combo box with lookup table names.
    m_Controls->m_LookupTableComboBox->insertItem(0, "NONE");
    for (unsigned int i = 0; i < m_LookupTableManager->GetNumberOfLookupTables(); i++)
    {
      const LookupTableContainer *container = m_LookupTableManager->GetLookupTableContainer(i);
      m_Controls->m_LookupTableComboBox->insertItem(container->GetOrder()+1, container->GetDisplayName());
    }

    this->RetrievePreferenceValues();
    this->CreateConnections();
  }

  MITK_DEBUG << "ImageLookupTablesView::CreateQtPartControl() end" << std::endl;
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::CreateConnections()
{
  connect(m_Controls->m_MinSlider, SIGNAL(valueChanged(double)), SLOT(OnWindowBoundsChanged()));
  connect(m_Controls->m_MaxSlider, SIGNAL(valueChanged(double)), SLOT(OnWindowBoundsChanged()));
  connect(m_Controls->m_LevelSlider, SIGNAL(valueChanged(double)), SLOT(OnLevelWindowChanged()));
  connect(m_Controls->m_WindowSlider, SIGNAL(valueChanged(double)), SLOT(OnLevelWindowChanged()));
  connect(m_Controls->m_MinLimitDoubleSpinBox, SIGNAL(editingFinished()), SLOT(OnRangeChanged()));
  connect(m_Controls->m_MaxLimitDoubleSpinBox, SIGNAL(editingFinished()), SLOT(OnRangeChanged()));
  connect(m_Controls->m_LookupTableComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnLookupTableComboBoxChanged(int)));
  connect(m_Controls->m_ResetButton, SIGNAL(pressed()), this, SLOT(OnResetButtonPressed()));
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(VIEW_ID))
        .Cast<berry::IBerryPreferences>();
  assert( prefs );

  m_InitialisationMethod = prefs->Get(QmitkImageLookupTablesPreferencePage::INITIALISATION_METHOD_NAME, QmitkImageLookupTablesPreferencePage::INITIALISATION_MIDAS);
  m_PercentageOfRange = prefs->GetDouble(QmitkImageLookupTablesPreferencePage::PERCENTAGE_NAME, 50);
  m_Precision = prefs->GetInt(QmitkImageLookupTablesPreferencePage::PRECISION_NAME, 2);

  m_Controls->m_MinSlider->setDecimals(m_Precision);
  m_Controls->m_MaxSlider->setDecimals(m_Precision);
  m_Controls->m_LevelSlider->setDecimals(m_Precision);
  m_Controls->m_WindowSlider->setDecimals(m_Precision);
  m_Controls->m_MinLimitDoubleSpinBox->setDecimals(m_Precision);
  m_Controls->m_MaxLimitDoubleSpinBox->setDecimals(m_Precision);

  MITK_DEBUG << "ImageLookupTablesView::RetrievePreferenceValues" \
      " , m_InitialisationMethod=" << m_InitialisationMethod \
      << ", m_PercentageOfRange=" << m_PercentageOfRange \
      << ", m_Precision=" << m_Precision \
      << std::endl;
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::EnableControls(bool b)
{
  m_Controls->m_LookupTableComboBox->setEnabled(b);
  m_Controls->m_MinSlider->setEnabled(b);
  m_Controls->m_MaxSlider->setEnabled(b);
  m_Controls->m_WindowSlider->setEnabled(b);
  m_Controls->m_LevelSlider->setEnabled(b);
  m_Controls->m_MinLimitDoubleSpinBox->setEnabled(b);
  m_Controls->m_MaxLimitDoubleSpinBox->setEnabled(b);
  m_Controls->m_ResetButton->setEnabled(b);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::BlockSignals(bool b)
{
  m_Controls->m_MinSlider->blockSignals(b);
  m_Controls->m_MaxSlider->blockSignals(b);
  m_Controls->m_WindowSlider->blockSignals(b);
  m_Controls->m_LevelSlider->blockSignals(b);
  m_Controls->m_MinLimitDoubleSpinBox->blockSignals(b);
  m_Controls->m_MaxLimitDoubleSpinBox->blockSignals(b);
  m_Controls->m_ResetButton->blockSignals(b);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnSelectionChanged( berry::IWorkbenchPart::Pointer /*source*/,
                                             const QList<mitk::DataNode::Pointer>& nodes )
{

  bool isValid = this->IsSelectionValid(nodes);

  if (!isValid
      || (nodes[0].IsNotNull() && nodes[0] != m_CurrentNode)
     )
  {
    this->Unregister();
  }

  if (isValid)
  {
    this->Register(nodes[0]);
  }

  this->EnableControls(isValid);
}

//-----------------------------------------------------------------------------
bool ImageLookupTablesView::IsSelectionValid(const QList<mitk::DataNode::Pointer>& nodes)
{
  bool isValid = true;

  if (nodes.count() != 1)
  {
    isValid = false;
  }

  // All nodes must be non null, non-helper images.
  foreach( mitk::DataNode::Pointer node, nodes )
  {
    if(node.IsNull())
    {
      isValid = false;
    }

    if (node.IsNotNull() && dynamic_cast<mitk::Image*>(node->GetData()) == NULL)
    {
      isValid = false;
    }

    bool isHelper(false);
    if (node->GetBoolProperty("helper object", isHelper) && isHelper)
    {
      isValid = false;
    }

    bool isSelected(false);
    node->GetBoolProperty("selected", isSelected);
    if (!isSelected)
    {
      isValid = false;
    }

    if (!node->GetProperty("levelwindow"))
    {
      isValid = false;
    }

  }

  return isValid;
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::Register(const mitk::DataNode::Pointer node)
{
  if (node.IsNotNull())
  {
    m_CurrentNode = node;

    this->DifferentImageSelected();
    this->OnRangeChanged();
    this->OnPropertyChanged();

    itk::ReceptorMemberCommand<ImageLookupTablesView>::Pointer command
      = itk::ReceptorMemberCommand<ImageLookupTablesView>::New();
    command->SetCallbackFunction(this, &ImageLookupTablesView::OnPropertyChanged);
    mitk::BaseProperty::Pointer property = node->GetProperty("levelwindow");

    m_PropertyObserverTag = property->AddObserver(itk::ModifiedEvent(), command);
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::Unregister()
{
  if (m_CurrentNode.IsNotNull())
  {
    mitk::BaseProperty::Pointer property = m_CurrentNode->GetProperty("levelwindow");
    property->RemoveObserver(m_PropertyObserverTag);

    m_CurrentNode = NULL;
    m_PropertyObserverTag = -1;
  }
}

//-----------------------------------------------------------------------------
void ImageLookupTablesView::DifferentImageSelected()
{
  this->BlockSignals(true);

  m_CurrentImage = dynamic_cast<mitk::Image*>(m_CurrentNode->GetData());

  int lookupTableIndex(0);
  float minDataLimit(0);
  float maxDataLimit(0);
  float meanData(0);
  float stdDevData(0);

  bool minDataLimitFound = m_CurrentNode->GetFloatProperty(DATA_MIN.c_str(), minDataLimit);
  bool maxDataLimitFound = m_CurrentNode->GetFloatProperty(DATA_MAX.c_str(), maxDataLimit);
  bool meanDataFound = m_CurrentNode->GetFloatProperty(DATA_MEAN.c_str(), meanData);
  bool stdDevDataFound = m_CurrentNode->GetFloatProperty(DATA_STDDEV.c_str(), stdDevData);
  bool lookupTableIndexFound = m_CurrentNode->GetIntProperty("LookupTableIndex", lookupTableIndex);

  if (!minDataLimitFound || !maxDataLimitFound || !meanDataFound || !stdDevDataFound)
  {
    try
    {
      if (m_CurrentImage->GetDimension() == 2)
      {
        AccessFixedDimensionByItk_n(m_CurrentImage,
            ITKGetStatistics, 2,
            (minDataLimit, maxDataLimit, meanData, stdDevData)
          );
      }
      else if (m_CurrentImage->GetDimension() == 3)
      {
        AccessFixedDimensionByItk_n(m_CurrentImage,
            ITKGetStatistics, 3,
            (minDataLimit, maxDataLimit, meanData, stdDevData)
          );
      }
      else if (m_CurrentImage->GetDimension() == 4)
      {
        AccessFixedDimensionByItk_n(m_CurrentImage,
            ITKGetStatistics, 4,
            (minDataLimit, maxDataLimit, meanData, stdDevData)
          );
      }
      m_CurrentNode->SetFloatProperty(DATA_MIN.c_str(), minDataLimit);
      m_CurrentNode->SetFloatProperty(DATA_MAX.c_str(), maxDataLimit);
      m_CurrentNode->SetFloatProperty(DATA_MEAN.c_str(), meanData);
      m_CurrentNode->SetFloatProperty(DATA_STDDEV.c_str(), stdDevData);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception during ImageLookupTablesView::ITKGetStatistics, so image statistics will be wrong." << e.what();
    }
  }

  double windowMin = 0;
  double windowMax = 0;
  mitk::LevelWindow levelWindow;

  if (!minDataLimitFound || !maxDataLimitFound || !meanDataFound || !stdDevDataFound)
  {
    // This image hasn't had the data members that this view needs (minDataLimit, maxDataLimit etc) initialized yet.
    // i.e. we haven't seen it before. So we have a choice of how to initialise the Level/Window.
    if (m_InitialisationMethod == QmitkImageLookupTablesPreferencePage::INITIALISATION_MIDAS)
    {
      double centre = (minDataLimit + 4.51*stdDevData)/2.0;
      double width = 4.5*stdDevData;
      windowMin = centre - width/2.0;
      windowMax = centre + width/2.0;

      MITK_DEBUG << "ImageLookupTablesView::DifferentImageSelected, initialise from MIDAS method" \
          << ", mean=" << meanData \
          << ", stdDev=" << stdDevData \
          << ", minDataLimit=" << minDataLimit \
          << ", maxDataLimit=" << maxDataLimit \
          << ", windowMin=" << windowMin \
          << ", windowMax=" << windowMax \
          << std::endl;
    }
    else if (m_InitialisationMethod == QmitkImageLookupTablesPreferencePage::INITIALISATION_PERCENTAGE)
    {
      windowMin = minDataLimit;
      windowMax = minDataLimit + (maxDataLimit - minDataLimit)*m_PercentageOfRange/100.0;

      MITK_DEBUG << "ImageLookupTablesView::DifferentImageSelected, initialise from range" \
          << ", m_PercentageOfRange=" << m_PercentageOfRange \
          << ", minDataLimit=" << minDataLimit \
          << ", maxDataLimit=" << maxDataLimit \
          << ", windowMin=" << windowMin \
          << ", windowMax=" << windowMax \
          << std::endl;
    }
    else
    {
      m_CurrentNode->GetLevelWindow(levelWindow);
      minDataLimit = levelWindow.GetRangeMin();
      maxDataLimit = levelWindow.GetRangeMax();
      windowMin = levelWindow.GetLowerWindowBound();
      windowMax = levelWindow.GetUpperWindowBound();

      MITK_DEBUG << "ImageLookupTablesView::DifferentImageSelected, initialise from current levelWindow: " \
          << ", minDataLimit=" << minDataLimit \
          << ", maxDataLimit=" << maxDataLimit \
          << ", windowMin=" << windowMin \
          << ", windowMax=" << windowMax << std::endl;
    }
  }
  else
  {
    m_CurrentNode->GetLevelWindow(levelWindow);
    minDataLimit = levelWindow.GetRangeMin();
    maxDataLimit = levelWindow.GetRangeMax();
    windowMin = levelWindow.GetLowerWindowBound();
    windowMax = levelWindow.GetUpperWindowBound();
  }

  MITK_DEBUG << "ImageLookupTablesView::DifferentImageSelected" \
      << ", dataMin=" << minDataLimit \
      << ", dataMax=" << maxDataLimit \
      << ", windowMin=" << windowMin \
      << ", windowMax=" << windowMax << std::endl;

  // Round up the current min and max.
  if (fabs(maxDataLimit - minDataLimit) > m_ThresholdForIntegerBehaviour)
  {
    windowMin = (int)(windowMin +0.5);
    windowMax = (int)(windowMax +0.5);
  }

  m_Controls->m_MinLimitDoubleSpinBox->setValue(minDataLimit);
  m_Controls->m_MaxLimitDoubleSpinBox->setValue(maxDataLimit);

  if (lookupTableIndexFound)
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(lookupTableIndex);
  }
  else
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(0);
  }

  levelWindow.SetRangeMinMax(minDataLimit, maxDataLimit);
  levelWindow.SetWindowBounds(windowMin, windowMax);

  m_CurrentNode->SetLevelWindow(levelWindow);

  this->BlockSignals(false);
}



//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ImageLookupTablesView
::ITKGetStatistics(
    itk::Image<TPixel, VImageDimension> *itkImage,
    float &min,
    float &max,
    float &mean,
    float &stdDev)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef itk::StatisticsImageFilter<ImageType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput(itkImage);
  filter->UpdateLargestPossibleRegion();
  min = filter->GetMinimum();
  max = filter->GetMaximum();
  mean = filter->GetMean();
  stdDev = filter->GetSigma();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnRangeChanged()
{
  this->BlockSignals(true);

  mitk::LevelWindow levelWindow;
  m_CurrentNode->GetLevelWindow(levelWindow);

  levelWindow.SetRangeMinMax(m_Controls->m_MinLimitDoubleSpinBox->value(), m_Controls->m_MaxLimitDoubleSpinBox->value());

  double rangeMin = levelWindow.GetRangeMin();
  double rangeMax = levelWindow.GetRangeMax();
  double range = levelWindow.GetRange();

  // Trac 1680 - don't forget, MIDAS generally deals with integer images
  // so the user requirements are such that they must be able to change
  // intensity ranges in steps of 1. If however, we are using float images
  // we will need to be able to change intensity values in much smaller stepps.
  double singleStep;
  double pageStep;

  if (fabs(rangeMin - rangeMax) > m_ThresholdForIntegerBehaviour)
  {
    // i.e. we have a large enough range to use integer page step and single step.
    singleStep = 1;
    pageStep = 10;
  }
  else
  {
    // i.e. in this case, use fractions.
    singleStep = range / 100.0;
    pageStep = range / 10.0;
  }

  m_Controls->m_MinSlider->setMinimum(rangeMin);
  m_Controls->m_MinSlider->setMaximum(rangeMax);
  m_Controls->m_MaxSlider->setMinimum(rangeMin);
  m_Controls->m_MaxSlider->setMaximum(rangeMax);
  m_Controls->m_MinSlider->setSingleStep(singleStep);
  m_Controls->m_MinSlider->setTickInterval(singleStep);
  m_Controls->m_MinSlider->setPageStep(pageStep);
  m_Controls->m_MaxSlider->setSingleStep(singleStep);
  m_Controls->m_MaxSlider->setTickInterval(singleStep);
  m_Controls->m_MaxSlider->setPageStep(pageStep);
  m_Controls->m_WindowSlider->setMinimum(0);
  m_Controls->m_WindowSlider->setMaximum(range);
  m_Controls->m_WindowSlider->setSingleStep(singleStep);
  m_Controls->m_LevelSlider->setMinimum(rangeMin);
  m_Controls->m_LevelSlider->setMaximum(rangeMax);
  m_Controls->m_LevelSlider->setSingleStep(singleStep);

  this->BlockSignals(false);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnPropertyChanged(const itk::EventObject&)
{
  this->OnPropertyChanged();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnPropertyChanged()
{
  this->UpdateGuiFromLevelWindow();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::UpdateGuiFromLevelWindow()
{
  this->BlockSignals(true);

  mitk::LevelWindow levelWindow;
  m_CurrentNode->GetLevelWindow(levelWindow);

  double min = levelWindow.GetLowerWindowBound();
  double max = levelWindow.GetUpperWindowBound();
  double level = levelWindow.GetLevel();
  double window = levelWindow.GetWindow();

  m_Controls->m_MinSlider->setValue(min);
  m_Controls->m_MaxSlider->setValue(max);
  m_Controls->m_LevelSlider->setValue(level);
  m_Controls->m_WindowSlider->setValue(window);

  this->BlockSignals(false);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnWindowBoundsChanged()
{
  // Get the current values
  mitk::LevelWindow levelWindow;
  m_CurrentNode->GetLevelWindow(levelWindow);

  // Note: This method is called when one of the sliders has been moved
  // So, it's purpose is to update the other sliders to match.

  // Update them from controls
  double min = m_Controls->m_MinSlider->value();
  double max = m_Controls->m_MaxSlider->value();

  levelWindow.SetWindowBounds(min, max);
  m_CurrentNode->SetLevelWindow(levelWindow);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnLevelWindowChanged()
{
  // Get the current values
  mitk::LevelWindow levelWindow;
  m_CurrentNode->GetLevelWindow(levelWindow);

  // Note: This method is called when one of the sliders has been moved
  // So, it's purpose is to update the other sliders to match.

  // Update them from controls
  double window = m_Controls->m_WindowSlider->value();
  double level = m_Controls->m_LevelSlider->value();

  levelWindow.SetLevelWindow(level, window);
  m_CurrentNode->SetLevelWindow(levelWindow);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnLookupTableComboBoxChanged(int comboBoxIndex)
{
  if (comboBoxIndex > 0)
  {
    // Copy the vtkLookupTable
    const LookupTableContainer* lutContainer = m_LookupTableManager->GetLookupTableContainer(comboBoxIndex - 1);
    const vtkLookupTable *vtkLUT = lutContainer->GetLookupTable();
    mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
    mitkLUT->SetVtkLookupTable(const_cast<vtkLookupTable*>(vtkLUT));
    const std::string& lutName = lutContainer->GetDisplayName().toStdString();
    mitk::NamedLookupTableProperty::Pointer mitkLUTProperty = mitk::NamedLookupTableProperty::New(lutName, mitkLUT);

    // and give to the node property.
    m_CurrentNode->ReplaceProperty("LookupTable", mitkLUTProperty);
    m_CurrentNode->SetBoolProperty("use color", false);
    m_CurrentNode->SetIntProperty("LookupTableIndex", comboBoxIndex);
  }
  else
  {
    m_CurrentNode->SetProperty("LookupTable", 0);
    m_CurrentNode->SetBoolProperty("use color", true);
    m_CurrentNode->GetPropertyList()->DeleteProperty("LookupTableIndex");
  }

  m_CurrentNode->Update();
  this->RequestRenderWindowUpdate();
}



//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnResetButtonPressed()
{

  mitk::LevelWindow levelWindow;
  m_CurrentNode->GetLevelWindow(levelWindow);

  float rangeMin(0);
  float rangeMax(0);

  if (m_CurrentNode->GetFloatProperty(DATA_MIN.c_str(), rangeMin)
      && m_CurrentNode->GetFloatProperty(DATA_MAX.c_str(), rangeMax))
  {
    levelWindow.SetRangeMinMax(rangeMin, rangeMax);
    levelWindow.SetWindowBounds(rangeMin, rangeMax);

    std::cerr << "Matt, setting range to " << rangeMin << ", " << rangeMax << std::endl;

    m_CurrentNode->SetLevelWindow(levelWindow);

    this->RequestRenderWindowUpdate();
  }
}

