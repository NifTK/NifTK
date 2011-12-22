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
#include "LookupTableManager.h"
#include "LookupTableContainer.h"
#include "vtkLookupTable.h"
#include "mitkImage.h"
#include "mitkLevelWindowManager.h"
#include "mitkRenderingManager.h"
#include "mitkLookupTable.h"
#include "mitkLookupTableProperty.h"
#include "mitkNodePredicateData.h"
#include "mitkNodePredicateDataType.h"
#include "mitkNodePredicateProperty.h"
#include "mitkNodePredicateAnd.h"
#include "mitkNodePredicateNot.h"
#include "QmitkImageLookupTablesPreferencePage.h"
#include "berryIPreferencesService.h"
#include "berryIBerryPreferences.h"
#include "NamedLookupTableProperty.h"

#include <itkEventObject.h>

const std::string ImageLookupTablesView::VIEW_ID = "uk.ac.ucl.cmic.imagelookuptables";

ImageLookupTablesView::ImageLookupTablesView()
: m_Controls(0)
, m_LookupTableManager(0)
, m_LevelWindowManager(0)
, m_UseImageRangeToInitialise(true)
, m_PercentageOfRange(100)
, m_Precision(2)
, m_CurrentNode(NULL)
, m_CurrentImage(NULL)
, m_InUpdate(false)
{
  m_LookupTableManager = new LookupTableManager();
  m_LevelWindowManager = mitk::LevelWindowManager::New();
}

ImageLookupTablesView::ImageLookupTablesView(const ImageLookupTablesView& other)
: berry::Object()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

ImageLookupTablesView::~ImageLookupTablesView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }

  if (m_LookupTableManager != NULL)
  {
    delete m_LookupTableManager;
  }
}

void ImageLookupTablesView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(VIEW_ID))
        .Cast<berry::IBerryPreferences>();
  assert( prefs );

  m_UseImageRangeToInitialise = prefs->GetBool(QmitkImageLookupTablesPreferencePage::USE_IMAGE_DATA_NAME, true);
  m_PercentageOfRange = prefs->GetDouble(QmitkImageLookupTablesPreferencePage::PERCENTAGE_NAME, 60);
  m_Precision = prefs->GetInt(QmitkImageLookupTablesPreferencePage::PRECISION_NAME, 2);

  m_Controls->m_MinSlider->setDecimals(m_Precision);
  m_Controls->m_MaxSlider->setDecimals(m_Precision);
  m_Controls->m_LevelSlider->setDecimals(m_Precision);
  m_Controls->m_WindowSlider->setDecimals(m_Precision);
  m_Controls->m_MinLimitDoubleSpinBox->setDecimals(m_Precision);
  m_Controls->m_MaxLimitDoubleSpinBox->setDecimals(m_Precision);

  MITK_DEBUG << "ImageLookupTablesView::RetrievePreferenceValues" \
      " , m_UseImageRangeToInitialise=" << m_UseImageRangeToInitialise \
      << ", m_PercentageOfRange=" << m_PercentageOfRange \
      << ", m_Precision=" << m_Precision \
      << std::endl;
}

void ImageLookupTablesView::CreateQtPartControl(QWidget *parent)
{
  MITK_DEBUG << "ImageLookupTablesView::CreateQtPartControl() begin" << std::endl;

  // setup the basic GUI of this view
  m_Parent = parent;

  if (!m_Controls)
  {
    // Create UI.
    m_Controls = new Ui::ImageLookupTablesViewControls();
    m_Controls->setupUi(parent);

    // Set defaults
    this->EnableControls(false);
    m_Controls->m_RangeGroupBox->setCollapsed(false);
    m_Controls->m_LimitsGroupBox->setCollapsed(true);

    // Populate combo box with lookup table names.
    m_Controls->m_LookupTableComboBox->insertItem(0, "NONE");
    for (unsigned int i = 0; i < m_LookupTableManager->GetNumberOfLookupTables(); i++)
    {
      const LookupTableContainer *container = m_LookupTableManager->GetLookupTableContainer(i);
      m_Controls->m_LookupTableComboBox->insertItem(container->GetOrder()+1, container->GetDisplayName());
    }

    // Retrieve and store preference values.
    RetrievePreferenceValues();

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
    CreateConnections();
  }
  MITK_DEBUG << "ImageLookupTablesView::CreateQtPartControl() end" << std::endl;
}

void ImageLookupTablesView::CreateConnections()
{
  connect(m_Controls->m_MinSlider, SIGNAL(valueChanged(double)), SLOT(OnWindowBoundsChanged()));
  connect(m_Controls->m_MaxSlider, SIGNAL(valueChanged(double)), SLOT(OnWindowBoundsChanged()));
  connect(m_Controls->m_LevelSlider, SIGNAL(valueChanged(double)), SLOT(OnLevelWindowChanged()));
  connect(m_Controls->m_WindowSlider, SIGNAL(valueChanged(double)), SLOT(OnLevelWindowChanged()));
  connect(m_Controls->m_MinLimitDoubleSpinBox, SIGNAL(valueChanged(double)), SLOT(OnRangeChanged()));
  connect(m_Controls->m_MaxLimitDoubleSpinBox, SIGNAL(valueChanged(double)), SLOT(OnRangeChanged()));
  connect(m_Controls->m_LookupTableComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnLookupTableComboBoxChanged(int)));
  connect(m_Controls->m_ResetButton, SIGNAL(pressed()), this, SLOT(OnResetButtonPressed()));
}

void ImageLookupTablesView::LevelWindowChanged()
{
  try
  {
    m_CurrentLevelWindow = m_LevelWindowManager->GetLevelWindow();
    UpdateGuiFromLevelWindowManager();
    EnableControls(true);
  }
  catch(itk::ExceptionObject&)
  {
    try
    {
      EnableControls(false);
    }
    catch(std::exception&)
    {
    }
  }
}

void ImageLookupTablesView::DifferentImageSelected(const mitk::DataNode* node, mitk::Image* image)
{
  m_CurrentNode = const_cast<mitk::DataNode*>(node);
  m_CurrentImage = image;

  int lookupTableIndex(0);
  float minDataLimit(0);
  float maxDataLimit(0);
  bool minDataLimitFound = m_CurrentNode->GetFloatProperty("minDataLimit", minDataLimit);
  bool maxDataLimitFound = m_CurrentNode->GetFloatProperty("maxDataLimit", maxDataLimit);
  bool lookupTableIndexFound = m_CurrentNode->GetIntProperty("LookupTableIndex", lookupTableIndex);
  double dataMin = image->GetScalarValueMin();
  double dataMax = image->GetScalarValueMax();
  double windowMin = 0;
  double windowMax = 0;
  mitk::LevelWindow levelWindow;

  if (!minDataLimitFound || !maxDataLimitFound)
  {
    // This image hasn't had the data members that this view needs (minDataLimit, maxDataLimit etc) initialized yet.
    // i.e. we haven't seen it before.
    if (m_UseImageRangeToInitialise)
    {
      minDataLimit = dataMin;
      maxDataLimit = dataMax;
      windowMin = dataMin;
      windowMax = dataMin + (dataMax - dataMin)*m_PercentageOfRange/100.0;

      MITK_DEBUG << "ImageLookupTablesView::DifferentImageSelected, initialise from range, m_PercentageOfRange=" << m_PercentageOfRange << ", minDataLimit=" << minDataLimit << ", maxDataLimit=" << maxDataLimit << ", windowMin=" << windowMin << ", windowMax=" << windowMax << std::endl;
    }
    else if (m_LevelWindowManager->GetLevelWindowProperty())
    {
      levelWindow = m_LevelWindowManager->GetLevelWindow();
      minDataLimit = levelWindow.GetRangeMin();
      maxDataLimit = levelWindow.GetRangeMax();
      windowMin = levelWindow.GetLowerWindowBound();
      windowMax = levelWindow.GetUpperWindowBound();

      MITK_DEBUG << "ImageLookupTablesView::DifferentImageSelected, initialise from current levelWindow, minDataLimit=" << minDataLimit << ", maxDataLimit=" << maxDataLimit << ", windowMin=" << windowMin << ", windowMax=" << windowMax << std::endl;
    }
    m_CurrentNode->SetFloatProperty("minDataLimit", minDataLimit);
    m_CurrentNode->SetFloatProperty("maxDataLimit", maxDataLimit);
  }
  else
  {
    m_CurrentNode->GetLevelWindow(levelWindow);
    minDataLimit = levelWindow.GetRangeMin();
    maxDataLimit = levelWindow.GetRangeMax();
    windowMin = levelWindow.GetLowerWindowBound();
    windowMax = levelWindow.GetUpperWindowBound();
  }

  MITK_DEBUG << "ImageLookupTablesView::DifferentImageSelected, dataMin=" << minDataLimit << ", dataMax=" << maxDataLimit << ", windowMin=" << windowMin << ", windowMax=" << windowMax << std::endl;

  // Propagate to min/max controls and back to m_LevelWindowManager
  m_Controls->m_MinLimitDoubleSpinBox->setValue(minDataLimit);
  m_Controls->m_MaxLimitDoubleSpinBox->setValue(maxDataLimit);
  levelWindow.SetRangeMinMax(minDataLimit, maxDataLimit);
  levelWindow.SetWindowBounds(windowMin, windowMax);
  m_LevelWindowManager->SetLevelWindow(levelWindow);
  m_CurrentLevelWindow = levelWindow;

  if (lookupTableIndexFound)
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(lookupTableIndex);
  }
  else
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(0);
  }
  OnRangeChanged();
  UpdateGuiFromLevelWindowManager();
  EnableControls(true);
}

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

void ImageLookupTablesView::BlockMinMaxSignals(bool b)
{
  m_Controls->m_MinSlider->blockSignals(b);
  m_Controls->m_MaxSlider->blockSignals(b);
  m_Controls->m_WindowSlider->blockSignals(b);
  m_Controls->m_LevelSlider->blockSignals(b);
  m_Controls->m_MinLimitDoubleSpinBox->blockSignals(b);
  m_Controls->m_MaxLimitDoubleSpinBox->blockSignals(b);
  m_Controls->m_ResetButton->blockSignals(b);
}

void ImageLookupTablesView::OnWindowBoundsChanged()
{
  if (!m_LevelWindowManager->GetLevelWindowProperty()) {
    return;
  }

  mitk::LevelWindow levelWindow = m_LevelWindowManager->GetLevelWindow();
  levelWindow.SetWindowBounds(m_Controls->m_MinSlider->value(), m_Controls->m_MaxSlider->value());
  m_LevelWindowManager->SetLevelWindow(levelWindow);

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void ImageLookupTablesView::OnLevelWindowChanged()
{
  if (!m_LevelWindowManager->GetLevelWindowProperty()) {
    return;
  }

  mitk::LevelWindow levelWindow = m_LevelWindowManager->GetLevelWindow();
  levelWindow.SetLevelWindow(m_Controls->m_LevelSlider->value(), m_Controls->m_WindowSlider->value());
  m_LevelWindowManager->SetLevelWindow(levelWindow);

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void ImageLookupTablesView::OnRangeChanged()
{
  if (!m_LevelWindowManager->GetLevelWindowProperty()) {
    return;
  }

  BlockMinMaxSignals(true);

  mitk::LevelWindow levelWindow = m_LevelWindowManager->GetLevelWindow();
  levelWindow.SetRangeMinMax(m_Controls->m_MinLimitDoubleSpinBox->value(), m_Controls->m_MaxLimitDoubleSpinBox->value());

  double rangeMin = levelWindow.GetRangeMin();
  double rangeMax = levelWindow.GetRangeMax();
  double range = levelWindow.GetRange();
  double singleStep = range / 100.0;

  m_Controls->m_MinSlider->setMinimum(rangeMin);
  m_Controls->m_MinSlider->setMaximum(rangeMax);
  m_Controls->m_MaxSlider->setMinimum(rangeMin);
  m_Controls->m_MaxSlider->setMaximum(rangeMax);
  m_Controls->m_MinSlider->setSingleStep(singleStep);
  m_Controls->m_MaxSlider->setSingleStep(singleStep);
  m_Controls->m_WindowSlider->setMinimum(0);
  m_Controls->m_WindowSlider->setMaximum(range);
  m_Controls->m_WindowSlider->setSingleStep(singleStep);
  m_Controls->m_LevelSlider->setMinimum(rangeMin);
  m_Controls->m_LevelSlider->setMaximum(rangeMax);
  m_Controls->m_LevelSlider->setSingleStep(singleStep);

  m_CurrentNode->SetFloatProperty("minDataLimit", levelWindow.GetRangeMin());
  m_CurrentNode->SetFloatProperty("maxDataLimit", levelWindow.GetRangeMax());

  BlockMinMaxSignals(false);

  m_LevelWindowManager->SetLevelWindow(levelWindow);
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void ImageLookupTablesView::UpdateGuiFromLevelWindowManager()
{
  if (!m_LevelWindowManager->GetLevelWindowProperty()) {
    return;
  }

  BlockMinMaxSignals(true);

  const mitk::LevelWindow& levelWindow = m_LevelWindowManager->GetLevelWindow();

  double min = levelWindow.GetLowerWindowBound();
  double max = levelWindow.GetUpperWindowBound();
  double level = levelWindow.GetLevel();
  double window = levelWindow.GetWindow();

  m_Controls->m_MinSlider->setValue(min);
  m_Controls->m_MaxSlider->setValue(max);
  m_Controls->m_LevelSlider->setValue(level);
  m_Controls->m_WindowSlider->setValue(window);

  BlockMinMaxSignals(false);
}

void ImageLookupTablesView::Activated()
{
  MITK_DEBUG << "ImageLookupTablesView::Activated() begin" << std::endl;

  m_LevelWindowManager->SetDataStorage(GetDefaultDataStorage());
  this->GetDefaultDataStorage()->ChangedNodeEvent.AddListener( mitk::MessageDelegate1<ImageLookupTablesView, const mitk::DataNode*>
      ( this, &ImageLookupTablesView::OnNodeChanged ) );

  mitk::Image::Pointer currentImage = m_LevelWindowManager->GetCurrentImage();
  if (currentImage.IsNotNull())
  {
    mitk::DataNode::Pointer node = this->FindNodeForImage(currentImage);
    if (node.IsNotNull())
    {
      DifferentImageSelected(node, currentImage);
    }
  }

  MITK_DEBUG << "ImageLookupTablesView::Activated() end" << std::endl;
}

void ImageLookupTablesView::Deactivated()
{
  this->GetDefaultDataStorage()->ChangedNodeEvent.RemoveListener( mitk::MessageDelegate1<ImageLookupTablesView, const mitk::DataNode*>
      ( this, &ImageLookupTablesView::OnNodeChanged ) );
}

mitk::DataNode* ImageLookupTablesView::FindNodeForImage(mitk::Image* image)
{
  mitk::DataNode::Pointer result = NULL;

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  mitk::DataStorage::SetOfObjects::ConstPointer possibleCandidates = this->GetDefaultDataStorage()->GetSubset(isImage);
  for (unsigned int i = 0; i < possibleCandidates->size(); i++)
  {
    mitk::DataNode* possibleNode = (*possibleCandidates)[i];
    mitk::Image* possibleImage = dynamic_cast<mitk::Image*>(possibleNode->GetData());
    if (possibleImage == image)
    {
      result = possibleNode;
    }
  }
  return result;
}

void ImageLookupTablesView::OnNodeChanged(const mitk::DataNode* nodeFromDataStore)
{
  // Could be an assert statement.
  if (nodeFromDataStore == NULL) return;

  // Just double check GUI has been created properly, and we are visible.
  if (!m_Parent || !m_Parent->isVisible()) return;

  std::string name;
  nodeFromDataStore->GetStringProperty("name", name);

  MITK_DEBUG << "ImageLookupTablesView::OnNodeChanged, name=" << name << ", node=" << nodeFromDataStore << std::endl;

  mitk::Image::Pointer imageFromDataStore = dynamic_cast<mitk::Image*>(nodeFromDataStore->GetData());
  if (imageFromDataStore.IsNotNull() && !m_InUpdate)
  {
    m_InUpdate = true;

    mitk::Image::Pointer currentImageFromLevelWindowManager = m_LevelWindowManager->GetCurrentImage();
    if (currentImageFromLevelWindowManager.IsNotNull())
    {
      if (m_CurrentImage != m_LevelWindowManager->GetCurrentImage())
      {
        mitk::DataNode::Pointer node = this->FindNodeForImage(m_LevelWindowManager->GetCurrentImage());
        node->GetStringProperty("name", name);

        MITK_DEBUG << "ImageLookupTablesView::OnNodeChanged, new image=" << m_LevelWindowManager->GetCurrentImage() << ", name=" << name << std::endl;
        DifferentImageSelected(node, m_LevelWindowManager->GetCurrentImage());
      }
      else if (m_CurrentImage == m_LevelWindowManager->GetCurrentImage() )
      {
        mitk::LevelWindow levelWindowFromDataStoreNode;
        if (nodeFromDataStore->GetLevelWindow(levelWindowFromDataStoreNode))
        {
          if (levelWindowFromDataStoreNode != m_CurrentLevelWindow)
          {
            MITK_DEBUG << "ImageLookupTablesView::Updating levels" << std::endl;
            LevelWindowChanged();
          }
        }
      }
    }

    m_InUpdate = false;
  }
  else
  {
    MITK_DEBUG << "ImageLookupTablesView::node=" << nodeFromDataStore << ", was not an image" << std::endl;
  }
}

void ImageLookupTablesView::OnLookupTableComboBoxChanged(int comboBoxIndex)
{
  // Just double check GUI has been created properly, and we are visible.
  if (!m_Parent || !m_Parent->isVisible()) return;

  if (comboBoxIndex > 0)
  {
    // Copy the vtkLookupTable, and give to the node property.
    const LookupTableContainer* lutContainer = m_LookupTableManager->GetLookupTableContainer(comboBoxIndex - 1);

    const vtkLookupTable *vtkLUT = lutContainer->GetLookupTable();
    mitk::LookupTable::Pointer mitkLUT = mitk::LookupTable::New();
    mitkLUT->SetVtkLookupTable(const_cast<vtkLookupTable*>(vtkLUT));
    const std::string& lutName = lutContainer->GetDisplayName().toStdString();
    NamedLookupTableProperty::Pointer mitkLUTProperty = NamedLookupTableProperty::New(lutName, mitkLUT);
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

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void ImageLookupTablesView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  RetrievePreferenceValues();
}

void ImageLookupTablesView::OnResetButtonPressed()
{
  if (!m_LevelWindowManager->GetLevelWindowProperty()) {
    return;
  }

  mitk::Image::Pointer image = m_LevelWindowManager->GetCurrentImage();
  if (image.IsNotNull())
  {
    mitk::LevelWindow levelWindow = m_LevelWindowManager->GetLevelWindow();
    double rangeMin = image->GetScalarValueMin();
    double rangeMax = image->GetScalarValueMax();
    levelWindow.SetRangeMinMax(rangeMin, rangeMax);
    levelWindow.SetWindowBounds(rangeMin, rangeMax);
    m_LevelWindowManager->SetLevelWindow(levelWindow);

    mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  }
}
