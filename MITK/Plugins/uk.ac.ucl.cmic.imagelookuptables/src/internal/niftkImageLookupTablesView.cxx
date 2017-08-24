/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkImageLookupTablesView.h"

#include <QButtonGroup>
#include <QColorDialog>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QSignalMapper>

#include <itkCommand.h>
#include <itkEventObject.h>
#include <itkImage.h>
#include <itkStatisticsImageFilter.h>

#include <vtkLookupTable.h>

#include <mitkImage.h>
#include <mitkImageAccessByItk.h>
#include <mitkIOUtil.h>
#include <mitkLevelWindowManager.h>
#include <mitkLookupTable.h>
#include <mitkLookupTableProperty.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateData.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkRenderingManager.h>
#include <mitkRenderingModeProperty.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include <usModule.h>
#include <usModuleContext.h>
#include <usModuleInitialization.h>
#include <usModuleRegistry.h>

#include <niftkDataStorageUtils.h>
#include <niftkLabeledLookupTableProperty.h>
#include <niftkLookupTableContainer.h>
#include <niftkNamedLookupTableProperty.h>

#include <niftkLookupTableProviderService.h>
#include <niftkVtkLookupTableUtils.h>

#include "niftkImageLookupTablesPreferencePage.h"
#include "niftkPluginActivator.h"


namespace niftk
{

//-----------------------------------------------------------------------------
ImageLookupTablesView::ImageLookupTablesView()
: m_Controls(0)
, m_SelectedNodes()
, m_Precision(2)
, m_InUpdate(false)
, m_ThresholdForIntegerBehaviour(50)
, m_LevelWindowPropertyObserverTag(0)
{
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
  if (!m_SelectedNodes.isEmpty())
  {
    this->UnregisterObservers();
  }

  if (m_Controls != nullptr)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::CreateQtPartControl(QWidget *parent)
{
  if (!m_Controls)
  {
    // Create UI.
    m_Controls = new Ui::niftkImageLookupTablesViewControls();
    m_Controls->setupUi(parent);

    // Decide which group boxes are open/closed.
    m_Controls->m_RangeGroupBox->setCollapsed(false);
    m_Controls->m_LimitsGroupBox->setCollapsed(true);

    this->EnableScaleControls(false);
    this->EnableLabelControls(false);

    /// This is probably superfluous because the AbstractView::AfterCreateQtPartControl() calls
    /// OnPreferencesChanged that calls RetrievePreferenceValues. It would need testing.
    this->RetrievePreferenceValues();

    this->UpdateLookupTableComboBoxEntries();
    this->CreateConnections();
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::CreateConnections()
{
  this->connect(m_Controls->m_MinSlider, SIGNAL(valueChanged(double)), SLOT(OnWindowBoundSlidersChanged()));
  this->connect(m_Controls->m_MaxSlider, SIGNAL(valueChanged(double)), SLOT(OnWindowBoundSlidersChanged()));
  this->connect(m_Controls->m_LevelSlider, SIGNAL(valueChanged(double)), SLOT(OnLevelWindowSlidersChanged()));
  this->connect(m_Controls->m_WindowSlider, SIGNAL(valueChanged(double)), SLOT(OnLevelWindowSlidersChanged()));
  this->connect(m_Controls->m_MinLimitDoubleSpinBox, SIGNAL(editingFinished()), SLOT(OnDataLimitSpinBoxesChanged()));
  this->connect(m_Controls->m_MaxLimitDoubleSpinBox, SIGNAL(editingFinished()), SLOT(OnDataLimitSpinBoxesChanged()));
  this->connect(m_Controls->m_LookupTableComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnLookupTableComboBoxChanged(int)));
  this->connect(m_Controls->m_ResetButton, SIGNAL(pressed()), this, SLOT(OnResetButtonPressed()));
  this->connect(m_Controls->m_SaveButton, SIGNAL(pressed()), this, SLOT(OnSaveButtonPressed()));
  this->connect(m_Controls->m_LoadButton, SIGNAL(pressed()), this, SLOT(OnLoadButtonPressed()));
  this->connect(m_Controls->m_NewButton, SIGNAL(pressed()), this, SLOT(OnNewButtonPressed()));

  this->connect(m_Controls->m_AddLabelButton, SIGNAL(pressed()), this, SLOT(OnAddLabelButtonPressed()));
  this->connect(m_Controls->m_RemoveLabelButton, SIGNAL(pressed()), this, SLOT(OnRemoveLabelButtonPressed()));
  this->connect(m_Controls->m_MoveLabelUpButton, SIGNAL(pressed()), this, SLOT(OnMoveLabelUpButtonPressed()));
  this->connect(m_Controls->m_MoveLabelDownButton, SIGNAL(pressed()), this, SLOT(OnMoveLabelDownButtonPressed()));
  this->connect(m_Controls->widget_LabelTable, SIGNAL(cellChanged(int,int)), this, SLOT(OnLabelMapTableCellChanged(int, int)));
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  QString pluginName = PluginActivator::GetInstance()->GetContext()->getPlugin()->getSymbolicName();
  berry::IPreferences::Pointer prefs = prefService->GetSystemPreferences()->Node(pluginName);
  assert(prefs);

  m_Precision = prefs->GetInt(ImageLookupTablesPreferencePage::PRECISION_NAME, 2);

  if (m_SelectedNodes.isEmpty())
  {
    this->BlockSignals(true);
  }

  m_Controls->m_MinSlider->setDecimals(m_Precision);
  m_Controls->m_MaxSlider->setDecimals(m_Precision);
  m_Controls->m_LevelSlider->setDecimals(m_Precision);
  m_Controls->m_WindowSlider->setDecimals(m_Precision);
  m_Controls->m_MinLimitDoubleSpinBox->setDecimals(m_Precision);
  m_Controls->m_MaxLimitDoubleSpinBox->setDecimals(m_Precision);

  if (m_SelectedNodes.isEmpty())
  {
    this->BlockSignals(false);
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::SetFocus()
{
  m_Controls->m_LookupTableComboBox->setFocus();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::EnableControls(bool enabled)
{
  m_Controls->m_LookupTableComboBox->setEnabled(enabled);
  m_Controls->m_LoadButton->setEnabled(enabled);
  m_Controls->m_NewButton->setEnabled(enabled);

  if (!enabled)
  {
    this->EnableScaleControls(false);
    this->EnableLabelControls(false);
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::EnableScaleControls(bool enabled)
{
  m_Controls->tabWidget->setTabEnabled(0, enabled);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::EnableLabelControls(bool enabled)
{
  m_Controls->tabWidget->setTabEnabled(1, enabled);

  if (enabled)
  {
    this->UpdateLabelMapTable();
  }
  else
  {
    m_Controls->widget_LabelTable->clearContents();
    m_Controls->widget_LabelTable->setRowCount(0);
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::BlockSignals(bool blocked)
{
  m_Controls->m_MinSlider->blockSignals(blocked);
  m_Controls->m_MaxSlider->blockSignals(blocked);
  m_Controls->m_WindowSlider->blockSignals(blocked);
  m_Controls->m_LevelSlider->blockSignals(blocked);
  m_Controls->m_MinLimitDoubleSpinBox->blockSignals(blocked);
  m_Controls->m_MaxLimitDoubleSpinBox->blockSignals(blocked);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnSelectionChanged(berry::IWorkbenchPart::Pointer /*source*/,
                                               const QList<mitk::DataNode::Pointer>& selectedNodes)
{
  if (selectedNodes != m_SelectedNodes)
  {
    bool isValid = this->IsSelectionValid(selectedNodes);

    if (m_SelectedNodes.isEmpty() && isValid)
    {
      m_SelectedNodes = selectedNodes;
      this->RegisterObservers();

      this->UpdateLookupTableComboBoxSelection();
    }
    else if (m_SelectedNodes.isEmpty() && !isValid)
    {
    }
    else if (isValid)
    {
      this->UnregisterObservers();
      m_SelectedNodes = selectedNodes;
      this->RegisterObservers();

      this->UpdateLookupTableComboBoxSelection();
    }
    else
    {
      this->UnregisterObservers();
      m_SelectedNodes.clear();
    }

    this->EnableControls(isValid);
  }
}


//-----------------------------------------------------------------------------
bool ImageLookupTablesView::IsSelectionValid(const QList<mitk::DataNode::Pointer>& selectedNodes)
{
  if (selectedNodes.isEmpty())
  {
    return false;
  }

  for (mitk::DataNode::Pointer node: selectedNodes)
  {
    // All nodes must be non null, non-helper images.
    if (node.IsNull())
    {
      return false;
    }
    else if (dynamic_cast<mitk::Image*>(node->GetData()) == nullptr)
    {
      return false;
    }

    bool isHelper;
    if (node->GetBoolProperty("helper object", isHelper) && isHelper)
    {
      return false;
    }

    bool isSelected;
    if (!node->GetBoolProperty("selected", isSelected) || !isSelected)
    {
      return false;
    }

    if (!niftk::IsNodeANonBinaryImage(node))
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::Activated()
{
  BaseView::Activated();

  berry::IWorkbenchPart::Pointer nullPart;
  this->OnSelectionChanged(nullPart, this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::RegisterObservers()
{
  itk::ReceptorMemberCommand<ImageLookupTablesView>::Pointer command
    = itk::ReceptorMemberCommand<ImageLookupTablesView>::New();

  command->SetCallbackFunction(this, &ImageLookupTablesView::OnPropertyChanged);

  mitk::BaseProperty::Pointer property = m_SelectedNodes[0]->GetProperty("levelwindow");
  m_LevelWindowPropertyObserverTag = property->AddObserver(itk::ModifiedEvent(), command);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::UnregisterObservers()
{
  mitk::BaseProperty::Pointer property = m_SelectedNodes[0]->GetProperty("levelwindow");
  property->RemoveObserver(m_LevelWindowPropertyObserverTag);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::UpdateLookupTableComboBoxSelection()
{
  std::string lookupTableName("");
  bool lookupTableNameFound = m_SelectedNodes[0]->GetStringProperty("LookupTableName", lookupTableName);

  signed int lookupTableIndex = -1;
  if (lookupTableNameFound)
  {
    lookupTableIndex = m_Controls->m_LookupTableComboBox->findText(QString::fromStdString(lookupTableName));
  }

  bool wasBlocked = m_Controls->m_LookupTableComboBox->blockSignals(true);
  if (lookupTableIndex > -1)
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(lookupTableIndex);
  }
  else
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(0);
  }
  m_Controls->m_LookupTableComboBox->blockSignals(wasBlocked);

  LookupTableProviderService* lutService = PluginActivator::GetInstance()->GetLookupTableProviderService();
  if (lutService == nullptr)
  {
    mitkThrow() << "Failed to find LookupTableProviderService." << std::endl;
  }

  bool isScaled = lutService->GetIsScaled(QString::fromStdString(lookupTableName));
  this->EnableScaleControls(isScaled);
  this->EnableLabelControls(!isScaled);

  if (isScaled)
  {
    this->UpdateLevelWindowControls();
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::UpdateLookupTableComboBoxEntries()
{
  bool en = m_Controls->m_LookupTableComboBox->blockSignals(true);
  int currentIndex = m_Controls->m_LookupTableComboBox->currentIndex();

  // create a lookup table
  LookupTableProviderService* lutService = PluginActivator::GetInstance()->GetLookupTableProviderService();
  if (lutService == nullptr)
  {
    mitkThrow() << "Failed to find LookupTableProviderService." << std::endl;
  }

  m_Controls->m_LookupTableComboBox->clear();
  m_Controls->m_LookupTableComboBox->addItem(" --- Scaled Lookup Tables --- ");

  std::vector<QString> names = lutService->GetTableNames();

  //// Populate combo box with lookup table names.
  for (unsigned int i = 0; i < names.size(); i++)
  {
    if ( lutService->GetIsScaled(names.at(i)) )
    {
      m_Controls->m_LookupTableComboBox->addItem(names.at(i));
    }
  }

  m_Controls->m_LookupTableComboBox->addItem(" ");
  m_Controls->m_LookupTableComboBox->addItem(" --- Labeled Lookup Tables --- ");

  for (unsigned int i = 0; i < names.size(); i++)
  {
    if ( !lutService->GetIsScaled(names.at(i)) )
    {
      m_Controls->m_LookupTableComboBox->addItem(names.at(i));
    }
  }

  m_Controls->m_LookupTableComboBox->setCurrentIndex(currentIndex);
  m_Controls->m_LookupTableComboBox->blockSignals(en);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnDataLimitSpinBoxesChanged()
{
  double rangeMin = m_Controls->m_MinLimitDoubleSpinBox->value();
  double rangeMax = m_Controls->m_MaxLimitDoubleSpinBox->value();

  for (mitk::DataNode::Pointer selectedNode: m_SelectedNodes)
  {
    mitk::LevelWindow levelWindow;
    selectedNode->GetLevelWindow(levelWindow);

    levelWindow.SetRangeMinMax(rangeMin, rangeMax);
    selectedNode->SetLevelWindow(levelWindow);
  }

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnPropertyChanged(const itk::EventObject&)
{
  this->UpdateLevelWindowControls();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::UpdateLevelWindowControls()
{
  assert(!m_SelectedNodes.isEmpty());

  mitk::LevelWindow levelWindow;
  m_SelectedNodes[0]->GetLevelWindow(levelWindow);

  double min = levelWindow.GetLowerWindowBound();
  double max = levelWindow.GetUpperWindowBound();
  double level = levelWindow.GetLevel();
  double window = levelWindow.GetWindow();
  double rangeMin = levelWindow.GetRangeMin();
  double rangeMax = levelWindow.GetRangeMax();

  for (auto it = m_SelectedNodes.begin() + 1; it < m_SelectedNodes.end(); ++it)
  {
    (*it)->GetLevelWindow(levelWindow);

    if (min > levelWindow.GetLowerWindowBound())
    {
      min = levelWindow.GetLowerWindowBound();
    }
    if (max < levelWindow.GetUpperWindowBound())
    {
      max = levelWindow.GetUpperWindowBound();
    }
    if (rangeMin > levelWindow.GetRangeMin())
    {
      rangeMin = levelWindow.GetRangeMin();
    }
    if (rangeMax < levelWindow.GetRangeMax())
    {
      rangeMax = levelWindow.GetRangeMax();
    }
  }

  double range = rangeMax - rangeMin;

  // Trac 1680 - don't forget, MIDAS generally deals with integer images
  // so the user requirements are such that they must be able to change
  // intensity ranges in steps of 1. If however, we are using float images
  // we will need to be able to change intensity values in much smaller steps.
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

  this->BlockSignals(true);

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

  m_Controls->m_MinSlider->setValue(min);
  m_Controls->m_MaxSlider->setValue(max);
  m_Controls->m_LevelSlider->setValue(level);
  m_Controls->m_WindowSlider->setValue(window);
  m_Controls->m_MinLimitDoubleSpinBox->setValue(rangeMin);
  m_Controls->m_MaxLimitDoubleSpinBox->setValue(rangeMax);

  this->BlockSignals(false);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnWindowBoundSlidersChanged()
{
  // Note: This method is called when one of the sliders has been moved
  // So, it's purpose is to update the other sliders to match.

  // Update them from controls
  double min = m_Controls->m_MinSlider->value();
  double max = m_Controls->m_MaxSlider->value();

  for (mitk::DataNode::Pointer selectedNode: m_SelectedNodes)
  {
    // Get the current values
    mitk::LevelWindow levelWindow;
    selectedNode->GetLevelWindow(levelWindow);
    levelWindow.SetWindowBounds(min, max);
    selectedNode->SetLevelWindow(levelWindow);
  }

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnLevelWindowSlidersChanged()
{
  // Note: This method is called when one of the sliders has been moved
  // So, it's purpose is to update the other sliders to match.

  // Update them from controls
  double window = m_Controls->m_WindowSlider->value();
  double level = m_Controls->m_LevelSlider->value();

  for (mitk::DataNode::Pointer selectedNode: m_SelectedNodes)
  {
    // Get the current values
    mitk::LevelWindow levelWindow;
    selectedNode->GetLevelWindow(levelWindow);
    levelWindow.SetLevelWindow(level, window);
    selectedNode->SetLevelWindow(levelWindow);
  }

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnLookupTablePropertyChanged(const itk::EventObject&)
{
  if (!m_SelectedNodes.isEmpty())
  {
    int comboIndex;
    m_SelectedNodes[0]->GetIntProperty("LookupTableIndex", comboIndex);
    this->OnLookupTableComboBoxChanged(comboIndex);
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnLookupTableComboBoxChanged(int comboBoxIndex)
{
  if (!m_SelectedNodes.isEmpty())
  {
    LookupTableProviderService* lutService = PluginActivator::GetInstance()->GetLookupTableProviderService();
    if (lutService == nullptr)
    {
      mitkThrow() << "Failed to find LookupTableProviderService." << std::endl;
    }

    QString lutName = m_Controls->m_LookupTableComboBox->itemText(comboBoxIndex);

    if( !lutService->CheckName(lutName) )
    {
      return;
    }

    for (mitk::DataNode::Pointer selectedNode: m_SelectedNodes)
    {
      selectedNode->SetStringProperty("LookupTableName", lutName.toStdString().c_str());
      bool isScaled = lutService->GetIsScaled(lutName);

      if (isScaled)
      {
        float lowestOpacity = 1;
        selectedNode->GetFloatProperty("Image Rendering.Lowest Value Opacity", lowestOpacity);

        float highestOpacity = 1;
        selectedNode->GetFloatProperty("Image Rendering.Highest Value Opacity", highestOpacity);

        // Get LUT from Micro Service.
        NamedLookupTableProperty::Pointer mitkLUTProperty = lutService->CreateLookupTableProperty(lutName, lowestOpacity, highestOpacity);
        selectedNode->ReplaceProperty("LookupTable", mitkLUTProperty);

        mitk::RenderingModeProperty::Pointer renderProp = mitk::RenderingModeProperty::New(mitk::RenderingModeProperty::LOOKUPTABLE_LEVELWINDOW_COLOR);
        selectedNode->ReplaceProperty("Image Rendering.Mode", renderProp);

        mitk::VtkResliceInterpolationProperty::Pointer resliceProp = mitk::VtkResliceInterpolationProperty::New(VTK_CUBIC_INTERPOLATION);
        selectedNode->ReplaceProperty("reslice interpolation", resliceProp);

        selectedNode->ReplaceProperty("texture interpolation", mitk::BoolProperty::New( true ));
      }
      else
      {
        // Get LUT from Micro Service.
        LabeledLookupTableProperty::Pointer mitkLUTProperty = lutService->CreateLookupTableProperty(lutName);
        selectedNode->ReplaceProperty("LookupTable", mitkLUTProperty);

        mitk::RenderingModeProperty::Pointer renderProp = mitk::RenderingModeProperty::New(mitk::RenderingModeProperty::LOOKUPTABLE_COLOR);
        selectedNode->ReplaceProperty("Image Rendering.Mode", renderProp);

        mitk::VtkResliceInterpolationProperty::Pointer resliceProp = mitk::VtkResliceInterpolationProperty::New(VTK_RESLICE_NEAREST);
        selectedNode->ReplaceProperty("reslice interpolation", resliceProp);

        selectedNode->ReplaceProperty("texture interpolation", mitk::BoolProperty::New( false ));
      }

      this->EnableScaleControls(isScaled);
      this->EnableLabelControls(!isScaled);

      // Force redraw.
      selectedNode->Update();
      selectedNode->Modified();
    }

    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnResetButtonPressed()
{
  for (auto selectedNode: m_SelectedNodes)
  {
    mitk::LevelWindow levelWindow;
    selectedNode->GetLevelWindow(levelWindow);

    float rangeMin = 0.0f;
    float rangeMax = 0.0f;

    if (selectedNode->GetFloatProperty("image data min", rangeMin)
        && selectedNode->GetFloatProperty("image data max", rangeMax))
    {
      levelWindow.SetRangeMinMax(rangeMin, rangeMax);
      levelWindow.SetWindowBounds(rangeMin, rangeMax);

      selectedNode->SetLevelWindow(levelWindow);
      selectedNode->Modified();
    }
  }

  this->UpdateLevelWindowControls();

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnLoadButtonPressed()
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

   // load a label
  QString filenameWithPath = QFileDialog::getOpenFileName(0, tr("Open File"), "", tr("Text files (*.txt);;XML files (*.xml);;LUT files (*.lut)"));

  if (filenameWithPath.isEmpty())
  {
    return;
  }

  LookupTableProviderService* lutService =
      PluginActivator::GetInstance()->GetLookupTableProviderService();

  if (lutService == nullptr)
  {
    mitkThrow() << "Failed to find LookupTableProviderService." << std::endl;
  }

  QString lutName = lutService->LoadLookupTable(filenameWithPath);

  if (lutName.isEmpty())
  {
    return;
  }

  this->UpdateLookupTableComboBoxEntries();

  // try to set the loaded reader as the selected container
  int index = m_Controls->m_LookupTableComboBox->findText(lutName);

  if (index > -1)
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(index);
  }

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  berry::IPreferences::Pointer prefs = prefService->GetSystemPreferences();

  // save the file to the list of names if not present
  QString cachedFileNames = prefs->Get("LABEL_MAP_NAMES", "");
  QString labelName = QFileInfo(filenameWithPath).baseName();

  if (!cachedFileNames.contains(labelName))
  {
    cachedFileNames.append(",");
    cachedFileNames.append(labelName.toStdString().c_str());

    prefs->Put("LABEL_MAP_NAMES", cachedFileNames);
  }

  // update the cached location of the file
  prefs->Put(labelName, filenameWithPath);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnSaveButtonPressed()
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());

  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    return;
  }

  QString lutName = m_Controls->m_LookupTableComboBox->currentText();

  if (lutName.isNull() || lutName.isEmpty())
  {
    lutName = QString("lookupTable.txt");
  }
  else if (!lutName.contains(".txt"))
  {
    lutName.append(".txt");
  }

  QFileInfo finfo(lutName);
  QString fileNameAndPath = QFileDialog::getSaveFileName(0, tr("Save File"), finfo.fileName(), tr("Text files (*.txt)"));

  if (fileNameAndPath.isEmpty())
  {
    return;
  }

  LookupTableContainer* newLUT =
      new LookupTableContainer(labelProperty->GetLookupTable()->GetVtkLookupTable(), labelProperty->GetLabels());
  newLUT->SetDisplayName(labelProperty->GetName());

  mitk::IOUtil::Save(newLUT, fileNameAndPath.toStdString());

  int index = fileNameAndPath.lastIndexOf("/")+1;
  QString labelName = fileNameAndPath.mid(index);
  index = labelName.lastIndexOf(".");
  labelName.truncate(index);

  int comboBoxIndex = -1;
  newLUT->SetOrder(comboBoxIndex);
  m_SelectedNodes[0]->GetIntProperty("LookupTableIndex", comboBoxIndex);

  LookupTableProviderService* lutService =
      PluginActivator::GetInstance()->GetLookupTableProviderService();

  if (lutService == nullptr)
  {
    mitkThrow() << "Failed to find LookupTableProviderService." << std::endl;
  }

  lutService->ReplaceLookupTableContainer(newLUT, newLUT->GetDisplayName());

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  berry::IPreferences::Pointer prefs = prefService->GetSystemPreferences();

  QString cachedFileNames;
  prefs->Get("LABEL_MAP_NAMES", cachedFileNames);

  // save the file to the list of names if not present
  if (!cachedFileNames.contains(labelName))
  {
    cachedFileNames.append(",");
    cachedFileNames.append(labelName.toStdString().c_str());

    prefs->Put("LABEL_MAP_NAMES", cachedFileNames);
  }

  // update the cached location of the file
  prefs->Put(labelName, fileNameAndPath);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnNewButtonPressed()
{
  // create an empty LookupTable
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  QString newLabelName = QInputDialog::getText(0, tr("Create New Label"),
                                         tr("New label name:"), QLineEdit::Normal );

  if (newLabelName.isEmpty())
  {
    return;
  }

  LookupTableProviderService* lutService = PluginActivator::GetInstance()->GetLookupTableProviderService();
  if (lutService == nullptr)
  {
    mitkThrow() << "Failed to find LookupTableProviderService." << std::endl;
  }

  float lowestOpacity = 1;
  m_SelectedNodes[0]->GetFloatProperty("Image Rendering.Lowest Value Opacity", lowestOpacity);

  float highestOpacity = 1;
  m_SelectedNodes[0]->GetFloatProperty("Image Rendering.Highest Value Opacity", highestOpacity);

  QColor lowColor(0, 0, 0, lowestOpacity);
  QColor highColor(0, 0, 0, highestOpacity);

  LookupTableContainer * newContainer = new LookupTableContainer(niftk::CreateEmptyLookupTable(lowColor, highColor));
  newContainer->SetDisplayName(newLabelName);
  newContainer->SetIsScaled(false);
  newContainer->SetOrder(lutService->GetNumberOfLookupTables());

  lutService->AddNewLookupTableContainer(newContainer);

  this->UpdateLookupTableComboBoxEntries();

  // try to set the loaded reader as the selected container
  int index = m_Controls->m_LookupTableComboBox->findText(newLabelName);
  if (index > -1)
  {
    m_Controls->m_LookupTableComboBox->setCurrentIndex(index);
  }
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::UpdateLabelMapTable()
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  bool en = m_Controls->widget_LabelTable->blockSignals(true);

  // initialize labels widget to empty
  m_Controls->widget_LabelTable->clearContents();
  m_Controls->widget_LabelTable->setRowCount(0);

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    m_Controls->widget_LabelTable->blockSignals(en);
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());

  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    m_Controls->widget_LabelTable->blockSignals(en);
    return;
  }

  // get labels and LUT
  LabeledLookupTableProperty::LabelListType labels = labelProperty->GetLabels();
  vtkSmartPointer<vtkLookupTable> vtkLUT = labelProperty->GetLookupTable()->GetVtkLookupTable();

  m_Controls->widget_LabelTable->setRowCount(labels.size());

  QSignalMapper* colorMapper = new QSignalMapper(this);

  for (unsigned int i = 0; i < labels.size(); i++)
  {
    // set value
    int value = labels.at(i).first;
    int vtkInd = value - vtkLUT->GetRange()[0] + 1;

    QTableWidgetItem * newValueItem = new QTableWidgetItem();
    newValueItem->setText(QString::number(value));
    m_Controls->widget_LabelTable->setItem(i, 1, newValueItem);

    // set name
    QTableWidgetItem * newNameItem = new QTableWidgetItem();
    newNameItem->setText(labels.at(i).second);
    m_Controls->widget_LabelTable->setItem(i, 2, newNameItem);

    // set color
    QPushButton* btnColor = new QPushButton;
    btnColor->setFixedWidth(35);
    btnColor->setAutoFillBackground(true);

    double rgb[3];
    vtkLUT->GetColor(value, rgb);

    QColor currColor(255 * rgb[0], 255 * rgb[1], 255 * rgb[2]);

    btnColor->setStyleSheet(QString("background-color:rgb(%1,%2, %3)")
                              .arg(currColor.red())
                              .arg(currColor.green())
                              .arg(currColor.blue())
                              );
    m_Controls->widget_LabelTable->setCellWidget(i, 0, btnColor);

    connect(btnColor, SIGNAL(clicked()), colorMapper, SLOT(map()));
    colorMapper->setMapping(btnColor, i);
  }

  connect(colorMapper, SIGNAL(mapped(int)), this, SLOT(OnColorButtonPressed(int)));
  m_Controls->widget_LabelTable->blockSignals(en);

  // Force redraw.
  m_SelectedNodes[0]->Update();
  m_SelectedNodes[0]->Modified();
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnAddLabelButtonPressed()
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());
  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    return;
  }

  LabeledLookupTableProperty::LabelListType labels = labelProperty->GetLabels();
  vtkSmartPointer<vtkLookupTable> oldLUT = labelProperty->GetLookupTable()->GetVtkLookupTable();

  bool en = m_Controls->widget_LabelTable->blockSignals(true);

  // get the range

  double* range = oldLUT->GetRange();
  QString newName(" ");

  int newValue = range[1];

  LookupTableContainer::LabelType newLabel = std::make_pair(newValue,newName);
  labels.push_back(newLabel);
  labelProperty->SetLabels(labels);

  // increment the range by 1
  vtkSmartPointer<vtkLookupTable> newLUT;
  newLUT.TakeReference(niftk::ResizeLookupTable(oldLUT,newValue+1));
  labelProperty->GetLookupTable()->SetVtkLookupTable(newLUT);

  UpdateLabelMapTable();

  m_Controls->widget_LabelTable->blockSignals(en);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnRemoveLabelButtonPressed()
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());
  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    return;
  }

  bool en = m_Controls->widget_LabelTable->blockSignals(true);

  QList<QTableWidgetSelectionRange> selectedItems = m_Controls->widget_LabelTable->selectedRanges();
  LabeledLookupTableProperty::LabelListType labels = labelProperty->GetLabels();
  vtkSmartPointer<vtkLookupTable> lut = labelProperty->GetLookupTable()->GetVtkLookupTable();

  QColor nanColor(lut->GetNanColor()[0], lut->GetNanColor()[1], lut->GetNanColor()[2], lut->GetNanColor()[3]);
  for (unsigned int i = 0; i < selectedItems.size(); i++)
  {
    int bottom = selectedItems.at(i).bottomRow()+1;
    int top = selectedItems.at(i).topRow();

    for (unsigned int j = top; j < bottom; j++)
    {
      int value = labels.at(j).first;
      vtkSmartPointer<vtkLookupTable> newLUT;
      newLUT.TakeReference(niftk::ChangeColor(lut, value, nanColor));
      labelProperty->GetLookupTable()->SetVtkLookupTable(newLUT);
    }

    labels.erase(labels.begin() + top, labels.begin() + bottom);
  }

  labelProperty->SetLabels(labels);
  UpdateLabelMapTable();

  m_Controls->widget_LabelTable->blockSignals(en);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnMoveLabelUpButtonPressed()
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());
  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    return;
  }

  bool en = m_Controls->widget_LabelTable->blockSignals(true);

  LabeledLookupTableProperty::LabelListType labels = labelProperty->GetLabels();
  QList<QTableWidgetSelectionRange> selectedItems = m_Controls->widget_LabelTable->selectedRanges();

  for (unsigned int i = 0; i < selectedItems.size(); i++)
  {
    int bottom = selectedItems.at(i).bottomRow()+1;
    int top = selectedItems.at(i).topRow();

    if (top == 0)
    {
      continue;
    }

    for (unsigned int j = top; j < bottom; j++)
    {
      std::iter_swap(labels.begin() + j - 1,labels.begin() + j);
    }
  }

  labelProperty->SetLabels(labels);
  UpdateLabelMapTable();

  m_Controls->widget_LabelTable->blockSignals(en);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnMoveLabelDownButtonPressed()
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());
  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    return;
  }

  bool en = m_Controls->widget_LabelTable->blockSignals(true);

  LabeledLookupTableProperty::LabelListType labels = labelProperty->GetLabels();
  QList<QTableWidgetSelectionRange> selectedItems = m_Controls->widget_LabelTable->selectedRanges();

  for (unsigned int i = 0; i < selectedItems.size(); i++)
  {
    int bottom = selectedItems.at(i).bottomRow() + 1;
    int top = selectedItems.at(i).topRow();

    if (bottom == labels.size())
    {
      continue;
    }

    for (unsigned int j = bottom; j > top; j--)
    {
      std::iter_swap(labels.begin()+ j - 1, labels.begin() + j);
    }
  }

  labelProperty->SetLabels(labels);
  UpdateLabelMapTable();

  m_Controls->widget_LabelTable->blockSignals(en);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnColorButtonPressed(int index)
{
  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());

  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    return;
  }

  bool en = m_Controls->widget_LabelTable->blockSignals(true);

  LabeledLookupTableProperty::LabelListType labels = labelProperty->GetLabels();
  vtkSmartPointer<vtkLookupTable> lut = labelProperty->GetLookupTable()->GetVtkLookupTable();

  int value = labels.at(index).first;

  double rgb[3];
  lut->GetColor(value, rgb);

  QColor initialColor(255 * rgb[0], 255 * rgb[1], 255 * rgb[2]);
  QColor newColor = QColorDialog::getColor(initialColor);

  if (newColor.spec() == 0)
  {
    m_Controls->widget_LabelTable->blockSignals(en);
    return;
  }

  vtkSmartPointer<vtkLookupTable> newLUT;
  newLUT.TakeReference(niftk::ChangeColor(lut, value,newColor));
  labelProperty->GetLookupTable()->SetVtkLookupTable(newLUT);

  QPushButton* btnColor = qobject_cast<QPushButton*>(m_Controls->widget_LabelTable->cellWidget(index, 0));
  if (btnColor != 0)
  {
    btnColor->setStyleSheet(QString("background-color:rgb(%1,%2, %3)")
                              .arg(newColor.red())
                              .arg(newColor.green())
                              .arg(newColor.blue())
                              );
  }

  m_Controls->widget_LabelTable->blockSignals(en);

  // Force redraw.
  m_SelectedNodes[0]->Update();
  m_SelectedNodes[0]->Modified();
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void ImageLookupTablesView::OnLabelMapTableCellChanged(int row, int column)
{
  if (column == 0)
  {
    return;
  }

  if (m_SelectedNodes.isEmpty())
  {
    return;
  }

  // get the labeledlookuptable property
  mitk::BaseProperty::Pointer mitkLUT = m_SelectedNodes[0]->GetProperty("LookupTable");
  if (mitkLUT.IsNull())
  {
    MITK_ERROR << "No lookup table assigned to " << m_SelectedNodes[0]->GetName();
    return;
  }

  LabeledLookupTableProperty::Pointer labelProperty
    = dynamic_cast<LabeledLookupTableProperty*>(mitkLUT.GetPointer());

  if (labelProperty.IsNull())
  {
    MITK_ERROR << "LookupTable is not a LabeledLookupTableProperty";
    return;
  }


  LabeledLookupTableProperty::LabelListType labels = labelProperty->GetLabels();
  vtkSmartPointer<vtkLookupTable> lut = labelProperty->GetLookupTable()->GetVtkLookupTable();

  if (column == 1)
  {
    QTableWidgetItem* item = m_Controls->widget_LabelTable->item(row,column);
    std::string valStr = item->text().toStdString();

    std::string::iterator valItr;
    for (valItr = valStr.begin(); valItr != valStr.end(); valItr++)
    {
      if (!isdigit(*valItr))
      {
        QMessageBox::warning(nullptr, "Label Map Editor", QString("Value must be a number. Resetting to old value."));

        QString value = QString::number(labels.at(row).first);
        item->setText(value);
        return;
      }
    }

    int newValue = atoi(valStr.c_str());

    for (unsigned int i = 0; i < labels.size(); i++)
    {
      if (i == row)
      {
        continue;
      }

      if (labels.at(i).first == newValue)
      {
        QMessageBox::warning(nullptr, "Label Map Editor", QString("Value is not unique. Resetting to old value."));

        QString oldValueStr = QString::number(labels.at(row).first);
        item->setText(oldValueStr);
        return;
      }
    }

    int oldValue = labels.at(row).first;

    vtkSmartPointer<vtkLookupTable> newLUT;
    newLUT.TakeReference(niftk::SwapColors(lut, oldValue, newValue));
    labelProperty->GetLookupTable()->SetVtkLookupTable(newLUT);

    labels.at(row).first = newValue;
  }
  else if (column == 2)
  {
    QTableWidgetItem* item = m_Controls->widget_LabelTable->item(row,column);
    labels.at(row).second = item->text();
  } // change the name

  labelProperty->SetLabels(labels);
  UpdateLabelMapTable();
}

}
