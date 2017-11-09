/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseApplicationPreferencePage.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QMessageBox>
#include <QRadioButton>
#include <QVBoxLayout>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include "internal/niftkPluginActivator.h"

namespace niftk
{

const QString BaseApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION("default reslice interpolation");
const QString BaseApplicationPreferencePage::IMAGE_TEXTURE_INTERPOLATION("default texture interpolation");
const QString BaseApplicationPreferencePage::LOWEST_VALUE_OPACITY("lowest value opacity");
const QString BaseApplicationPreferencePage::HIGHEST_VALUE_OPACITY("highest value opacity");
const QString BaseApplicationPreferencePage::BINARY_OPACITY_NAME("binary opacity");
const double BaseApplicationPreferencePage::BINARY_OPACITY_VALUE = 1.0;

//-----------------------------------------------------------------------------
BaseApplicationPreferencePage::BaseApplicationPreferencePage()
: m_MainControl(0)
, m_ResliceInterpolation(0)
, m_TextureInterpolation(0)
, m_LowestValueOpacity(0)
, m_HighestValueOpacity(0)
, m_BinaryOpacity(0)
, m_UseMidasInitialisationRadioButton(0)
, m_UseLevelWindowRadioButton(0)
, m_UseImageDataRadioButton(0)
, m_PercentageOfDataRangeDoubleSpinBox(0)
, m_UseSetRange(0)
, m_RangeLowerBound(0)
, m_RangeUpperBound(0)
, m_Initializing(false)
{
}


//-----------------------------------------------------------------------------
BaseApplicationPreferencePage::BaseApplicationPreferencePage(const BaseApplicationPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
BaseApplicationPreferencePage::~BaseApplicationPreferencePage()
{
}


//-----------------------------------------------------------------------------
void BaseApplicationPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void BaseApplicationPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  m_PreferencesNode = prefService->GetSystemPreferences();

  m_MainControl = new QWidget(parent);

  m_ResliceInterpolation = new QComboBox();
  m_ResliceInterpolation->insertItem(0, "none");
  m_ResliceInterpolation->insertItem(1, "linear");
  m_ResliceInterpolation->insertItem(2, "cubic");

  m_TextureInterpolation = new QComboBox();
  m_TextureInterpolation->insertItem(0, "none");
  m_TextureInterpolation->insertItem(1, "linear");

  m_LowestValueOpacity = new QDoubleSpinBox();
  m_LowestValueOpacity->setMinimum(0);
  m_LowestValueOpacity->setMaximum(1);
  m_LowestValueOpacity->setSingleStep(0.1);
  m_LowestValueOpacity->setValue(1);
  m_HighestValueOpacity = new QDoubleSpinBox();
  m_HighestValueOpacity->setMinimum(0);
  m_HighestValueOpacity->setMaximum(1);
  m_HighestValueOpacity->setSingleStep(0.1);
  m_HighestValueOpacity->setValue(1);

  m_BinaryOpacity = new QDoubleSpinBox();
  m_BinaryOpacity->setMinimum(0);
  m_BinaryOpacity->setMaximum(1);
  m_BinaryOpacity->setSingleStep(0.1);

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow( "Default image reslice interpolation:", m_ResliceInterpolation );
  formLayout->addRow( "Default image texture interpolation:", m_TextureInterpolation );
  formLayout->addRow( "Default lowest lookup table value opacity:", m_LowestValueOpacity);
  formLayout->addRow( "Default highest lookup table value opacity:", m_HighestValueOpacity);
  formLayout->addRow( "Default opacity when loading binary images:", m_BinaryOpacity);

  QVBoxLayout* initialisationOptionsLayout = new QVBoxLayout;
  m_UseMidasInitialisationRadioButton = new QRadioButton( "as per MIDAS default", m_MainControl);
  initialisationOptionsLayout->addWidget( m_UseMidasInitialisationRadioButton );
  m_UseLevelWindowRadioButton = new QRadioButton( "from Level/Window widget", m_MainControl);
  initialisationOptionsLayout->addWidget( m_UseLevelWindowRadioButton );

  m_UseImageDataRadioButton = new QRadioButton( "from image data", m_MainControl);
  initialisationOptionsLayout->addWidget( m_UseImageDataRadioButton );
  m_PercentageOfDataRangeDoubleSpinBox = new QDoubleSpinBox(m_MainControl );
  m_PercentageOfDataRangeDoubleSpinBox->setMinimum(0);
  m_PercentageOfDataRangeDoubleSpinBox->setMaximum(100);
  m_PercentageOfDataRangeDoubleSpinBox->setSingleStep(0.1);
  QFormLayout *percentageFormLayout = new QFormLayout;
  percentageFormLayout->addRow("percentage:", m_PercentageOfDataRangeDoubleSpinBox);
  initialisationOptionsLayout->addLayout(percentageFormLayout);

  m_UseSetRange = new QRadioButton( "from set range", m_MainControl);
  initialisationOptionsLayout->addWidget( m_UseSetRange );
  m_RangeLowerBound = new QSpinBox(m_MainControl);
  m_RangeLowerBound->setMinimum(std::numeric_limits<int>::min());
  m_RangeLowerBound->setMaximum(std::numeric_limits<int>::max());
  m_RangeLowerBound->setValue(0);
  m_RangeUpperBound = new QSpinBox(m_MainControl);
  m_RangeUpperBound->setMinimum(std::numeric_limits<int>::min());
  m_RangeUpperBound->setMaximum(std::numeric_limits<int>::max());
  m_RangeUpperBound->setValue(255);
  QFormLayout *defaultRangeFormLayout = new QFormLayout;
  defaultRangeFormLayout->addRow("min:", m_RangeLowerBound);
  defaultRangeFormLayout->addRow("max:", m_RangeUpperBound);
  initialisationOptionsLayout->addLayout(defaultRangeFormLayout);

  formLayout->addRow( "Image Level/Window initialization:", initialisationOptionsLayout );

  this->connect( m_UseLevelWindowRadioButton, SIGNAL(toggled(bool)), SLOT(UpdateSpinBoxes()));
  this->connect( m_UseImageDataRadioButton, SIGNAL(toggled(bool)), SLOT(UpdateSpinBoxes()));
  this->connect( m_UseMidasInitialisationRadioButton, SIGNAL(toggled(bool)), SLOT(UpdateSpinBoxes()));
  this->connect( m_UseSetRange, SIGNAL(toggled(bool)), SLOT(UpdateSpinBoxes()));

  m_MainControl->setLayout(formLayout);
  m_Initializing = false;

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* BaseApplicationPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool BaseApplicationPreferencePage::PerformOk()
{
  m_PreferencesNode->PutInt(IMAGE_RESLICE_INTERPOLATION, m_ResliceInterpolation->currentIndex());
  m_PreferencesNode->PutInt(IMAGE_TEXTURE_INTERPOLATION, m_TextureInterpolation->currentIndex());
  m_PreferencesNode->PutDouble(LOWEST_VALUE_OPACITY, m_LowestValueOpacity->value());
  m_PreferencesNode->PutDouble(HIGHEST_VALUE_OPACITY, m_HighestValueOpacity->value());
  m_PreferencesNode->PutDouble(BINARY_OPACITY_NAME, m_BinaryOpacity->value());

  QString method;

  if (m_UseMidasInitialisationRadioButton->isChecked())
  {
    method = niftk::PluginActivator::IMAGE_INITIALISATION_MIDAS;
  }
  else if (m_UseLevelWindowRadioButton->isChecked())
  {
    method = niftk::PluginActivator::IMAGE_INITIALISATION_LEVELWINDOW;
  }
  else if (m_UseImageDataRadioButton->isChecked())
  {
    method = niftk::PluginActivator::IMAGE_INITIALISATION_PERCENTAGE;
  }
  else if (m_UseSetRange->isChecked())
  {
    method = niftk::PluginActivator::IMAGE_INITIALISATION_RANGE;
  }

  m_PreferencesNode->Put(niftk::PluginActivator::IMAGE_INITIALISATION_METHOD_NAME, method);
  m_PreferencesNode->PutDouble(niftk::PluginActivator::IMAGE_INITIALISATION_PERCENTAGE, m_PercentageOfDataRangeDoubleSpinBox->value());
  m_PreferencesNode->PutInt(niftk::PluginActivator::IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME, m_RangeLowerBound->value());
  m_PreferencesNode->PutInt(niftk::PluginActivator::IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME, m_RangeUpperBound->value());

  return true;
}


//-----------------------------------------------------------------------------
void BaseApplicationPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void BaseApplicationPreferencePage::Update()
{
  m_ResliceInterpolation->setCurrentIndex(m_PreferencesNode->GetInt(IMAGE_RESLICE_INTERPOLATION, 2));
  m_TextureInterpolation->setCurrentIndex(m_PreferencesNode->GetInt(IMAGE_TEXTURE_INTERPOLATION, 1));
  m_LowestValueOpacity->setValue(m_PreferencesNode->GetDouble(LOWEST_VALUE_OPACITY, 1));
  m_HighestValueOpacity->setValue(m_PreferencesNode->GetDouble(HIGHEST_VALUE_OPACITY, 1));
  m_BinaryOpacity->setValue(m_PreferencesNode->GetDouble(BINARY_OPACITY_NAME, BaseApplicationPreferencePage::BINARY_OPACITY_VALUE));

  QString method = m_PreferencesNode->Get(niftk::PluginActivator::IMAGE_INITIALISATION_METHOD_NAME, niftk::PluginActivator::IMAGE_INITIALISATION_PERCENTAGE);
  if (method == niftk::PluginActivator::IMAGE_INITIALISATION_LEVELWINDOW)
  {
    m_UseLevelWindowRadioButton->setChecked(true);
  }
  else if (method == niftk::PluginActivator::IMAGE_INITIALISATION_PERCENTAGE)
  {
    m_UseImageDataRadioButton->setChecked(true);
  }
  else if (method == niftk::PluginActivator::IMAGE_INITIALISATION_RANGE)
  {
    m_UseSetRange->setChecked(true);
  }
  else
  {
    m_UseMidasInitialisationRadioButton->setChecked(true);
  }

  m_PercentageOfDataRangeDoubleSpinBox->setValue(m_PreferencesNode->GetDouble(niftk::PluginActivator::IMAGE_INITIALISATION_PERCENTAGE, 50));
  m_RangeLowerBound->setValue(m_PreferencesNode->GetInt(niftk::PluginActivator::IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME, 0));
  m_RangeUpperBound->setValue(m_PreferencesNode->GetInt(niftk::PluginActivator::IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME, 255));
}


//-----------------------------------------------------------------------------
void BaseApplicationPreferencePage::UpdateSpinBoxes()
{
  if (m_Initializing)
  {
    return;
  }

  if (m_UseSetRange->isChecked())
  {
    m_RangeLowerBound->setEnabled(true);
    m_RangeUpperBound->setEnabled(true);
  }
  else
  {
    m_RangeLowerBound->setEnabled(false);
    m_RangeUpperBound->setEnabled(false);
  }
  if (m_UseImageDataRadioButton->isChecked())
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(true);
  }
  else
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(false);
  }
}

} // end namespace
