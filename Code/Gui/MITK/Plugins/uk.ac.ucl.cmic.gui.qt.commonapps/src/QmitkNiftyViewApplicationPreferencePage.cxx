/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyViewApplicationPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QDoubleSpinBox>
#include <QMessageBox>
#include <QSpinBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include <limits>

const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_METHOD_NAME("window/level initialisation method");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_MIDAS("window/level initialisation by MIDAS convention");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_LEVELWINDOW("window/level initialisation by window level widget");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE("window/level initialisation by percentage of data range");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE_NAME("window/level initialisation percentage");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_RANGE("window/level initialisation by set data range");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME("window/level initialisation lower bound");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME("window/level initialisation upper bound");


//-----------------------------------------------------------------------------
QmitkNiftyViewApplicationPreferencePage::QmitkNiftyViewApplicationPreferencePage()
: m_MainControl(0)
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
QmitkNiftyViewApplicationPreferencePage::QmitkNiftyViewApplicationPreferencePage(const QmitkNiftyViewApplicationPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QmitkNiftyViewApplicationPreferencePage::~QmitkNiftyViewApplicationPreferencePage()
{
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_PreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.gui.qt.niftyview");

  m_MainControl = new QWidget(parent);

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

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow( "Image Level/Window initialization:", initialisationOptionsLayout );

  connect( m_UseLevelWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnLevelWindowRadioButtonChecked(bool)));
  connect( m_UseImageDataRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnImageDataRadioButtonChecked(bool)));
  connect( m_UseMidasInitialisationRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnMIDASInitialisationRadioButtonChecked(bool)));
  connect( m_UseSetRange, SIGNAL(toggled(bool)), this, SLOT(OnIntensityRangeRadioButtonChecked(bool)));

  m_MainControl->setLayout(formLayout);
  m_Initializing = false;

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* QmitkNiftyViewApplicationPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool QmitkNiftyViewApplicationPreferencePage::PerformOk()
{
  std::string method;

  if (m_UseMidasInitialisationRadioButton->isChecked())
  {
    method = IMAGE_INITIALISATION_MIDAS;
  }
  else if (m_UseLevelWindowRadioButton->isChecked())
  {
    method = IMAGE_INITIALISATION_LEVELWINDOW;
  }
  else if (m_UseImageDataRadioButton->isChecked())
  {
    method = IMAGE_INITIALISATION_PERCENTAGE;
  }
  else if (m_UseSetRange->isChecked())
  {
    method = IMAGE_INITIALISATION_RANGE;
  }

  m_PreferencesNode->Put(IMAGE_INITIALISATION_METHOD_NAME, method);
  m_PreferencesNode->PutDouble(IMAGE_INITIALISATION_PERCENTAGE_NAME, m_PercentageOfDataRangeDoubleSpinBox->value());
  m_PreferencesNode->PutInt(IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME, m_RangeLowerBound->value());
  m_PreferencesNode->PutInt(IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME, m_RangeUpperBound->value());
  return true;
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::Update()
{
  std::string method = m_PreferencesNode->Get(IMAGE_INITIALISATION_METHOD_NAME, IMAGE_INITIALISATION_MIDAS);
  if (method == IMAGE_INITIALISATION_LEVELWINDOW)
  {
    m_UseLevelWindowRadioButton->setChecked(true);
  }
  else if (method == IMAGE_INITIALISATION_PERCENTAGE)
  {
    m_UseImageDataRadioButton->setChecked(true);
  }
  else if (method == IMAGE_INITIALISATION_RANGE)
  {
    m_UseSetRange->setChecked(true);
  }
  else
  {
    m_UseMidasInitialisationRadioButton->setChecked(true);
  }

  m_PercentageOfDataRangeDoubleSpinBox->setValue(m_PreferencesNode->GetDouble(IMAGE_INITIALISATION_PERCENTAGE_NAME, 50));
  m_RangeLowerBound->setValue(m_PreferencesNode->GetInt(IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME, 0));
  m_RangeUpperBound->setValue(m_PreferencesNode->GetInt(IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME, 255));
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::OnMIDASInitialisationRadioButtonChecked(bool checked)
{
  this->UpdateSpinBoxes();
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::OnLevelWindowRadioButtonChecked(bool checked)
{
  this->UpdateSpinBoxes();
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::OnImageDataRadioButtonChecked(bool checked)
{
  this->UpdateSpinBoxes();
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::OnIntensityRangeRadioButtonChecked(bool checked)
{
  this->UpdateSpinBoxes();
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::UpdateSpinBoxes()
{
  if (m_Initializing) return;

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
