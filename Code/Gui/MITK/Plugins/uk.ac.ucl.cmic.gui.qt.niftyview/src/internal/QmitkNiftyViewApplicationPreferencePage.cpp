/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-28 10:00:55 +0100 (Wed, 28 Sep 2011) $
 Revision          : $Revision: 7379 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkNiftyViewApplicationPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QDoubleSpinBox>
#include <QMessageBox>
#include <QSpinBox>
#include <QComboBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_METHOD_NAME("window/level initialisation method");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_MIDAS("window/level initialisation by MIDAS convention");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_LEVELWINDOW("window/level initialisation by window level widget");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE("window/level initialisation by percentage of data range");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE_NAME("window/level initialisation percentage");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION("default reslice interpolation");
const std::string QmitkNiftyViewApplicationPreferencePage::IMAGE_TEXTURE_INTERPOLATION("default texture interpolation");

//-----------------------------------------------------------------------------
QmitkNiftyViewApplicationPreferencePage::QmitkNiftyViewApplicationPreferencePage()
: m_Parent(0)
, m_MainControl(0)
, m_UseMidasInitialisationRadioButton(0)
, m_UseLevelWindowRadioButton(0)
, m_UseImageDataRadioButton(0)
, m_PercentageOfDataRangeDoubleSpinBox(0)
, m_ResliceInterpolationComboBox(0)
, m_TextureInterpolationComboBox(0)
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
  m_Parent = parent;

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
  percentageFormLayout->addRow("Percentage:", m_PercentageOfDataRangeDoubleSpinBox);
  initialisationOptionsLayout->addLayout(percentageFormLayout);

  m_ResliceInterpolationComboBox = new QComboBox();
  m_ResliceInterpolationComboBox->insertItem(0, "none");
  m_ResliceInterpolationComboBox->insertItem(1, "linear");
  m_ResliceInterpolationComboBox->insertItem(2, "cubic");

  m_TextureInterpolationComboBox = new QComboBox();
  m_TextureInterpolationComboBox->insertItem(0, "none");
  m_TextureInterpolationComboBox->insertItem(1, "linear");

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow( "Image Level/Window initialization:", initialisationOptionsLayout );
  formLayout->addRow( "Image reslice interpolation:", m_ResliceInterpolationComboBox );
  formLayout->addRow( "Image texture interpolation:", m_TextureInterpolationComboBox );

  connect( m_UseLevelWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnLevelWindowRadioButtonChecked(bool)));
  connect( m_UseImageDataRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnImageDataRadioButtonChecked(bool)));
  connect( m_UseMidasInitialisationRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnMIDASInitialisationRadioButtonChecked(bool)));

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* QmitkNiftyViewApplicationPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool QmitkNiftyViewApplicationPreferencePage::PerformOk()
{
  QMessageBox::warning(m_Parent, tr("NiftyView"),
                                 tr("Changing application settings will require a restart.\n"),
                                 QMessageBox::Ok);
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

  m_PreferencesNode->Put(IMAGE_INITIALISATION_METHOD_NAME, method);
  m_PreferencesNode->PutDouble(IMAGE_INITIALISATION_PERCENTAGE_NAME, m_PercentageOfDataRangeDoubleSpinBox->value());
  m_PreferencesNode->PutInt(IMAGE_RESLICE_INTERPOLATION, m_ResliceInterpolationComboBox->currentIndex());
  m_PreferencesNode->PutInt(IMAGE_TEXTURE_INTERPOLATION, m_TextureInterpolationComboBox->currentIndex());
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
  if (method == IMAGE_INITIALISATION_MIDAS)
  {
    m_UseMidasInitialisationRadioButton->setChecked(true);
  }
  else if (method == IMAGE_INITIALISATION_LEVELWINDOW)
  {
    m_UseLevelWindowRadioButton->setChecked(true);
  }
  else if (method == IMAGE_INITIALISATION_PERCENTAGE)
  {
    m_UseImageDataRadioButton->setChecked(true);
  }
  m_PercentageOfDataRangeDoubleSpinBox->setValue(m_PreferencesNode->GetDouble(IMAGE_INITIALISATION_PERCENTAGE_NAME, 50));
  m_ResliceInterpolationComboBox->setCurrentIndex(m_PreferencesNode->GetInt(IMAGE_RESLICE_INTERPOLATION, 2));
  m_TextureInterpolationComboBox->setCurrentIndex(m_PreferencesNode->GetInt(IMAGE_TEXTURE_INTERPOLATION, 1));
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::OnMIDASInitialisationRadioButtonChecked(bool checked)
{
  if (m_Initializing) return;

  if (checked)
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(false);
  }
  else
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::OnLevelWindowRadioButtonChecked(bool checked)
{
  if (m_Initializing) return;

  if (checked)
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(false);
  }
  else
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPreferencePage::OnImageDataRadioButtonChecked(bool checked)
{
  if (m_Initializing) return;

  if (checked)
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(true);
  }
  else
  {
    m_PercentageOfDataRangeDoubleSpinBox->setEnabled(false);
  }
}
