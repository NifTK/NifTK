/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "PivotCalibrationViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QSpinBox>
#include <QMessageBox>
#include <QPushButton>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string PivotCalibrationViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igipivotcalibration");
const std::string PivotCalibrationViewPreferencePage::OUTPUT_DIRECTORY("Output directory");

//-----------------------------------------------------------------------------
PivotCalibrationViewPreferencePage::PivotCalibrationViewPreferencePage()
: m_MainControl(0)
, m_Initializing(false)
, m_PivotCalibrationViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
PivotCalibrationViewPreferencePage::PivotCalibrationViewPreferencePage(const PivotCalibrationViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
PivotCalibrationViewPreferencePage::~PivotCalibrationViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void PivotCalibrationViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void PivotCalibrationViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_PivotCalibrationViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_OutputDirectoryChooser = new ctkPathLineEdit();
  m_OutputDirectoryChooser->setFilters(ctkPathLineEdit::Dirs);
  m_OutputDirectoryChooser->setOptions(ctkPathLineEdit::ShowDirsOnly);
  m_OutputDirectoryChooser->setCurrentPath(tr("D:/data/"));
  formLayout->addRow("Output directory", m_OutputDirectoryChooser);

  //spacer
  QSpacerItem *spacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  QVBoxLayout* vBoxLayout = new QVBoxLayout;
  vBoxLayout->addLayout(formLayout);
  vBoxLayout->addSpacerItem(spacer);

  //m_MainControl->setLayout(formLayout);
  m_MainControl->setLayout(vBoxLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* PivotCalibrationViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool PivotCalibrationViewPreferencePage::PerformOk()
{
  m_PivotCalibrationViewPreferencesNode->Put(OUTPUT_DIRECTORY, m_OutputDirectoryChooser->currentPath().toStdString());

  return true;
}


//-----------------------------------------------------------------------------
void PivotCalibrationViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void PivotCalibrationViewPreferencePage::Update()
{
  m_OutputDirectoryChooser->setCurrentPath(QString(m_PivotCalibrationViewPreferencesNode->Get(OUTPUT_DIRECTORY, " ").c_str()));
}
