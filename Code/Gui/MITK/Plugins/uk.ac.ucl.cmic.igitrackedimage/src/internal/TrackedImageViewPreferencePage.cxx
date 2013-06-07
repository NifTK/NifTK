/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TrackedImageViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>
#include <ctkPathLineEdit.h>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string TrackedImageViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igitrackedimage");
const std::string TrackedImageViewPreferencePage::CALIBRATION_FILE_NAME("calibration file name");

//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::TrackedImageViewPreferencePage()
: m_MainControl(0)
, m_CalibrationFileName(0)
, m_Initializing(false)
, m_TrackedImageViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::TrackedImageViewPreferencePage(const TrackedImageViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
TrackedImageViewPreferencePage::~TrackedImageViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_TrackedImageViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_CalibrationFileName = new ctkPathLineEdit();
  formLayout->addRow("calibration matrix file name", m_CalibrationFileName);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* TrackedImageViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool TrackedImageViewPreferencePage::PerformOk()
{
  m_TrackedImageViewPreferencesNode->Put(CALIBRATION_FILE_NAME, m_CalibrationFileName->currentPath().toStdString());
  return true;
}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void TrackedImageViewPreferencePage::Update()
{
  m_CalibrationFileName->setCurrentPath(QString(m_TrackedImageViewPreferencesNode->Get(CALIBRATION_FILE_NAME, "").c_str()));
}
