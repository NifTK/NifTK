/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TagTrackerViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QDoubleSpinBox>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string TagTrackerViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igitagtracker");
const bool TagTrackerViewPreferencePage::LISTEN_TO_EVENT_BUS(true);
const std::string TagTrackerViewPreferencePage::LISTEN_TO_EVENT_BUS_NAME("listen to event bus");
const bool TagTrackerViewPreferencePage::DO_MONO_LEFT_CAMERA(false);
const std::string TagTrackerViewPreferencePage::DO_MONO_LEFT_CAMERA_NAME("mono left camera");

//-----------------------------------------------------------------------------
TagTrackerViewPreferencePage::TagTrackerViewPreferencePage()
: m_MainControl(0)
, m_ListenToEventBusPulse()
, m_ManualUpdate(0)
, m_DoMonoLeftCamera(0)
, m_Initializing(false)
, m_TagTrackerViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
TagTrackerViewPreferencePage::TagTrackerViewPreferencePage(const TagTrackerViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
TagTrackerViewPreferencePage::~TagTrackerViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void TagTrackerViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void TagTrackerViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_TagTrackerViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_ListenToEventBusPulse = new QRadioButton();
  formLayout->addRow("listen to event bus", m_ListenToEventBusPulse);

  m_ManualUpdate = new QRadioButton();
  formLayout->addRow("manual update", m_ManualUpdate);

  m_DoMonoLeftCamera = new QCheckBox();
  formLayout->addRow("mono left camera only", m_DoMonoLeftCamera);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* TagTrackerViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool TagTrackerViewPreferencePage::PerformOk()
{
  if (m_ListenToEventBusPulse->isChecked())
  {
    m_TagTrackerViewPreferencesNode->PutBool(LISTEN_TO_EVENT_BUS_NAME, true);
  }
  else
  {
    m_TagTrackerViewPreferencesNode->PutBool(LISTEN_TO_EVENT_BUS_NAME, false);
  }
  m_TagTrackerViewPreferencesNode->PutBool(DO_MONO_LEFT_CAMERA_NAME, m_DoMonoLeftCamera->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void TagTrackerViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void TagTrackerViewPreferencePage::Update()
{
  bool listenToEventBus = m_TagTrackerViewPreferencesNode->GetBool(LISTEN_TO_EVENT_BUS_NAME, LISTEN_TO_EVENT_BUS);
  if (listenToEventBus)
  {
    m_ListenToEventBusPulse->setChecked(true);
  }
  else
  {
    m_ManualUpdate->setChecked(true);
  }
  m_DoMonoLeftCamera->setChecked(m_TagTrackerViewPreferencesNode->GetBool(DO_MONO_LEFT_CAMERA_NAME, DO_MONO_LEFT_CAMERA));
}
