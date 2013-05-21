/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TrackedPointerViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QCheckBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>
#include <mitkTrackedPointerCommand.h>

const std::string TrackedPointerViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igitrackedpointer");
const std::string TrackedPointerViewPreferencePage::UPDATE_VIEW_COORDINATE_NAME("update view coordinate");

//-----------------------------------------------------------------------------
TrackedPointerViewPreferencePage::TrackedPointerViewPreferencePage()
: m_MainControl(0)
, m_UpdateViewCoordinate(0)
, m_Initializing(false)
, m_TrackedPointerViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
TrackedPointerViewPreferencePage::TrackedPointerViewPreferencePage(const TrackedPointerViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
TrackedPointerViewPreferencePage::~TrackedPointerViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void TrackedPointerViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void TrackedPointerViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_TrackedPointerViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_UpdateViewCoordinate = new QCheckBox();
  m_UpdateViewCoordinate->setChecked(false);
  formLayout->addRow("update view coordinate", m_UpdateViewCoordinate);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* TrackedPointerViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool TrackedPointerViewPreferencePage::PerformOk()
{
  m_TrackedPointerViewPreferencesNode->PutBool(UPDATE_VIEW_COORDINATE_NAME, m_UpdateViewCoordinate->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void TrackedPointerViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void TrackedPointerViewPreferencePage::Update()
{
  bool updateViewCoordinate = m_TrackedPointerViewPreferencesNode->GetBool(UPDATE_VIEW_COORDINATE_NAME, mitk::TrackedPointerCommand::UPDATE_VIEW_COORDINATE_DEFAULT);
  m_UpdateViewCoordinate->setChecked(updateViewCoordinate);
}
