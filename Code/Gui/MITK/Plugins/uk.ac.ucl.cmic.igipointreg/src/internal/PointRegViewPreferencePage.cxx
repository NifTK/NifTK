/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "PointRegViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string PointRegViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igipointreg");

//-----------------------------------------------------------------------------
PointRegViewPreferencePage::PointRegViewPreferencePage()
: m_MainControl(0)
, m_DummyButton(0)
, m_Initializing(false)
, m_PointRegViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
PointRegViewPreferencePage::PointRegViewPreferencePage(const PointRegViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
PointRegViewPreferencePage::~PointRegViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void PointRegViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void PointRegViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_PointRegViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_DummyButton = new QPushButton();
  formLayout->addRow("dummy", m_DummyButton);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* PointRegViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool PointRegViewPreferencePage::PerformOk()
{
  return true;
}


//-----------------------------------------------------------------------------
void PointRegViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void PointRegViewPreferencePage::Update()
{
}
