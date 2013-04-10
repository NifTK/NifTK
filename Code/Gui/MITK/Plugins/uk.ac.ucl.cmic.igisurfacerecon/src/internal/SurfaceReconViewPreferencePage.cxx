/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceReconViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string SurfaceReconViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igisurfacerecon");

//-----------------------------------------------------------------------------
SurfaceReconViewPreferencePage::SurfaceReconViewPreferencePage()
: m_MainControl(0)
, m_DummyButton(0)
, m_Initializing(false)
, m_SurfaceReconViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
SurfaceReconViewPreferencePage::SurfaceReconViewPreferencePage(const SurfaceReconViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
SurfaceReconViewPreferencePage::~SurfaceReconViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void SurfaceReconViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void SurfaceReconViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_SurfaceReconViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_DummyButton = new QPushButton();
  formLayout->addRow("dummy", m_DummyButton);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* SurfaceReconViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool SurfaceReconViewPreferencePage::PerformOk()
{
  return true;
}


//-----------------------------------------------------------------------------
void SurfaceReconViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void SurfaceReconViewPreferencePage::Update()
{
}
