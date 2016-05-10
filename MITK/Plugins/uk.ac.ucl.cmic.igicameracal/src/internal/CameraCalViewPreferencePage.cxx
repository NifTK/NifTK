/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CameraCalViewPreferencePage.h"
#include "ui_CameraCalViewPreferencePage.h"

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

namespace niftk
{

const QString CameraCalViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igicameracal");

//-----------------------------------------------------------------------------
CameraCalViewPreferencePage::CameraCalViewPreferencePage()
: m_Control(nullptr)
, m_Ui(new Ui::CameraCalViewPreferencePage)
, m_Initializing(false)
, m_CameraCalViewPreferencesNode(NULL)
{
}


//-----------------------------------------------------------------------------
CameraCalViewPreferencePage::CameraCalViewPreferencePage(const CameraCalViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
CameraCalViewPreferencePage::~CameraCalViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  m_Control = new QWidget(parent);
  m_Ui->setupUi(m_Control);

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_CameraCalViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* CameraCalViewPreferencePage::GetQtControl() const
{
  return m_Control;
}


//-----------------------------------------------------------------------------
bool CameraCalViewPreferencePage::PerformOk()
{
  return true;
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void CameraCalViewPreferencePage::Update()
{
}

} // end namespace
