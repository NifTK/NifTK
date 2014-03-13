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
#include <mitkPointBasedRegistration.h>

const std::string PointRegViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igipointreg");
const std::string PointRegViewPreferencePage::USE_ICP_INITIALISATION("use ICP initialisation");
const std::string PointRegViewPreferencePage::USE_POINT_ID_FOR_MATCHING("use point ID for matching");

//-----------------------------------------------------------------------------
PointRegViewPreferencePage::PointRegViewPreferencePage()
: m_MainControl(NULL)
, m_UseICPInitialisation(NULL)
, m_UsePointIDForMatching(NULL)
, m_Initializing(false)
, m_PointRegViewPreferencesNode(NULL)
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

  m_UseICPInitialisation = new QCheckBox();
  formLayout->addRow("use ICP initialisation", m_UseICPInitialisation);

  m_UsePointIDForMatching = new QCheckBox();
  formLayout->addRow("use point ID for matching", m_UsePointIDForMatching);

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
  m_PointRegViewPreferencesNode->PutBool(PointRegViewPreferencePage::USE_ICP_INITIALISATION, m_UseICPInitialisation->isChecked());
  m_PointRegViewPreferencesNode->PutBool(PointRegViewPreferencePage::USE_POINT_ID_FOR_MATCHING, m_UsePointIDForMatching->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void PointRegViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void PointRegViewPreferencePage::Update()
{
  m_UseICPInitialisation->setChecked(m_PointRegViewPreferencesNode->GetBool(PointRegViewPreferencePage::USE_ICP_INITIALISATION, mitk::PointBasedRegistration::DEFAULT_USE_ICP_INITIALISATION));
  m_UsePointIDForMatching->setChecked(m_PointRegViewPreferencesNode->GetBool(PointRegViewPreferencePage::USE_POINT_ID_FOR_MATCHING, mitk::PointBasedRegistration::DEFAULT_USE_POINT_ID_TO_MATCH));
}
