/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegmentorPreferencePage.h"

#include <QFormLayout>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>
#include <ctkPathLineEdit.h>

#include <niftkBaseView.h>

namespace niftk
{

const QString CaffeSegmentorPreferencePage::PREFERENCES_NODE_NAME("/uk_ac_ucl_cmic_caffesegmentor");
const QString CaffeSegmentorPreferencePage::NETWORK_DESCRIPTION_FILE_NAME("network description");
const QString CaffeSegmentorPreferencePage::NETWORK_WEIGHTS_FILE_NAME("network weights");

//-----------------------------------------------------------------------------
CaffeSegmentorPreferencePage::CaffeSegmentorPreferencePage()
: m_MainControl(0)
, m_Initializing(false)
{

}


//-----------------------------------------------------------------------------
CaffeSegmentorPreferencePage::CaffeSegmentorPreferencePage(const CaffeSegmentorPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
CaffeSegmentorPreferencePage::~CaffeSegmentorPreferencePage()
{
}


//-----------------------------------------------------------------------------
void CaffeSegmentorPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void CaffeSegmentorPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_CaffeSegmentorPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;
  m_MainControl->setLayout(formLayout);

  m_NetworkDescriptionFileName = new ctkPathLineEdit();
  formLayout->addRow("network description file name", m_NetworkDescriptionFileName);

  m_NetworkWeightsFileName = new ctkPathLineEdit();
  formLayout->addRow("network weights file name", m_NetworkWeightsFileName);

  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* CaffeSegmentorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool CaffeSegmentorPreferencePage::PerformOk()
{
  m_CaffeSegmentorPreferencesNode->Put(CaffeSegmentorPreferencePage::NETWORK_DESCRIPTION_FILE_NAME, m_NetworkDescriptionFileName->currentPath());
  m_CaffeSegmentorPreferencesNode->Put(CaffeSegmentorPreferencePage::NETWORK_WEIGHTS_FILE_NAME, m_NetworkWeightsFileName->currentPath());
  return true;
}


//-----------------------------------------------------------------------------
void CaffeSegmentorPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void CaffeSegmentorPreferencePage::Update()
{
  m_NetworkDescriptionFileName->setCurrentPath(m_CaffeSegmentorPreferencesNode->Get(CaffeSegmentorPreferencePage::NETWORK_DESCRIPTION_FILE_NAME, ""));
  m_NetworkWeightsFileName->setCurrentPath(m_CaffeSegmentorPreferencesNode->Get(CaffeSegmentorPreferencePage::NETWORK_WEIGHTS_FILE_NAME, ""));
}

} // end namespace
