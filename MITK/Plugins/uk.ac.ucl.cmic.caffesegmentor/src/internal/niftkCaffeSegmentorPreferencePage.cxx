/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegmentorPreferencePage.h"
#include "niftkCaffeSegmentorView.h"
#include <QFormLayout>
#include <QCheckBox>
#include <QLineEdit>
#include <QSpinBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>
#include <ctkPathLineEdit.h>

#include <niftkBaseView.h>

namespace niftk
{

//-----------------------------------------------------------------------------
CaffeSegmentorPreferencePage::CaffeSegmentorPreferencePage()
: m_MainControl(0)
, m_Initializing(false)
, m_CaffeSegPrefs(nullptr)
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
  m_CaffeSegmentorPreferencesNode = prefService->GetSystemPreferences()->Node(niftk::CaffeSegmentorView::VIEW_ID);

  m_MainControl = new QWidget(parent);

  m_CaffeSegPrefs = new CaffePreferencesWidget();
  m_MainControl->setLayout(m_CaffeSegPrefs->GetUILayout());

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
  m_CaffeSegmentorPreferencesNode->Put(CaffePreferencesWidget::NETWORK_DESCRIPTION_FILE_NAME, m_CaffeSegPrefs->GetNetworkDescriptionFileName());
  m_CaffeSegmentorPreferencesNode->Put(CaffePreferencesWidget::NETWORK_WEIGHTS_FILE_NAME, m_CaffeSegPrefs->GetNetworkWeightsFileName());
  m_CaffeSegmentorPreferencesNode->PutBool(CaffePreferencesWidget::DO_TRANSPOSE_NAME, m_CaffeSegPrefs->GetDoTranspose());
  m_CaffeSegmentorPreferencesNode->Put(CaffePreferencesWidget::INPUT_LAYER_NAME, m_CaffeSegPrefs->GetMemoryLayerName());
  m_CaffeSegmentorPreferencesNode->Put(CaffePreferencesWidget::OUTPUT_BLOB_NAME, m_CaffeSegPrefs->GetOutputBlobName());
  m_CaffeSegmentorPreferencesNode->PutInt(CaffePreferencesWidget::GPU_DEVICE_NAME, m_CaffeSegPrefs->GetGPUDevice());

  return true;
}


//-----------------------------------------------------------------------------
void CaffeSegmentorPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void CaffeSegmentorPreferencePage::Update()
{
  m_CaffeSegPrefs->SetNetworkDescriptionFileName(m_CaffeSegmentorPreferencesNode->Get(CaffePreferencesWidget::NETWORK_DESCRIPTION_FILE_NAME,
                                                                                      CaffePreferencesWidget::DEFAULT_NETWORK_DESCRIPTION_FILE));
  m_CaffeSegPrefs->SetNetworkWeightsFileName(m_CaffeSegmentorPreferencesNode->Get(CaffePreferencesWidget::NETWORK_WEIGHTS_FILE_NAME,
                                                                                  CaffePreferencesWidget::DEFAULT_NETWORK_WEIGHTS_FILE));
  m_CaffeSegPrefs->SetDoTranspose(m_CaffeSegmentorPreferencesNode->GetBool(CaffePreferencesWidget::DO_TRANSPOSE_NAME,
                                                                           CaffePreferencesWidget::DEFAULT_DO_TRANSPOSE));
  m_CaffeSegPrefs->SetMemoryLayerName(m_CaffeSegmentorPreferencesNode->Get(CaffePreferencesWidget::INPUT_LAYER_NAME,
                                                                           CaffePreferencesWidget::DEFAULT_INPUT_LAYER));
  m_CaffeSegPrefs->SetOutputBlobName(m_CaffeSegmentorPreferencesNode->Get(CaffePreferencesWidget::OUTPUT_BLOB_NAME,
                                                                          CaffePreferencesWidget::DEFAULT_OUTPUT_BLOB));
  m_CaffeSegPrefs->SetGPUDevice(m_CaffeSegmentorPreferencesNode->GetInt(CaffePreferencesWidget::GPU_DEVICE_NAME,
                                                                        CaffePreferencesWidget::DEFAULT_GPU_DEVICE));
}

} // end namespace
