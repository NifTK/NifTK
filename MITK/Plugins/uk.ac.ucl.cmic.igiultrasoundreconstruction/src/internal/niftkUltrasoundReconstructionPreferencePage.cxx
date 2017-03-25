/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasoundReconstructionPreferencePage.h"
#include "niftkUltrasoundReconstructionView.h"
#include <QFormLayout>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>
#include <ctkDirectoryButton.h>

#include <niftkBaseView.h>

namespace niftk
{

//-----------------------------------------------------------------------------
UltrasoundReconstructionPreferencePage::UltrasoundReconstructionPreferencePage()
: m_MainControl(0)
, m_Initializing(false)
{

}


//-----------------------------------------------------------------------------
UltrasoundReconstructionPreferencePage::UltrasoundReconstructionPreferencePage(const UltrasoundReconstructionPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
UltrasoundReconstructionPreferencePage::~UltrasoundReconstructionPreferencePage()
{
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_UltrasoundReconstructionPreferencesNode = prefService->GetSystemPreferences()->Node(niftk::UltrasoundReconstructionView::VIEW_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;
  m_MainControl->setLayout(formLayout);

  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* UltrasoundReconstructionPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool UltrasoundReconstructionPreferencePage::PerformOk()
{
  return true;
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionPreferencePage::Update()
{
}

} // end namespace
