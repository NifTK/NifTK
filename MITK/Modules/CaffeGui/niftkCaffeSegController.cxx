/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegController.h"
#include <niftkCaffeManager.h>
#include <Internal/niftkCaffeSegGUI.h>
#include <itkFastMutexLock.h>

namespace niftk
{

class CaffeSegControllerPrivate
{
  Q_DECLARE_PUBLIC(CaffeSegController);
  CaffeSegController* const q_ptr;

public:

  CaffeSegControllerPrivate(CaffeSegController* q);
  ~CaffeSegControllerPrivate();

  CaffeSegGUI*                 m_GUI;
  std::string                  m_NetworkDescriptionFileName;
  std::string                  m_NetworkWeightsFileName;
  niftk::CaffeManager::Pointer m_LeftManager;
  mitk::DataNode*              m_LeftDataNode;
  niftk::CaffeManager::Pointer m_RightManager;
  mitk::DataNode*              m_RightDataNode;
  bool                         m_IsUpdatingManually;
  itk::FastMutexLock::Pointer  m_Mutex;
};


//-----------------------------------------------------------------------------
CaffeSegControllerPrivate::CaffeSegControllerPrivate(CaffeSegController* caffeSegController)
: q_ptr(caffeSegController)
, m_LeftManager(nullptr)
, m_LeftDataNode(nullptr)
, m_RightManager(nullptr)
, m_RightDataNode(nullptr)
, m_IsUpdatingManually(false)
, m_Mutex(itk::FastMutexLock::New())
{
  Q_Q(CaffeSegController);
}


//-----------------------------------------------------------------------------
CaffeSegControllerPrivate::~CaffeSegControllerPrivate()
{
}


//-----------------------------------------------------------------------------
CaffeSegController::CaffeSegController(IBaseView* view)
: BaseController(view)
, d_ptr(new CaffeSegControllerPrivate(this))
{
}


//-----------------------------------------------------------------------------
CaffeSegController::~CaffeSegController()
{

}


//-----------------------------------------------------------------------------
BaseGUI* CaffeSegController::CreateGUI(QWidget* parent)
{
  return new CaffeSegGUI(parent);
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetupGUI(QWidget* parent)
{
  Q_D(CaffeSegController);
  BaseController::SetupGUI(parent);
  d->m_GUI = dynamic_cast<CaffeSegGUI*>(this->GetGUI());
  d->m_GUI->SetDataStorage(this->GetDataStorage());
  connect(d->m_GUI, SIGNAL(OnLeftSelectionChanged(const mitk::DataNode*)), this, SLOT(OnLeftSelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(OnRightSelectionChanged(const mitk::DataNode*)), this, SLOT(OnRightSelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(OnDoItNowPressed()), this, SLOT(OnDoItNowPressed()));
  connect(d->m_GUI, SIGNAL(OnManualUpdateClicked(bool)), this, SLOT(OnManualUpdateClicked(bool)));
  connect(d->m_GUI, SIGNAL(OnAutomaticUpdateClicked(bool)), this, SLOT(OnAutomaticUpdateClicked(bool)));
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetNetworkDescriptionFileName(const std::string& description)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_NetworkDescriptionFileName = description;
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetNetworkWeightsFileName(const std::string& weights)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_NetworkWeightsFileName = weights;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnManualUpdateClicked(bool isChecked)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_IsUpdatingManually = isChecked;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnAutomaticUpdateClicked(bool isChecked)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_IsUpdatingManually = !isChecked;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnDoItNowPressed()
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (d->m_IsUpdatingManually)
  {
    this->InternalUpdate();
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::Update()
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (!d->m_IsUpdatingManually)
  {
    this->InternalUpdate();
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnLeftSelectionChanged(const mitk::DataNode* node)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (node != nullptr
      && d->m_LeftManager.IsNull()
      && !(d->m_NetworkDescriptionFileName.empty())
      && !(d->m_NetworkWeightsFileName.empty()))
  {
    d->m_LeftManager = niftk::CaffeManager::New(d->m_NetworkDescriptionFileName,
                                                d->m_NetworkWeightsFileName
                                               );
  }
  d->m_LeftDataNode = const_cast<mitk::DataNode*>(node);
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnRightSelectionChanged(const mitk::DataNode* node)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (node != nullptr
      && d->m_RightManager.IsNull()
      && !(d->m_NetworkDescriptionFileName.empty())
      && !(d->m_NetworkWeightsFileName.empty()))
  {
    d->m_RightManager = niftk::CaffeManager::New(d->m_NetworkDescriptionFileName,
                                                 d->m_NetworkWeightsFileName
                                                );
  }
  d->m_RightDataNode = const_cast<mitk::DataNode*>(node);
}


//-----------------------------------------------------------------------------
void CaffeSegController::InternalUpdate()
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  // We could parallelise this?

  if (   d->m_LeftManager.IsNotNull()
      && d->m_LeftDataNode != nullptr)
  {
    d->m_LeftManager->Segment(this->GetDataStorage(), d->m_LeftDataNode);
  }

  if (   d->m_RightManager.IsNotNull()
      && d->m_RightDataNode != nullptr)
  {
    d->m_RightManager->Segment(this->GetDataStorage(), d->m_RightDataNode);
  }
}

} // end namespace
