/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseController.h"

#include <mitkBaseRenderer.h>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <QmitkRenderWindow.h>

#include "niftkBaseGUI.h"
#include "niftkIBaseView.h"

namespace niftk
{

class BaseControllerPrivate
{
  Q_DECLARE_PUBLIC(BaseController);

  BaseController* const q_ptr;

  BaseGUI* m_GUI;

  niftkIBaseView* m_View;

public:

  BaseControllerPrivate(BaseController* q, niftkIBaseView* view);
  ~BaseControllerPrivate();

  void OnFocusedWindowDeleted()
  {
    m_Focused2DRenderer = 0;
  }

  void OnFocusChanged();

  /// \brief Used for the mitkFocusManager to register callbacks to track the currently focused window.
  unsigned long m_FocusManagerObserverTag;

  /// \brief Used to track the currently focused renderer.
  mitk::BaseRenderer* m_Focused2DRenderer;

  /// \brief Observer to get notified of the deletion of the focused renderer.
  unsigned long m_FocusedWindowDeletedObserverTag;

};

}

//-----------------------------------------------------------------------------
niftk::BaseControllerPrivate::BaseControllerPrivate(BaseController* baseController, niftkIBaseView* view)
  : q_ptr(baseController),
    m_GUI(nullptr),
    m_View(view),
    m_FocusManagerObserverTag(0),
    m_Focused2DRenderer(nullptr),
    m_FocusedWindowDeletedObserverTag(0)
{
  Q_Q(BaseController);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    itk::SimpleMemberCommand<BaseControllerPrivate>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<BaseControllerPrivate>::New();
    onFocusChangedCommand->SetCallbackFunction(this, &BaseControllerPrivate::OnFocusChanged);

    m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
  }
}


//-----------------------------------------------------------------------------
niftk::BaseControllerPrivate::~BaseControllerPrivate()
{
  Q_Q(BaseController);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }
}


//-----------------------------------------------------------------------------
void niftk::BaseControllerPrivate::OnFocusChanged()
{
  Q_Q(BaseController);

  if (m_Focused2DRenderer)
  {
    m_Focused2DRenderer->RemoveObserver(m_FocusedWindowDeletedObserverTag);
    m_Focused2DRenderer = 0;
  }

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    mitk::BaseRenderer* renderer = focusManager->GetFocused();
    if (renderer != NULL && renderer->GetMapperID() == mitk::BaseRenderer::Standard2D)
    {
      itk::SimpleMemberCommand<BaseControllerPrivate>::Pointer command = itk::SimpleMemberCommand<BaseControllerPrivate>::New();
      command->SetCallbackFunction(this, &BaseControllerPrivate::OnFocusedWindowDeleted);
      m_FocusedWindowDeletedObserverTag = renderer->AddObserver(itk::DeleteEvent(), command);

      m_Focused2DRenderer = renderer;
    }
  }

  q->OnFocusChanged();
}


//-----------------------------------------------------------------------------
niftk::BaseController::BaseController(niftkIBaseView* view)
  : d_ptr(new niftk::BaseControllerPrivate(this, view))
{
}


//-----------------------------------------------------------------------------
niftk::BaseController::~BaseController()
{
}


//-----------------------------------------------------------------------------
void niftk::BaseController::SetupGUI(QWidget* parent)
{
  Q_D(BaseController);
  d->m_GUI = this->CreateGUI(parent);
}


//-----------------------------------------------------------------------------
niftk::BaseGUI* niftk::BaseController::GetGUI() const
{
  Q_D(const BaseController);
  return d->m_GUI;
}


//-----------------------------------------------------------------------------
niftkIBaseView* niftk::BaseController::GetView() const
{
  Q_D(const BaseController);
  return d->m_View;
}


//-----------------------------------------------------------------------------
mitk::DataStorage* niftk::BaseController::GetDataStorage() const
{
  return this->GetView()->GetDataStorage();
}


//-----------------------------------------------------------------------------
void niftk::BaseController::RequestRenderWindowUpdate() const
{
  this->GetView()->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> niftk::BaseController::GetDataManagerSelection() const
{
  return this->GetView()->GetDataManagerSelection();
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* niftk::BaseController::GetSliceNavigationController() const
{
  return this->GetView()->GetSliceNavigationController();
}


//-----------------------------------------------------------------------------
void niftk::BaseController::OnFocusChanged()
{
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* niftk::BaseController::GetFocusedRenderer() const
{
  Q_D(const BaseController);
  return d->m_Focused2DRenderer;
}


//-----------------------------------------------------------------------------
niftk::ImageOrientation niftk::BaseController::GetImageOrientation()
{
  ImageOrientation orientation = IMAGE_ORIENTATION_UNKNOWN;
  const mitk::SliceNavigationController* sliceNavigationController = this->GetSliceNavigationController();
  if (sliceNavigationController != NULL)
  {
    mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

    if (viewDirection == mitk::SliceNavigationController::Axial)
    {
      orientation = IMAGE_ORIENTATION_AXIAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Sagittal)
    {
      orientation = IMAGE_ORIENTATION_SAGITTAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Frontal)
    {
      orientation = IMAGE_ORIENTATION_CORONAL;
    }
  }
  return orientation;
}
