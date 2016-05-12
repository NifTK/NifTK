/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseController.h"

#include <QApplication>

#include <mitkBaseRenderer.h>
#include <mitkDataStorage.h>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <QmitkRenderWindow.h>

#include <mitkDataStorageListener.h>
#include <mitkDataNodePropertyListener.h>

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

  class DataStorageListener : private mitk::DataStorageListener
  {
  public:
    DataStorageListener(BaseControllerPrivate* d, mitk::DataStorage* dataStorage)
      : mitk::DataStorageListener(dataStorage),
        m_D(d)
    {
    }
  private:
    virtual void OnNodeAdded(mitk::DataNode* node) override
    {
      m_D->OnNodeAdded(node);
    }

    virtual void OnNodeChanged(mitk::DataNode* node) override
    {
      m_D->OnNodeChanged(node);
    }

    virtual void OnNodeRemoved(mitk::DataNode* node) override
    {
      m_D->OnNodeRemoved(node);
    }

    virtual void OnNodeDeleted(mitk::DataNode* node) override
    {
      m_D->OnNodeDeleted(node);
    }

    BaseControllerPrivate* m_D;
  };

  class VisibilityListener : private mitk::DataNodePropertyListener
  {
  public:
    VisibilityListener(BaseControllerPrivate* d, mitk::DataStorage* dataStorage)
      : mitk::DataNodePropertyListener(dataStorage, "visible"),
        m_D(d)
    {
    }
  private:
    virtual void OnPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer) override
    {
      m_D->OnNodeVisibilityChanged(node, renderer);
    }

    BaseControllerPrivate* m_D;
  };


  void OnFocusedWindowDeleted();

  void OnFocusChanged();

  void OnNodeAdded(mitk::DataNode* node);

  void OnNodeChanged(mitk::DataNode* node);

  void OnNodeRemoved(mitk::DataNode* node);

  void OnNodeDeleted(mitk::DataNode* node);

  void OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer);

public:

  BaseControllerPrivate(BaseController* q, niftkIBaseView* view);
  ~BaseControllerPrivate();

  /// \brief Used for the mitkFocusManager to register callbacks to track the currently focused window.
  unsigned long m_FocusManagerObserverTag;

  /// \brief Used to track the currently focused renderer.
  mitk::BaseRenderer* m_Focused2DRenderer;

  /// \brief Observer to get notified of the deletion of the focused renderer.
  unsigned long m_FocusedWindowDeletedObserverTag;

  DataStorageListener* m_DataStorageListener;

  VisibilityListener* m_VisibilityListener;

};

}

//-----------------------------------------------------------------------------
niftk::BaseControllerPrivate::BaseControllerPrivate(BaseController* baseController, niftkIBaseView* view)
  : q_ptr(baseController),
    m_GUI(nullptr),
    m_View(view),
    m_FocusManagerObserverTag(0),
    m_Focused2DRenderer(nullptr),
    m_FocusedWindowDeletedObserverTag(0),
    m_DataStorageListener(new DataStorageListener(this, view->GetDataStorage())),
    m_VisibilityListener(new VisibilityListener(this, view->GetDataStorage()))
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

  delete m_DataStorageListener;
  delete m_VisibilityListener;
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
void niftk::BaseControllerPrivate::OnFocusedWindowDeleted()
{
  m_Focused2DRenderer = 0;
}


//-----------------------------------------------------------------------------
void niftk::BaseControllerPrivate::OnNodeAdded(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeAdded(node);
}


//-----------------------------------------------------------------------------
void niftk::BaseControllerPrivate::OnNodeChanged(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeChanged(node);
}


//-----------------------------------------------------------------------------
void niftk::BaseControllerPrivate::OnNodeRemoved(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeRemoved(node);
}


//-----------------------------------------------------------------------------
void niftk::BaseControllerPrivate::OnNodeDeleted(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeDeleted(node);
}


//-----------------------------------------------------------------------------
void niftk::BaseControllerPrivate::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  Q_Q(BaseController);
  q->OnNodeVisibilityChanged(node, renderer);
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
mitk::BaseRenderer* niftk::BaseController::GetFocused2DRenderer() const
{
  Q_D(const BaseController);
  return d->m_Focused2DRenderer;
}


//-----------------------------------------------------------------------------
void niftk::BaseController::OnNodeAdded(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void niftk::BaseController::OnNodeChanged(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void niftk::BaseController::OnNodeRemoved(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void niftk::BaseController::OnNodeDeleted(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void niftk::BaseController::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
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


//-----------------------------------------------------------------------------
void niftk::BaseController::WaitCursorOn()
{
  QApplication::setOverrideCursor( QCursor(Qt::WaitCursor) );
}


//-----------------------------------------------------------------------------
void niftk::BaseController::WaitCursorOff()
{
  this->RestoreOverrideCursor();
}


//-----------------------------------------------------------------------------
void niftk::BaseController::BusyCursorOn()
{
  QApplication::setOverrideCursor( QCursor(Qt::BusyCursor) );
}


//-----------------------------------------------------------------------------
void niftk::BaseController::BusyCursorOff()
{
  this->RestoreOverrideCursor();
}


//-----------------------------------------------------------------------------
void niftk::BaseController::RestoreOverrideCursor()
{
  QApplication::restoreOverrideCursor();
}
