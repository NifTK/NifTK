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

#include <niftkDataStorageListener.h>
#include <niftkDataNodePropertyListener.h>

#include "niftkBaseGUI.h"
#include "niftkIBaseView.h"

namespace niftk
{

class BaseControllerPrivate
{
  Q_DECLARE_PUBLIC(BaseController)

  BaseController* const q_ptr;

  BaseGUI* m_GUI;

  IBaseView* m_View;

  class DataStorageListener : public niftk::DataStorageListener
  {
  public:
    mitkClassMacro(DataStorageListener, niftk::DataStorageListener)
    mitkNewMacro2Param(DataStorageListener, BaseControllerPrivate*, mitk::DataStorage*)

    DataStorageListener(BaseControllerPrivate* d, mitk::DataStorage* dataStorage)
      : niftk::DataStorageListener(dataStorage),
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

  class VisibilityListener : public niftk::DataNodePropertyListener
  {
  public:
    mitkClassMacro(VisibilityListener, niftk::DataStorageListener)
    mitkNewMacro2Param(VisibilityListener, BaseControllerPrivate*, mitk::DataStorage*)

    VisibilityListener(BaseControllerPrivate* d, mitk::DataStorage* dataStorage)
      : niftk::DataNodePropertyListener(dataStorage, "visible"),
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


  void OnFocusedRendererDeleted();

  void OnFocusChanged();

  void OnSelectedSliceChanged();

  void OnNodeAdded(mitk::DataNode* node);

  void OnNodeChanged(mitk::DataNode* node);

  void OnNodeRemoved(mitk::DataNode* node);

  void OnNodeDeleted(mitk::DataNode* node);

  void OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer);

public:

  BaseControllerPrivate(BaseController* q, IBaseView* view);
  ~BaseControllerPrivate();

  void SetupGUI();

  /// \brief Used to track the currently focused renderer.
  mitk::BaseRenderer* m_FocusedRenderer;

  /// \brief Keep track of this to SliceNavigationController register and unregister event listeners.
  mitk::SliceNavigationController* m_SliceNavigationController;

  /// \brief Observer to get notified of the slice change in the focused renderer.
  unsigned long m_SliceChangeObserverTag;

private:

  /// \brief Used for the mitkFocusManager to register callbacks to track the currently focused window.
  unsigned long m_FocusChangeObserverTag;

  /// \brief Observer to get notified of the deletion of the focused renderer.
  unsigned long m_FocusedRendererDeletionObserverTag;

  DataStorageListener::Pointer m_DataStorageListener;

  VisibilityListener::Pointer m_VisibilityListener;

};


//-----------------------------------------------------------------------------
BaseControllerPrivate::BaseControllerPrivate(BaseController* baseController, IBaseView* view)
  : q_ptr(baseController),
    m_GUI(nullptr),
    m_View(view),
    m_FocusChangeObserverTag(0ul),
    m_FocusedRenderer(nullptr),
    m_FocusedRendererDeletionObserverTag(0ul),
    m_SliceNavigationController(nullptr),
    m_SliceChangeObserverTag(0ul),
    m_DataStorageListener(DataStorageListener::New(this, view->GetDataStorage())),
    m_VisibilityListener(VisibilityListener::New(this, view->GetDataStorage()))
{
}


//-----------------------------------------------------------------------------
BaseControllerPrivate::~BaseControllerPrivate()
{
  Q_Q(BaseController);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  assert(focusManager);

  focusManager->RemoveObserver(m_FocusChangeObserverTag);
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::SetupGUI()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  assert(focusManager);

  itk::SimpleMemberCommand<BaseControllerPrivate>::Pointer onFocusChangedCommand =
    itk::SimpleMemberCommand<BaseControllerPrivate>::New();
  onFocusChangedCommand->SetCallbackFunction(this, &BaseControllerPrivate::OnFocusChanged);

  m_FocusChangeObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);

  this->OnFocusChanged();
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnFocusChanged()
{
  Q_Q(BaseController);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  assert(focusManager);

  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  if (focusedRenderer != m_FocusedRenderer)
  {
    if (m_FocusedRenderer)
    {
      m_FocusedRenderer->RemoveObserver(m_FocusedRendererDeletionObserverTag);
      m_FocusedRenderer = nullptr;
      m_FocusedRendererDeletionObserverTag = 0ul;
    }

    if (m_SliceNavigationController)
    {
      m_SliceNavigationController->RemoveObserver(m_SliceChangeObserverTag);
      m_SliceNavigationController = nullptr;
      m_SliceChangeObserverTag = 0ul;
    }

    assert(focusedRenderer);

    /// We are only interested in render windows of the active render window part.
    /// Auxiliary renderers in side views are ignored.
    QHash<QString, QmitkRenderWindow*> viewRenderWindows = q->GetView()->GetQmitkRenderWindows();
    bool focusedRendererIsInActiveEditor = false;
    for (auto viewRenderWindow: viewRenderWindows.values())
    {
      if (focusedRenderer == viewRenderWindow->GetRenderer())
      {
        focusedRendererIsInActiveEditor = true;
        break;
      }
    }
    if (focusedRendererIsInActiveEditor)
    {
      itk::SimpleMemberCommand<BaseControllerPrivate>::Pointer focusedRendererDeletedCommand = itk::SimpleMemberCommand<BaseControllerPrivate>::New();
      focusedRendererDeletedCommand->SetCallbackFunction(this, &BaseControllerPrivate::OnFocusedRendererDeleted);
      m_FocusedRendererDeletionObserverTag = focusedRenderer->AddObserver(itk::DeleteEvent(), focusedRendererDeletedCommand);

      m_FocusedRenderer = focusedRenderer;

      mitk::SliceNavigationController* sliceNavigationController = focusedRenderer->GetSliceNavigationController();
      assert(sliceNavigationController);

      if (focusedRenderer->GetMapperID() == mitk::BaseRenderer::Standard2D)
      {
        m_SliceNavigationController = sliceNavigationController;

        itk::SimpleMemberCommand<BaseControllerPrivate>::Pointer sliceChangedCommand = itk::SimpleMemberCommand<BaseControllerPrivate>::New();
        sliceChangedCommand->SetCallbackFunction(this, &BaseControllerPrivate::OnSelectedSliceChanged);
        m_SliceChangeObserverTag = sliceNavigationController->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), sliceChangedCommand);
      }

      q->OnFocusChanged();

      /// The renderer always has a slice navigation controller, but if it has not been initialised with a valid geometry,
      /// the slice navigation controller won't have a plane geometry.
      if (sliceNavigationController->GetCurrentPlaneGeometry())
      {
        ImageOrientation orientation = q->GetOrientation();
        int sliceIndex = q->GetSliceIndex();
        q->OnSelectedSliceChanged(orientation, sliceIndex);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnFocusedRendererDeleted()
{
  m_FocusedRenderer = nullptr;
  m_SliceNavigationController = nullptr;
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnSelectedSliceChanged()
{
  Q_Q(BaseController);
  ImageOrientation orientation = q->GetOrientation();
  int sliceIndex = q->GetSliceIndex();
  q->OnSelectedSliceChanged(orientation, sliceIndex);
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnNodeAdded(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeAdded(node);
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnNodeChanged(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeChanged(node);
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnNodeRemoved(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeRemoved(node);
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnNodeDeleted(mitk::DataNode* node)
{
  Q_Q(BaseController);
  q->OnNodeDeleted(node);
}


//-----------------------------------------------------------------------------
void BaseControllerPrivate::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  Q_Q(BaseController);
  q->OnNodeVisibilityChanged(node, renderer);
}


//-----------------------------------------------------------------------------
BaseController::BaseController(IBaseView* view)
  : d_ptr(new BaseControllerPrivate(this, view))
{
}


//-----------------------------------------------------------------------------
BaseController::~BaseController()
{
}


//-----------------------------------------------------------------------------
void BaseController::SetupGUI(QWidget* parent)
{
  Q_D(BaseController);
  d->m_GUI = this->CreateGUI(parent);
  d->SetupGUI();
}


//-----------------------------------------------------------------------------
BaseGUI* BaseController::GetGUI() const
{
  Q_D(const BaseController);
  return d->m_GUI;
}


//-----------------------------------------------------------------------------
IBaseView* BaseController::GetView() const
{
  Q_D(const BaseController);
  return d->m_View;
}


//-----------------------------------------------------------------------------
mitk::DataStorage* BaseController::GetDataStorage() const
{
  return this->GetView()->GetDataStorage();
}


//-----------------------------------------------------------------------------
void BaseController::RequestRenderWindowUpdate() const
{
  this->GetView()->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> BaseController::GetDataManagerSelection() const
{
  return this->GetView()->GetDataManagerSelection();
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* BaseController::GetSliceNavigationController() const
{
  Q_D(const BaseController);
  return d->m_SliceNavigationController;
}


//-----------------------------------------------------------------------------
void BaseController::OnFocusChanged()
{
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* BaseController::GetFocused2DRenderer() const
{
  Q_D(const BaseController);
  return d->m_FocusedRenderer;
}


//-----------------------------------------------------------------------------
void BaseController::OnViewGetsActivated()
{
}


//-----------------------------------------------------------------------------
void BaseController::OnViewGetsDeactivated()
{
}


//-----------------------------------------------------------------------------
void BaseController::OnViewGetsVisible()
{
}


//-----------------------------------------------------------------------------
void BaseController::OnViewGetsHidden()
{
  Q_D(const BaseController);
  if (d->m_SliceNavigationController)
  {
    d->m_SliceNavigationController->RemoveObserver(d->m_SliceChangeObserverTag);
  }
}


//-----------------------------------------------------------------------------
void BaseController::OnSelectedSliceChanged(ImageOrientation orientation, int sliceIndex)
{
}


//-----------------------------------------------------------------------------
void BaseController::OnNodeAdded(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void BaseController::OnNodeChanged(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void BaseController::OnNodeRemoved(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void BaseController::OnNodeDeleted(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void BaseController::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
}


//-----------------------------------------------------------------------------
ImageOrientation BaseController::GetOrientation() const
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
int BaseController::GetSliceIndex() const
{
  Q_D(const BaseController);

  int sliceIndex = -1;

  if (d->m_SliceNavigationController)
  {
    sliceIndex = d->m_SliceNavigationController->GetSlice()->GetPos();
  }

  return sliceIndex;
}


//-----------------------------------------------------------------------------
mitk::Point3D BaseController::GetSelectedPosition() const
{
  return this->GetView()->GetSelectedPosition();
}


//-----------------------------------------------------------------------------
void BaseController::WaitCursorOn()
{
  QApplication::setOverrideCursor( QCursor(Qt::WaitCursor) );
}


//-----------------------------------------------------------------------------
void BaseController::WaitCursorOff()
{
  this->RestoreOverrideCursor();
}


//-----------------------------------------------------------------------------
void BaseController::BusyCursorOn()
{
  QApplication::setOverrideCursor( QCursor(Qt::BusyCursor) );
}


//-----------------------------------------------------------------------------
void BaseController::BusyCursorOff()
{
  this->RestoreOverrideCursor();
}


//-----------------------------------------------------------------------------
void BaseController::RestoreOverrideCursor()
{
  QApplication::restoreOverrideCursor();
}

}
