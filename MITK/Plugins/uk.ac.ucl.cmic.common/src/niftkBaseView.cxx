/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseView.h"

#include <itkCommand.h>

#include <berryIWorkbenchPage.h>
#include <berryIWorkbenchWindow.h>
#include <berryQtSelectionProvider.h>

#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkSliceNavigationController.h>
#include <mitkWeakPointerProperty.h>
#include <QmitkRenderWindow.h>


namespace niftk
{

class BaseViewPrivate
{
public:

  BaseViewPrivate();
  ~BaseViewPrivate();

  void OnFocusedWindowDeleted()
  {
    m_Focused2DRenderer = 0;
  }

  QList<mitk::DataNode::Pointer> DataNodeSelectionToQList(mitk::DataNodeSelection::ConstPointer currentSelection) const;

  mitk::DataNodeSelection::ConstPointer QListToDataNodeSelection(const QList<mitk::DataNode::Pointer>& currentSelection) const;

  /// \brief Used to store the parent of this view, and should normally be set from within CreateQtPartControl().
  QWidget *m_Parent;

  /// \brief Stores the activation status.
  bool m_IsActivated;

  /// \brief Stores the visible status.
  bool m_IsVisible;

  /// \brief Used for the mitkFocusManager to register callbacks to track the currently focused window.
  unsigned long m_FocusManagerObserverTag;

  /// \brief Used to track the currently focused renderer.
  mitk::BaseRenderer* m_Focused2DRenderer;

  /// \brief Observer to get notified of the deletion of the focused renderer.
  unsigned long m_FocusedWindowDeletedObserverTag;
};


//-----------------------------------------------------------------------------
BaseViewPrivate::BaseViewPrivate()
{
  m_Parent = NULL;
  m_IsActivated = false;
  m_IsVisible = false;
  m_FocusManagerObserverTag = 0;
  m_Focused2DRenderer = NULL;
  m_FocusedWindowDeletedObserverTag = 0;
}


//-----------------------------------------------------------------------------
BaseViewPrivate::~BaseViewPrivate()
{
}


//-----------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> BaseViewPrivate::DataNodeSelectionToQList(mitk::DataNodeSelection::ConstPointer selection) const
{
  if (selection.IsNull())
  {
    return QList<mitk::DataNode::Pointer>();
  }
  return QList<mitk::DataNode::Pointer>::fromStdList(selection->GetSelectedDataNodes());
}


//-----------------------------------------------------------------------------
mitk::DataNodeSelection::ConstPointer BaseViewPrivate::QListToDataNodeSelection(const QList<mitk::DataNode::Pointer>& selectionList) const
{
  std::vector<mitk::DataNode::Pointer> selectionVector{selectionList.begin(), selectionList.end()};
  mitk::DataNodeSelection::ConstPointer selection(new mitk::DataNodeSelection(selectionVector));
  return selection;
}


//-----------------------------------------------------------------------------
BaseView::BaseView()
: QmitkAbstractView(),
  d_ptr(new BaseViewPrivate)
{
  Q_D(BaseView);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    itk::SimpleMemberCommand<BaseView>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<BaseView>::New();
    onFocusChangedCommand->SetCallbackFunction( this, &BaseView::OnFocusChanged );

    d->m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
  }
}


//-----------------------------------------------------------------------------
BaseView::~BaseView()
{
  Q_D(BaseView);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    focusManager->RemoveObserver(d->m_FocusManagerObserverTag);
  }
}


//-----------------------------------------------------------------------------
QWidget* BaseView::GetParent()
{
  Q_D(BaseView);
  return d->m_Parent;
}


//-----------------------------------------------------------------------------
void BaseView::SetParent(QWidget* parent)
{
  Q_D(BaseView);
  d->m_Parent = parent;
}


//-----------------------------------------------------------------------------
void BaseView::OnFocusChanged()
{
  Q_D(BaseView);

  if (d->m_Focused2DRenderer)
  {
    d->m_Focused2DRenderer->RemoveObserver(d->m_FocusedWindowDeletedObserverTag);
    d->m_Focused2DRenderer = 0;
  }

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    mitk::BaseRenderer* renderer = focusManager->GetFocused();
    if (renderer != NULL && renderer->GetMapperID() == mitk::BaseRenderer::Standard2D)
    {
      itk::SimpleMemberCommand<BaseViewPrivate>::Pointer command = itk::SimpleMemberCommand<BaseViewPrivate>::New();
      command->SetCallbackFunction(d, &BaseViewPrivate::OnFocusedWindowDeleted);
      d->m_FocusedWindowDeletedObserverTag = renderer->AddObserver(itk::DeleteEvent(), command);

      d->m_Focused2DRenderer = renderer;
    }
  }
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* BaseView::GetSliceNavigationController()
{
  mitk::SliceNavigationController::Pointer result = NULL;

  Q_D(BaseView);
  if (d->m_Focused2DRenderer != NULL)
  {
    result = d->m_Focused2DRenderer->GetSliceNavigationController();
  }
  if (result.IsNull())
  {
    mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
    if (renderWindowPart != NULL)
    {
      QmitkRenderWindow *renderWindow = renderWindowPart->GetActiveQmitkRenderWindow();
      if (renderWindow != NULL)
      {
        result = renderWindow->GetSliceNavigationController();
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
void BaseView::Activated()
{
  Q_D(BaseView);
  d->m_IsActivated = true;
}


//-----------------------------------------------------------------------------
void BaseView::Deactivated()
{
  Q_D(BaseView);
  d->m_IsActivated = false;
}


//-----------------------------------------------------------------------------
void BaseView::Visible()
{
  Q_D(BaseView);
  d->m_IsVisible = true;
}


//-----------------------------------------------------------------------------
void BaseView::Hidden()
{
  Q_D(BaseView);
  d->m_IsVisible = false;
}


//-----------------------------------------------------------------------------
bool BaseView::IsActivated()
{
  Q_D(BaseView);
  return d->m_IsActivated;
}


//-----------------------------------------------------------------------------
bool BaseView::IsVisible()
{
  Q_D(BaseView);
  return d->m_IsVisible;
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer BaseView::GetDataStorage() const
{
  return SuperClass::GetDataStorage();
}


//-----------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> BaseView::GetDataManagerSelection() const
{
  Q_D(const BaseView);

  berry::IWorkbenchPage::Pointer activePage = this->GetSite()->GetWorkbenchWindow()->GetActivePage();
  if (activePage.IsNull())
  {
    return QList<mitk::DataNode::Pointer>();
  }

  berry::IViewPart::Pointer dataManagerView = activePage->FindView("org.mitk.views.datamanager");
  if (dataManagerView.IsNull())
  {
    return QList<mitk::DataNode::Pointer>();
  }

  berry::QtSelectionProvider::Pointer selectionProvider = dataManagerView->GetSite()->GetSelectionProvider().Cast<berry::QtSelectionProvider>();

  mitk::DataNodeSelection::ConstPointer selection = selectionProvider->GetSelection().Cast<const mitk::DataNodeSelection>();
  return d->DataNodeSelectionToQList(selection);
}


//-----------------------------------------------------------------------------
void BaseView::SetDataManagerSelection(const QList<mitk::DataNode::Pointer>& selectedNodes) const
{
  Q_D(const BaseView);

  berry::IWorkbenchPage::Pointer activePage = this->GetSite()->GetWorkbenchWindow()->GetActivePage();
  if (activePage.IsNull())
  {
    return;
  }

  berry::IViewPart::Pointer dataManagerView = activePage->FindView("org.mitk.views.datamanager");
  if (dataManagerView.IsNull())
  {
    return;
  }

  berry::QtSelectionProvider::Pointer selectionProvider = dataManagerView->GetSite()->GetSelectionProvider().Cast<berry::QtSelectionProvider>();

  mitk::DataNodeSelection::ConstPointer selection = d->QListToDataNodeSelection(selectedNodes);
  selectionProvider->SetSelection(selection);

  /// We also set the same selection to the current view, so that it can notify
  /// other workbench parts about the selection change. This is necessary because
  /// the parts notify other parts about their selection change only if they
  /// (themselves) are active. Therefore, with the call above this function will
  /// not be called:
  ///
  ///     QmitkAbstractView::OnSelectionChanged(berry::IWorkbenchPart::Pointer source,
  ///         const QList<mitk::DataNode::Pointer>& selectedNodes)
  ///
  (const_cast<BaseView*>(this))->SetSelectedNodes(selectedNodes);
}


//-----------------------------------------------------------------------------
void BaseView::SetDataManagerSelection(mitk::DataNode::Pointer selectedNode) const
{
  QList<mitk::DataNode::Pointer> selectedNodes;
  selectedNodes << selectedNode;
  this->SetDataManagerSelection(selectedNodes);
}


//-----------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> BaseView::GetSelectedNodes() const
{
  Q_D(const BaseView);

  berry::QtSelectionProvider::Pointer selectionProvider = this->GetSite()->GetSelectionProvider().Cast<berry::QtSelectionProvider>();
  mitk::DataNodeSelection::ConstPointer selection = selectionProvider->GetSelection().Cast<const mitk::DataNodeSelection>();
  return d->DataNodeSelectionToQList(selection);
}


//-----------------------------------------------------------------------------
void BaseView::SetSelectedNodes(const QList<mitk::DataNode::Pointer>& selectedNodes)
{
  Q_D(BaseView);

  mitk::DataNodeSelection::ConstPointer selection = d->QListToDataNodeSelection(selectedNodes);
  this->GetSite()->GetSelectionProvider()->SetSelection(selection);
  this->FireNodesSelected(selectedNodes);
}


//-----------------------------------------------------------------------------
bool BaseView::IsActiveEditorCursorVisible() const
{
  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (!dataStorage)
  {
    return false;
  }

  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();

  mitk::BaseRenderer* mainAxialRenderer = renderWindowPart->GetQmitkRenderWindow("axial")->GetRenderer();
  mitk::BaseRenderer* mainSagittalRenderer = renderWindowPart->GetQmitkRenderWindow("sagittal")->GetRenderer();

  mitk::WeakPointerProperty::Pointer crossPlaneRendererProperty = mitk::WeakPointerProperty::New();

  mitk::NodePredicateAnd::Pointer crossPlanePredicate = mitk::NodePredicateAnd::New(
        mitk::NodePredicateDataType::New("PlaneGeometryData"),
        mitk::NodePredicateProperty::New("renderer", crossPlaneRendererProperty));

  crossPlaneRendererProperty->SetValue(mainAxialRenderer);
  mitk::DataNode* axialCrossPlaneNode = dataStorage->GetNode(crossPlanePredicate);
  if (!axialCrossPlaneNode)
  {
    return false;
  }

  bool isAxialPlaneNodeVisible = false;
  axialCrossPlaneNode->GetVisibility(isAxialPlaneNodeVisible, mainSagittalRenderer);

  return isAxialPlaneNodeVisible;
}


//-----------------------------------------------------------------------------
void BaseView::SetActiveEditorCursorVisible(bool visible) const
{
  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (!dataStorage)
  {
    return;
  }

  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();

  mitk::BaseRenderer* mainAxialRenderer = renderWindowPart->GetQmitkRenderWindow("axial")->GetRenderer();
  mitk::BaseRenderer* mainSagittalRenderer = renderWindowPart->GetQmitkRenderWindow("sagittal")->GetRenderer();
  mitk::BaseRenderer* mainCoronalRenderer = renderWindowPart->GetQmitkRenderWindow("coronal")->GetRenderer();

  mitk::WeakPointerProperty::Pointer crossPlaneRendererProperty = mitk::WeakPointerProperty::New();

  mitk::NodePredicateAnd::Pointer crossPlanePredicate = mitk::NodePredicateAnd::New(
        mitk::NodePredicateDataType::New("PlaneGeometryData"),
        mitk::NodePredicateProperty::New("renderer", crossPlaneRendererProperty));

  crossPlaneRendererProperty->SetValue(mainAxialRenderer);
  mitk::DataNode* axialCrossPlaneNode = dataStorage->GetNode(crossPlanePredicate);
  if (!axialCrossPlaneNode)
  {
    return;
  }

  crossPlaneRendererProperty->SetValue(mainSagittalRenderer);
  mitk::DataNode* sagittalCrossPlaneNode = dataStorage->GetNode(crossPlanePredicate);

  crossPlaneRendererProperty->SetValue(mainCoronalRenderer);
  mitk::DataNode* coronalCrossPlaneNode = dataStorage->GetNode(crossPlanePredicate);

  axialCrossPlaneNode->SetVisibility(visible, mainAxialRenderer);
  axialCrossPlaneNode->SetVisibility(visible, mainSagittalRenderer);
  axialCrossPlaneNode->SetVisibility(visible, mainCoronalRenderer);
  sagittalCrossPlaneNode->SetVisibility(visible, mainAxialRenderer);
  sagittalCrossPlaneNode->SetVisibility(visible, mainSagittalRenderer);
  sagittalCrossPlaneNode->SetVisibility(visible, mainCoronalRenderer);
  coronalCrossPlaneNode->SetVisibility(visible, mainAxialRenderer);
  coronalCrossPlaneNode->SetVisibility(visible, mainSagittalRenderer);
  coronalCrossPlaneNode->SetVisibility(visible, mainCoronalRenderer);

  mainAxialRenderer->RequestUpdate();
  mainSagittalRenderer->RequestUpdate();
  mainCoronalRenderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void BaseView::RequestRenderWindowUpdate(mitk::RenderingManager::RequestType requestType)
{
  SuperClass::RequestRenderWindowUpdate(requestType);
}


//-----------------------------------------------------------------------------
QHash<QString,QmitkRenderWindow*> BaseView::GetQmitkRenderWindows() const
{
  QHash<QString,QmitkRenderWindow*> renderWindows;

  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  if (renderWindowPart)
  {
    renderWindows = renderWindowPart->GetQmitkRenderWindows();
  }

  return renderWindows;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* BaseView::GetRenderWindow(QString id)
{
  QmitkRenderWindow* window = NULL;

  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  if (renderWindowPart != NULL)
  {
    window = renderWindowPart->GetQmitkRenderWindow(id);
  }

  return window;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* BaseView::GetSelectedRenderWindow() const
{
  QmitkRenderWindow* renderWindow = nullptr;

  if (mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart())
  {
    renderWindow = renderWindowPart->GetActiveQmitkRenderWindow();
  }

  return renderWindow;
}


//-----------------------------------------------------------------------------
void BaseView::SetViewToCoordinate(const mitk::Point3D &coordinate)
{
  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  if (renderWindowPart != NULL)
  {
    renderWindowPart->SetSelectedPosition(coordinate, "axial");
    renderWindowPart->SetSelectedPosition(coordinate, "sagittal");
    renderWindowPart->SetSelectedPosition(coordinate, "coronal");
    renderWindowPart->SetSelectedPosition(coordinate, "3d");
  }
}


//-----------------------------------------------------------------------------
void BaseView::FocusOnCurrentWindow() const
{
  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  if (renderWindowPart != NULL)
  {
    QmitkRenderWindow* window = renderWindowPart->GetActiveQmitkRenderWindow();
    if (window != NULL)
    {
      mitk::BaseRenderer* base = window->GetRenderer();
      mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();

      focusManager->SetFocused(base);
    }
  }
}


//-----------------------------------------------------------------------------
mitk::Point3D BaseView::GetSelectedPosition() const
{
  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  assert(renderWindowPart);
  return renderWindowPart->GetSelectedPosition();
}


//-----------------------------------------------------------------------------
void BaseView::SetSelectedPosition(const mitk::Point3D& selectedPosition)
{
  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  assert(renderWindowPart);
  renderWindowPart->SetSelectedPosition(selectedPosition);
}

}
