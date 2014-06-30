/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkBaseView.h"
#include "internal/VisibilityChangedCommand.h"
#include <itkCommand.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>
#include <mitkSliceNavigationController.h>
#include <mitkWeakPointerProperty.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateAnd.h>
#include <QmitkRenderWindow.h>

class QmitkBaseViewPrivate
{
public:

  QmitkBaseViewPrivate();
  ~QmitkBaseViewPrivate();

  QMap<const mitk::DataNode*, unsigned long> visibilityObserverTags;

  mitk::MessageDelegate1<QmitkBaseView, const mitk::DataNode*>* addNodeEventListener;
  mitk::MessageDelegate1<QmitkBaseView, const mitk::DataNode*>* removeNodeEventListener;

  void OnFocusedWindowDeleted()
  {
    m_Focused2DRenderer = 0;
  }

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
QmitkBaseViewPrivate::QmitkBaseViewPrivate()
{
  m_Parent = NULL;
  m_IsActivated = false;
  m_IsVisible = false;
  m_FocusManagerObserverTag = 0;
  m_Focused2DRenderer = NULL;
  m_FocusedWindowDeletedObserverTag = 0;
}


//-----------------------------------------------------------------------------
QmitkBaseViewPrivate::~QmitkBaseViewPrivate()
{
}


//-----------------------------------------------------------------------------
QmitkBaseView::QmitkBaseView()
: QmitkAbstractView(),
  d_ptr(new QmitkBaseViewPrivate)
{
  Q_D(QmitkBaseView);

  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (dataStorage) {

    mitk::DataStorage::SetOfObjects::ConstPointer everyNode = dataStorage->GetAll();
    mitk::DataStorage::SetOfObjects::ConstIterator it = everyNode->Begin();
    mitk::DataStorage::SetOfObjects::ConstIterator end = everyNode->End();
    while (it != end)
    {
      this->onNodeAddedInternal(it->Value());
      ++it;
    }

    d->addNodeEventListener =
        new mitk::MessageDelegate1<QmitkBaseView, const mitk::DataNode*>(this, &QmitkBaseView::onNodeAddedInternal);
    dataStorage->AddNodeEvent.AddListener(*d->addNodeEventListener);

    d->removeNodeEventListener =
        new mitk::MessageDelegate1<QmitkBaseView, const mitk::DataNode*>(this, &QmitkBaseView::onNodeRemovedInternal);
    dataStorage->RemoveNodeEvent.AddListener(*d->removeNodeEventListener);
  }

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    itk::SimpleMemberCommand<QmitkBaseView>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<QmitkBaseView>::New();
    onFocusChangedCommand->SetCallbackFunction( this, &QmitkBaseView::OnFocusChanged );

    d->m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
  }
}


//-----------------------------------------------------------------------------
QmitkBaseView::~QmitkBaseView() {
  Q_D(QmitkBaseView);

  mitk::DataStorage* dataStorage = GetDataStorage();
  if (dataStorage)
  {
    dataStorage->AddNodeEvent.RemoveListener(*d->addNodeEventListener);
    dataStorage->RemoveNodeEvent.RemoveListener(*d->removeNodeEventListener);

    delete d->addNodeEventListener;
    delete d->removeNodeEventListener;
  }

  foreach (const mitk::DataNode* node, d->visibilityObserverTags.keys())
  {
    mitk::BaseProperty* property = node->GetProperty("visible");
    if (property)
    {
      property->RemoveObserver(d->visibilityObserverTags[node]);
    }
  }

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    focusManager->RemoveObserver(d->m_FocusManagerObserverTag);
  }
}


//-----------------------------------------------------------------------------
bool QmitkBaseView::IsExclusiveFunctionality() const
{
  return false;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::onNodeAddedInternal(const mitk::DataNode* node)
{
  Q_D(QmitkBaseView);
  mitk::BaseProperty* property = node->GetProperty("visible");
  if (property)
  {
    VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node);
    d->visibilityObserverTags[node] = property->AddObserver(itk::ModifiedEvent(), command);
  }
}


//-----------------------------------------------------------------------------
void QmitkBaseView::onNodeRemovedInternal(const mitk::DataNode* node)
{
  Q_D(QmitkBaseView);
  if (d->visibilityObserverTags.contains(node))
  {
    mitk::BaseProperty* property = node->GetProperty("visible");
    if (property) {
      property->RemoveObserver(d->visibilityObserverTags[node]);
    }
    d->visibilityObserverTags.remove(node);
  }
}


//-----------------------------------------------------------------------------
void QmitkBaseView::onVisibilityChanged(const mitk::DataNode* /*node*/)
{
}


//-----------------------------------------------------------------------------
QWidget* QmitkBaseView::GetParent()
{
  Q_D(QmitkBaseView);
  return d->m_Parent;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::SetParent(QWidget* parent)
{
  Q_D(QmitkBaseView);
  d->m_Parent = parent;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::OnFocusChanged()
{
  Q_D(QmitkBaseView);

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
      itk::SimpleMemberCommand<QmitkBaseViewPrivate>::Pointer command = itk::SimpleMemberCommand<QmitkBaseViewPrivate>::New();
      command->SetCallbackFunction(d, &QmitkBaseViewPrivate::OnFocusedWindowDeleted);
      d->m_FocusedWindowDeletedObserverTag = renderer->AddObserver(itk::DeleteEvent(), command);

      d->m_Focused2DRenderer = renderer;
    }
  }
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* QmitkBaseView::GetFocusedRenderer()
{
  Q_D(QmitkBaseView);
  return d->m_Focused2DRenderer;
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* QmitkBaseView::GetSliceNavigationController()
{
  return this->GetSliceNavigationControllerInternal();
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* QmitkBaseView::GetSliceNavigationControllerInternal()
{
  mitk::SliceNavigationController::Pointer result = NULL;

  Q_D(QmitkBaseView);
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
int QmitkBaseView::GetSliceNumberFromSliceNavigationController()
{
  int sliceNumber = -1;

  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController();
  if (snc.IsNotNull())
  {
    sliceNumber = snc->GetSlice()->GetPos();
  }
  return sliceNumber;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::Activated()
{
  Q_D(QmitkBaseView);
  d->m_IsActivated = true;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::Deactivated()
{
  Q_D(QmitkBaseView);
  d->m_IsActivated = false;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::Visible()
{
  Q_D(QmitkBaseView);
  d->m_IsVisible = true;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::Hidden()
{
  Q_D(QmitkBaseView);
  d->m_IsVisible = false;
}


//-----------------------------------------------------------------------------
bool QmitkBaseView::IsActivated()
{
  Q_D(QmitkBaseView);
  return d->m_IsActivated;
}


//-----------------------------------------------------------------------------
bool QmitkBaseView::IsVisible()
{
  Q_D(QmitkBaseView);
  return d->m_IsVisible;
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer QmitkBaseView::GetDataStorage() const
{
  return SuperClass::GetDataStorage();
}


//-----------------------------------------------------------------------------
bool QmitkBaseView::SetMainWindowCursorVisible(bool visible)
{
  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();

  mitk::BaseRenderer* mainAxialRenderer = renderWindowPart->GetQmitkRenderWindow("axial")->GetRenderer();
  mitk::BaseRenderer* mainSagittalRenderer = renderWindowPart->GetQmitkRenderWindow("sagittal")->GetRenderer();
  mitk::BaseRenderer* mainCoronalRenderer = renderWindowPart->GetQmitkRenderWindow("coronal")->GetRenderer();

  mitk::StringProperty::Pointer crossPlaneNameProperty = mitk::StringProperty::New();
  mitk::WeakPointerProperty::Pointer crossPlaneRendererProperty = mitk::WeakPointerProperty::New();

  mitk::NodePredicateAnd::Pointer crossPlanePredicate = mitk::NodePredicateAnd::New(
        mitk::NodePredicateProperty::New("name", crossPlaneNameProperty),
        mitk::NodePredicateProperty::New("renderer", crossPlaneRendererProperty));

  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (!dataStorage)
  {
    return false;
  }

  crossPlaneNameProperty->SetValue("widget1Plane");
  crossPlaneRendererProperty->SetValue(mainAxialRenderer);
  mitk::DataNode* axialCrossPlaneNode = dataStorage->GetNode(crossPlanePredicate);
  if (!axialCrossPlaneNode)
  {
    return false;
  }

  crossPlaneNameProperty->SetValue("widget2Plane");
  crossPlaneRendererProperty->SetValue(mainSagittalRenderer);
  mitk::DataNode* sagittalCrossPlaneNode = dataStorage->GetNode(crossPlanePredicate);

  crossPlaneNameProperty->SetValue("widget3Plane");
  crossPlaneRendererProperty->SetValue(mainCoronalRenderer);
  mitk::DataNode* coronalCrossPlaneNode = dataStorage->GetNode(crossPlanePredicate);

  bool wasVisible;
  axialCrossPlaneNode->GetVisibility(wasVisible, mainSagittalRenderer);

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

  return wasVisible;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::SetCurrentSelection(mitk::DataNode::Pointer dataNode)
{
  if (dataNode.IsNull())
  {
    return;
  }

  // Select the node in the data manager.
  mitk::DataNodeSelection::ConstPointer dataNodeSelection(new mitk::DataNodeSelection(dataNode));
  this->SetDataManagerSelection(dataNodeSelection);

  // Note that the data manager clears the "selected" property of the previously
  // selected nodes but it does not set it for the new selection. So we do it here.
  dataNode->SetSelected(true);

  // Notify the current view about the selection change.
  QList<mitk::DataNode::Pointer> dataNodeList;
  dataNodeList.push_back(dataNode);
  berry::IWorkbenchPart::Pointer nullPart;
  this->OnSelectionChanged(nullPart, dataNodeList);

  this->FireNodeSelected(dataNode);
}


//-----------------------------------------------------------------------------
void QmitkBaseView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  // Nothing to do. This function must be defined here because it is a private
  // virtual function in the base class, but it is called by SetCurrentSelection.
  // Derived classes should override this.
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkBaseView::GetRenderWindow(QString id)
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
QmitkRenderWindow* QmitkBaseView::GetSelectedRenderWindow()
{
  QmitkRenderWindow* renderWindow = 0;

  if (mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart())
  {
    renderWindow = renderWindowPart->GetActiveQmitkRenderWindow();
  }

  return renderWindow;
}


//-----------------------------------------------------------------------------
void QmitkBaseView::SetViewToCoordinate(const mitk::Point3D &coordinate)
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
void QmitkBaseView::FocusOnCurrentWindow()
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
