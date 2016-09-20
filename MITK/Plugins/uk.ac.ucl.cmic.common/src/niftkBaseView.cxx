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
BaseView::~BaseView() {
  Q_D(BaseView);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    focusManager->RemoveObserver(d->m_FocusManagerObserverTag);
  }
}


//-----------------------------------------------------------------------------
bool BaseView::IsExclusiveFunctionality() const
{
  return false;
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
  return SuperClass::GetDataManagerSelection();
}


//-----------------------------------------------------------------------------
void BaseView::FireNodeSelected(mitk::DataNode::Pointer node)
{
  SuperClass::FireNodeSelected(node);
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
void BaseView::SetCurrentSelection(mitk::DataNode::Pointer dataNode)
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
void BaseView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  // Nothing to do. This function must be defined here because it is a private
  // virtual function in the base class, but it is called by SetCurrentSelection.
  // Derived classes should override this.
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
