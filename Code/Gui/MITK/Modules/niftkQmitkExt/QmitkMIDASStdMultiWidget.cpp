/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASStdMultiWidget.h"
#include "QmitkRenderWindow.h"
#include "vtkRenderWindow.h"
#include "vtkSmartPointer.h"
#include "vtkMatrix4x4.h"
#include "vtkLinearTransform.h"
#include <QStackedLayout>
#include <QGridLayout>
#include <QFrame>
#include "itkConversionUtils.h"
#include "itkMatrix.h"
#include "itkSpatialOrientationAdapter.h"
#include <cmath>
#include "mitkMIDASOrientationUtils.h"

/**
 * This class is to notify the SingleViewWidget about the display geometry changes of a render window.
 */
class DisplayGeometryModificationCommand : public itk::Command
{
public:
  mitkNewMacro3Param(DisplayGeometryModificationCommand, QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, mitk::DisplayGeometry*);

  DisplayGeometryModificationCommand(QmitkMIDASStdMultiWidget* stdMultiWidget, QmitkRenderWindow* renderWindow, mitk::DisplayGeometry* displayGeometry)
  : itk::Command()
  , m_StdMultiWidget(stdMultiWidget)
  , m_RenderWindow(renderWindow)
  , m_DisplayGeometry(displayGeometry)
  , m_LastScaleFactor(displayGeometry->GetScaleFactorMMPerDisplayUnit())
//  , m_LastOriginInMM(displayGeometry->GetOriginInMM())
  {
  }

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    double scaleFactor = m_DisplayGeometry->GetScaleFactorMMPerDisplayUnit();
    // Here we could distinguish the different kinds of geometry changes.
    if (scaleFactor != m_LastScaleFactor)
    {
      m_StdMultiWidget->OnScaleFactorChanged(m_RenderWindow);
      m_LastScaleFactor = scaleFactor;
    }

    // Not sure if we need this:
//    mitk::Vector2D originInMM = m_DisplayGeometry->GetOriginInMM();
//    if (originInMM != m_LastOriginInMM)
//    {
//      m_StdMultiWidget->OnOriginChanged(m_RenderWindow);
//      m_LastOriginInMM = originInMM;
//    }
  }

private:
  QmitkMIDASStdMultiWidget* const m_StdMultiWidget;
  QmitkRenderWindow* const m_RenderWindow;
  mitk::DisplayGeometry* const m_DisplayGeometry;
  double m_LastScaleFactor;
//  mitk::Vector2D m_LastOriginInMM;
};

QmitkMIDASStdMultiWidget::QmitkMIDASStdMultiWidget(
    QWidget* parent,
    Qt::WindowFlags f,
    mitk::RenderingManager* renderingManager,
    mitk::DataStorage* dataStorage
    )
: QmitkStdMultiWidget(parent, f, renderingManager)
, m_GridLayout(NULL)
, m_AxialSliceTag(0)
, m_SagittalSliceTag(0)
, m_CoronalSliceTag(0)
, m_IsSelected(false)
, m_IsEnabled(false)
, m_Display2DCursorsLocally(true)
, m_Display2DCursorsGlobally(false)
, m_Display3DViewInOrthoView(false)
, m_View(MIDAS_VIEW_ORTHO)
, m_MagnificationFactor(0.0)
, m_Geometry(NULL)
, m_BlockDisplayGeometryEvents(false)
{
  m_CreatedGeometries[0] = NULL;
  m_CreatedGeometries[1] = NULL;
  m_CreatedGeometries[2] = NULL;

  if (dataStorage != NULL)
  {
    this->SetDataStorage(dataStorage);
  }

  // We don't need these 4 lines if we pass in a widget specific RenderingManager.
  // If we are using a global one then we should use them to try and avoid Invalid Drawable errors on Mac.
  if (m_RenderingManager == mitk::RenderingManager::GetInstance())
  {
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget1->GetVtkRenderWindow());
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget2->GetVtkRenderWindow());
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget3->GetVtkRenderWindow());
    m_RenderingManager->RemoveRenderWindow(this->mitkWidget4->GetVtkRenderWindow());
  }

  // See also SetEnabled(bool) to see things that are dynamically on/off
  this->HideAllWidgetToolbars();
  this->DisableStandardLevelWindow();
  this->DisableDepartmentLogo();
  this->ActivateMenuWidget(false);
  this->SetBackgroundColor(QColor(0, 0, 0));

  // 3D planes should only be visible in this specific widget, not globally, so we create them, then make them globally invisible.
  this->AddDisplayPlaneSubTree();
  this->SetDisplay2DCursorsGlobally(false);
  this->SetDisplay2DCursorsLocally(false);
  this->SetWidgetPlanesLocked(false);
  this->SetWidgetPlanesRotationLocked(true);

  // Need each widget to react to Qt drag/drop events.
  this->mitkWidget1->setAcceptDrops(true);
  this->mitkWidget2->setAcceptDrops(true);
  this->mitkWidget3->setAcceptDrops(true);
  this->mitkWidget4->setAcceptDrops(true);

  // Create 4 cameras for temporary storage.
  for (unsigned int i = 0; i < 4; i++)
  {
    m_Cameras[i] = vtkCamera::New();
  }

  // Set these off, as it wont matter until there is an image dropped, with a specific layout and orientation.
  this->m_CornerAnnotaions[0].cornerText->SetText(0, "");
  this->m_CornerAnnotaions[1].cornerText->SetText(0, "");
  this->m_CornerAnnotaions[2].cornerText->SetText(0, "");

  // Set default layout. This must be ORTHO.
  this->SetMIDASView(MIDAS_VIEW_ORTHO, true);

  // Default to unselected, so borders are off.
  this->SetSelected(false);

  // Need each widget to signal when something is dropped, so connect signals to OnNodesDropped.
  connect(this->mitkWidget1, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(this->mitkWidget2, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(this->mitkWidget3, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(this->mitkWidget4, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));

  // Register to listen to SliceNavigators, slice changed events.
  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onAxialSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onAxialSliceChangedCommand->SetCallbackFunction( this, &QmitkMIDASStdMultiWidget::OnAxialSliceChanged );
  m_AxialSliceTag = mitkWidget1->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onAxialSliceChangedCommand);

  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onSagittalSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onSagittalSliceChangedCommand->SetCallbackFunction( this, &QmitkMIDASStdMultiWidget::OnSagittalSliceChanged );
  m_SagittalSliceTag = mitkWidget2->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSagittalSliceChangedCommand);

  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onCoronalSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onCoronalSliceChangedCommand->SetCallbackFunction( this, &QmitkMIDASStdMultiWidget::OnCoronalSliceChanged );
  m_CoronalSliceTag = mitkWidget3->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onCoronalSliceChangedCommand);

  // Listen to the display geometry changes so we raise an event when
  // the geometry changes through the display interactor (e.g. zooming with the mouse).
  std::vector<QmitkRenderWindow*> renderWindows = this->GetAllWindows();
  for (int i = 0; i < 3; ++i)
  {
    AddDisplayGeometryModificationObserver(renderWindows[i]);
  }
}

QmitkMIDASStdMultiWidget::~QmitkMIDASStdMultiWidget()
{
  if (mitkWidget1 != NULL && m_AxialSliceTag != 0)
  {
    mitkWidget1->GetSliceNavigationController()->RemoveObserver(m_AxialSliceTag);
  }
  if (mitkWidget2 != NULL && m_SagittalSliceTag != 0)
  {
    mitkWidget2->GetSliceNavigationController()->RemoveObserver(m_SagittalSliceTag);
  }
  if (mitkWidget3 != NULL && m_CoronalSliceTag != 0)
  {
    mitkWidget3->GetSliceNavigationController()->RemoveObserver(m_CoronalSliceTag);
  }

  for (unsigned int i = 0; i < 4; i++)
  {
    if (m_Cameras[i] != NULL)
    {
      m_Cameras[i]->Delete();
    }
  }

  // Stop listening to the display geometry changes so we raise an event when
  // the geometry changes through the display interactor (e.g. zooming with the mouse).
  std::vector<QmitkRenderWindow*> renderWindows = this->GetAllWindows();
  for (int i = 0; i < 3; ++i)
  {
    RemoveDisplayGeometryModificationObserver(renderWindows[i]);
  }
}

void QmitkMIDASStdMultiWidget::AddDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow)
{
  mitk::VtkPropRenderer* vtkPropRenderer = renderWindow->GetRenderer();
  assert(vtkPropRenderer);

  mitk::BaseRenderer* baseRenderer = dynamic_cast<mitk::BaseRenderer*>(vtkPropRenderer);
  assert(baseRenderer);

  mitk::DisplayGeometry* displayGeometry = baseRenderer->GetDisplayGeometry();
  assert(displayGeometry);

  DisplayGeometryModificationCommand::Pointer command = DisplayGeometryModificationCommand::New(this, renderWindow, displayGeometry);
  unsigned long observerTag = displayGeometry->AddObserver(itk::ModifiedEvent(), command);
  m_DisplayGeometryModificationObservers[renderWindow] = observerTag;
}

void QmitkMIDASStdMultiWidget::RemoveDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow)
{
  mitk::VtkPropRenderer* vtkPropRenderer = renderWindow->GetRenderer();
  assert(vtkPropRenderer);

  mitk::BaseRenderer* baseRenderer = dynamic_cast<mitk::BaseRenderer*>(vtkPropRenderer);
  assert(baseRenderer);

  mitk::DisplayGeometry* displayGeometry = baseRenderer->GetDisplayGeometry();
  assert(displayGeometry);

  displayGeometry->RemoveObserver(m_DisplayGeometryModificationObservers[renderWindow]);
  m_DisplayGeometryModificationObservers.erase(renderWindow);
}

void QmitkMIDASStdMultiWidget::OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  emit NodesDropped(this, window, nodes);
}

void QmitkMIDASStdMultiWidget::OnAxialSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->OnPositionChanged(MIDAS_ORIENTATION_AXIAL);
}

void QmitkMIDASStdMultiWidget::OnSagittalSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->OnPositionChanged(MIDAS_ORIENTATION_SAGITTAL);
}

void QmitkMIDASStdMultiWidget::OnCoronalSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->OnPositionChanged(MIDAS_ORIENTATION_CORONAL);
}

void QmitkMIDASStdMultiWidget::SetBackgroundColor(QColor colour)
{
  m_BackgroundColor = colour;
  m_GradientBackground1->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground1->Enable();
  m_GradientBackground2->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground2->Enable();
  m_GradientBackground3->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground3->Enable();
  m_GradientBackground4->SetGradientColors(colour.redF(),colour.greenF(),colour.blueF(),colour.redF(),colour.greenF(),colour.blueF());
  m_GradientBackground4->Enable();
}

QColor QmitkMIDASStdMultiWidget::GetBackgroundColor() const
{
  return m_BackgroundColor;
}

void QmitkMIDASStdMultiWidget::SetSelected(bool b)
{
  m_IsSelected = b;

  if (b)
  {
    this->EnableColoredRectangles();
  }
  else
  {
    this->DisableColoredRectangles();
  }
}

bool QmitkMIDASStdMultiWidget::IsSelected() const
{
  return m_IsSelected;
}

void QmitkMIDASStdMultiWidget::SetSelectedWindow(vtkRenderWindow* window)
{
  // When we "Select", the selection is at the level of the QmitkMIDASStdMultiWidget
  // so the whole of this widget is selected. However, we may have clicked in
  // a specific view, so it still helps to highlight the most recently clicked on view.
  // Also, if you are displaying orthoview then you actually have 4 windows present,
  // then highlighting them all starts to look a bit confusing, so we just highlight the
  // most recently focussed window, (eg. axial, sagittal, coronal or 3D).

  if (this->ContainsVtkRenderWindow(window))
  {
    m_IsSelected = true;

    if (window == this->GetRenderWindow1()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Enable(1.0, 0.0, 0.0);
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (window == this->GetRenderWindow2()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Enable(0.0, 1.0, 0.0);
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (window == this->GetRenderWindow3()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Enable(0.0, 0.0, 1.0);
      m_RectangleRendering4->Disable();
    }
    else if (window == this->GetRenderWindow4()->GetVtkRenderWindow())
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Enable(1.0, 1.0, 0.0);
    }
  }
  else
  {
    this->SetSelected(false);
  }
  this->ForceImmediateUpdate();
}

std::vector<QmitkRenderWindow*> QmitkMIDASStdMultiWidget::GetSelectedWindows() const
{

  std::vector<QmitkRenderWindow*> result;

  if (m_RectangleRendering1->IsEnabled())
  {
    result.push_back(this->GetRenderWindow1());
  }
  if (m_RectangleRendering2->IsEnabled())
  {
    result.push_back(this->GetRenderWindow2());
  }
  if (m_RectangleRendering3->IsEnabled())
  {
    result.push_back(this->GetRenderWindow3());
  }
  if (m_RectangleRendering4->IsEnabled())
  {
    result.push_back(this->GetRenderWindow4());
  }
  return result;
}

void QmitkMIDASStdMultiWidget::RequestUpdate()
{
  // The point of all this is to minimise the number of Updates.
  // So, ONLY call RequestUpdate on the specific window that is shown.

  if (this->isVisible())
  {
    switch(this->m_View)
    {
    case MIDAS_VIEW_AXIAL:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      break;
    case MIDAS_VIEW_SAGITTAL:
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      break;
    case MIDAS_VIEW_CORONAL:
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      break;
    case MIDAS_VIEW_ORTHO:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    case MIDAS_VIEW_3H:
    case MIDAS_VIEW_3V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      break;
    case MIDAS_VIEW_3D:
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    case MIDAS_VIEW_SAG_COR_H:
    case MIDAS_VIEW_SAG_COR_V:
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
    break;
    case MIDAS_VIEW_AX_COR_H:
    case MIDAS_VIEW_AX_COR_V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
    break;
    case MIDAS_VIEW_AX_SAG_H:
    case MIDAS_VIEW_AX_SAG_V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
    break;
    default:
      // die, this should never happen
      assert((m_View >= 0 && m_View <= 6) || (m_View >= 9 && m_View <= 14));
      break;
    }
  }
}

void QmitkMIDASStdMultiWidget::SetEnabled(bool b)
{
  // See also constructor for things that are ALWAYS on/off.
  if (b && !m_IsEnabled)
  {
    this->AddPlanesToDataStorage();
  }
  else if (!b && m_IsEnabled)
  {
    this->RemovePlanesFromDataStorage();
  }
  m_IsEnabled = b;
}

bool QmitkMIDASStdMultiWidget::IsEnabled() const
{
  return m_IsEnabled;
}

void QmitkMIDASStdMultiWidget::SetDisplay2DCursorsLocally(bool visible)
{
  // Here, "locally" means, for this widget, so we are setting Renderer Specific properties.
  m_Display2DCursorsLocally = visible;
  this->SetVisibility(mitkWidget1, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget1, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget1, m_PlaneNode3, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode3, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode3, visible);
}

bool QmitkMIDASStdMultiWidget::GetDisplay2DCursorsLocally() const
{
  return m_Display2DCursorsLocally;
}

void QmitkMIDASStdMultiWidget::SetDisplay2DCursorsGlobally(bool visible)
{
  // Here, "globally" means the plane nodes created within this widget will be available in ALL other render windows.
  m_Display2DCursorsGlobally = visible;
  m_PlaneNode1->SetVisibility(visible);
  m_PlaneNode1->Modified();
  m_PlaneNode2->SetVisibility(visible);
  m_PlaneNode2->Modified();
  m_PlaneNode3->SetVisibility(visible);
  m_PlaneNode3->Modified();
}

bool QmitkMIDASStdMultiWidget::GetDisplay2DCursorsGlobally() const
{
  return m_Display2DCursorsGlobally;
}

void QmitkMIDASStdMultiWidget::SetDisplay3DViewInOrthoView(bool visible)
{
  m_Display3DViewInOrthoView = visible;
  this->Update3DWindowVisibility();
}

bool QmitkMIDASStdMultiWidget::GetDisplay3DViewInOrthoView() const
{
  return m_Display3DViewInOrthoView;
}

void QmitkMIDASStdMultiWidget::Update3DWindowVisibility()
{
  if (this->m_DataStorage.IsNotNull())
  {
    vtkRenderWindow *axialVtkRenderWindow = this->mitkWidget1->GetVtkRenderWindow();
    mitk::BaseRenderer* axialRenderer = mitk::BaseRenderer::GetInstance(axialVtkRenderWindow);

    bool show3DPlanes = false;

    mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
    for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
    {
      if (it->Value().IsNull())
      {
        continue;
      }

      bool visibleIn3DWindow = false;
      if ((this->m_View == MIDAS_VIEW_ORTHO && this->m_Display3DViewInOrthoView)
          || this->m_View == MIDAS_VIEW_3D)
      {
        visibleIn3DWindow = true;
      }

      bool visibleInAxialView = false;
      if (it->Value()->GetBoolProperty("visible", visibleInAxialView, axialRenderer))
      {
        if (!visibleInAxialView)
        {
          visibleIn3DWindow = false;
        }
      }
      this->SetVisibility(this->mitkWidget4, it->Value(), visibleIn3DWindow);
      if (visibleIn3DWindow)
      {
        show3DPlanes = true;
      }
    }
    this->SetVisibility(this->mitkWidget4, m_PlaneNode1, show3DPlanes);
    this->SetVisibility(this->mitkWidget4, m_PlaneNode2, show3DPlanes);
    this->SetVisibility(this->mitkWidget4, m_PlaneNode3, show3DPlanes);
  }
}

void QmitkMIDASStdMultiWidget::SetVisibility(QmitkRenderWindow *window, mitk::DataNode *node, bool visible)
{
  if (window != NULL && node != NULL)
  {
    vtkRenderWindow *vtkRenderWindow = window->GetVtkRenderWindow();
    if (vtkRenderWindow != NULL)
    {
      mitk::BaseRenderer* renderer = mitk::BaseRenderer::GetInstance(vtkRenderWindow);
      if (renderer != NULL)
      {
        bool currentVisibility = false;
        node->GetVisibility(currentVisibility, renderer);

        if (visible != currentVisibility)
        {
          node->SetVisibility(visible, renderer);
        }
      }
    }
  }
}

void QmitkMIDASStdMultiWidget::SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible)
{
  for (unsigned int i = 0; i < nodes.size(); i++)
  {
    this->SetVisibility(mitkWidget1, nodes[i], visible);
    this->SetVisibility(mitkWidget2, nodes[i], visible);
    this->SetVisibility(mitkWidget3, nodes[i], visible);
  }
  this->Update3DWindowVisibility();
}


bool QmitkMIDASStdMultiWidget::ContainsWindow(QmitkRenderWindow *window) const
{
  bool result = false;
  if (   mitkWidget1 == window
      || mitkWidget2 == window
      || mitkWidget3 == window
      || mitkWidget4 == window
      )
  {
    result = true;
  }
  return result;
}

bool QmitkMIDASStdMultiWidget::ContainsVtkRenderWindow(vtkRenderWindow *window) const
{
  bool result = false;
  if (   mitkWidget1->GetVtkRenderWindow() == window
      || mitkWidget2->GetVtkRenderWindow() == window
      || mitkWidget3->GetVtkRenderWindow() == window
      || mitkWidget4->GetVtkRenderWindow() == window
      )
  {
    result = true;
  }
  return result;
}

std::vector<QmitkRenderWindow*> QmitkMIDASStdMultiWidget::GetAllWindows() const
{
  std::vector<QmitkRenderWindow*> result;
  result.push_back(this->GetRenderWindow1());
  result.push_back(this->GetRenderWindow2());
  result.push_back(this->GetRenderWindow3());
  result.push_back(this->GetRenderWindow4());
  return result;
}

std::vector<vtkRenderWindow*> QmitkMIDASStdMultiWidget::GetAllVtkWindows() const
{
  std::vector<vtkRenderWindow*> result;
  result.push_back(this->GetRenderWindow1()->GetVtkRenderWindow());
  result.push_back(this->GetRenderWindow2()->GetVtkRenderWindow());
  result.push_back(this->GetRenderWindow3()->GetVtkRenderWindow());
  result.push_back(this->GetRenderWindow4()->GetVtkRenderWindow());
  return result;
}

MIDASOrientation QmitkMIDASStdMultiWidget::GetOrientation()
{
  MIDASOrientation result = MIDAS_ORIENTATION_UNKNOWN;
  if (m_View == MIDAS_VIEW_AXIAL)
  {
    result = MIDAS_ORIENTATION_AXIAL;
  }
  else if (m_View == MIDAS_VIEW_SAGITTAL)
  {
    result = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (m_View == MIDAS_VIEW_CORONAL)
  {
    result = MIDAS_ORIENTATION_CORONAL;
  }
  else if (m_View == MIDAS_VIEW_ORTHO
           || m_View == MIDAS_VIEW_3H
           || m_View == MIDAS_VIEW_3V
           || m_View == MIDAS_VIEW_SAG_COR_H
           || m_View == MIDAS_VIEW_SAG_COR_V
           || m_View == MIDAS_VIEW_AX_COR_H
           || m_View == MIDAS_VIEW_AX_COR_V
           || m_View == MIDAS_VIEW_AX_SAG_H
           || m_View == MIDAS_VIEW_AX_SAG_V
           )
  {
    if (m_RectangleRendering1->IsEnabled())
    {
      result = MIDAS_ORIENTATION_AXIAL;
    }
    else if (m_RectangleRendering2->IsEnabled())
    {
      result = MIDAS_ORIENTATION_SAGITTAL;
    }
    else if (m_RectangleRendering3->IsEnabled())
    {
      result = MIDAS_ORIENTATION_CORONAL;
    }
  }
  return result;
}

void QmitkMIDASStdMultiWidget::FitToDisplay()
{
  std::vector<vtkRenderWindow*> vtkWindows = this->GetAllVtkWindows();
  for (unsigned int window = 0; window < vtkWindows.size(); window++)
  {
    mitk::BaseRenderer *baseRenderer = mitk::BaseRenderer::GetInstance(vtkWindows[window]);
    baseRenderer->GetDisplayGeometry()->Fit();
  }
}

void QmitkMIDASStdMultiWidget::SetMIDASView(MIDASView view, mitk::Geometry3D *geometry)
{
  this->SetGeometry(geometry);
  this->SetMIDASView(view, false);
}

void QmitkMIDASStdMultiWidget::SetGeometry(mitk::Geometry3D *geometry)
{
  if (geometry != NULL)
  {
    m_Geometry = geometry;

    // Add these annotations the first time we have a real geometry.
    this->m_CornerAnnotaions[0].cornerText->SetText(0, "Axial");
    this->m_CornerAnnotaions[1].cornerText->SetText(0, "Sagittal");
    this->m_CornerAnnotaions[2].cornerText->SetText(0, "Coronal");

    // If m_RenderingManager is a local rendering manager
    // not the global singleton instance, then we never have to worry about this.
    if (m_RenderingManager == mitk::RenderingManager::GetInstance())
    {
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow1()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow2()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow3()->GetVtkRenderWindow());
      m_RenderingManager->AddRenderWindow(this->GetRenderWindow4()->GetVtkRenderWindow());
    }

    // Inspired by:
    // http://www.na-mic.org/Wiki/index.php/Coordinate_System_Conversion_Between_ITK_and_Slicer3

    mitk::AffineTransform3D::Pointer affineTransform = geometry->GetIndexToWorldTransform();
    itk::Matrix<float, 3, 3> affineTransformMatrix = affineTransform->GetMatrix();
    mitk::AffineTransform3D::MatrixType::InternalMatrixType normalisedAffineTransformMatrix;
    for (unsigned int i=0; i < 3; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        normalisedAffineTransformMatrix[i][j] = affineTransformMatrix[i][j];
      }
    }
    normalisedAffineTransformMatrix.normalize_columns();
    for (unsigned int i=0; i < 3; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        affineTransformMatrix[i][j] = normalisedAffineTransformMatrix[i][j];
      }
    }

    mitk::AffineTransform3D::MatrixType::InternalMatrixType inverseTransformMatrix = affineTransformMatrix.GetInverse();

    int dominantAxisRL = itk::Function::Max3(inverseTransformMatrix[0][0],inverseTransformMatrix[1][0],inverseTransformMatrix[2][0]);
    int signRL = itk::Function::Sign(inverseTransformMatrix[dominantAxisRL][0]);
    int dominantAxisAP = itk::Function::Max3(inverseTransformMatrix[0][1],inverseTransformMatrix[1][1],inverseTransformMatrix[2][1]);
    int signAP = itk::Function::Sign(inverseTransformMatrix[dominantAxisAP][1]);
    int dominantAxisSI = itk::Function::Max3(inverseTransformMatrix[0][2],inverseTransformMatrix[1][2],inverseTransformMatrix[2][2]);
    int signSI = itk::Function::Sign(inverseTransformMatrix[dominantAxisSI][2]);

    int permutedBoundingBox[3];
    int permutedAxes[3];
    int flippedAxes[3];
    double permutedSpacing[3];

    permutedAxes[0] = dominantAxisRL;
    permutedAxes[1] = dominantAxisAP;
    permutedAxes[2] = dominantAxisSI;

    flippedAxes[0] = signRL;
    flippedAxes[1] = signAP;
    flippedAxes[2] = signSI;

    permutedBoundingBox[0] = geometry->GetExtent(dominantAxisRL);
    permutedBoundingBox[1] = geometry->GetExtent(dominantAxisAP);
    permutedBoundingBox[2] = geometry->GetExtent(dominantAxisSI);

    permutedSpacing[0] = geometry->GetSpacing()[permutedAxes[0]];
    permutedSpacing[1] = geometry->GetSpacing()[permutedAxes[1]];
    permutedSpacing[2] = geometry->GetSpacing()[permutedAxes[2]];

    mitk::AffineTransform3D::MatrixType::InternalMatrixType permutedMatrix;
    permutedMatrix.set_column(0, inverseTransformMatrix.get_row(permutedAxes[0]) * flippedAxes[0]);
    permutedMatrix.set_column(1, inverseTransformMatrix.get_row(permutedAxes[1]) * flippedAxes[1]);
    permutedMatrix.set_column(2, inverseTransformMatrix.get_row(permutedAxes[2]) * flippedAxes[2]);

    m_OrientationToAxisMap.clear();
    m_OrientationToAxisMap.insert(std::pair<MIDASOrientation, int>(MIDAS_ORIENTATION_AXIAL,    dominantAxisSI));
    m_OrientationToAxisMap.insert(std::pair<MIDASOrientation, int>(MIDAS_ORIENTATION_SAGITTAL, dominantAxisRL));
    m_OrientationToAxisMap.insert(std::pair<MIDASOrientation, int>(MIDAS_ORIENTATION_CORONAL,  dominantAxisAP));

    MITK_DEBUG << "Matt, extent=" << geometry->GetExtent(0) << ", " << geometry->GetExtent(1) << ", " << geometry->GetExtent(2) << std::endl;
    MITK_DEBUG << "Matt, domRL=" << dominantAxisRL << ", signRL=" << signRL << ", domAP=" << dominantAxisAP << ", signAP=" << signAP << ", dominantAxisSI=" << dominantAxisSI << ", signSI=" << signSI << std::endl;
    MITK_DEBUG << "Matt, permutedBoundingBox=" << permutedBoundingBox[0] << ", " << permutedBoundingBox[1] << ", " << permutedBoundingBox[2] << std::endl;
    MITK_DEBUG << "Matt, permutedAxes=" << permutedAxes[0] << ", " << permutedAxes[1] << ", " << permutedAxes[2] << std::endl;
    MITK_DEBUG << "Matt, permutedSpacing=" << permutedSpacing[0] << ", " << permutedSpacing[1] << ", " << permutedSpacing[2] << std::endl;
    MITK_DEBUG << "Matt, flippedAxes=" << flippedAxes[0] << ", " << flippedAxes[1] << ", " << flippedAxes[2] << std::endl;

    MITK_DEBUG << "Matt, input normalised matrix=" << std::endl;

    for (unsigned int i=0; i < 3; i++)
    {
      MITK_DEBUG << affineTransformMatrix[i][0] << " " << affineTransformMatrix[i][1] << " " << affineTransformMatrix[i][2];
    }

    MITK_DEBUG << "Matt, inverse normalised matrix=" << std::endl;

    for (unsigned int i=0; i < 3; i++)
    {
      MITK_DEBUG << inverseTransformMatrix[i][0] << " " << inverseTransformMatrix[i][1] << " " << inverseTransformMatrix[i][2];
    }

    MITK_DEBUG << "Matt, permuted matrix=" << std::endl;

    for (unsigned int i=0; i < 3; i++)
    {
      MITK_DEBUG << permutedMatrix[i][0] << " " << permutedMatrix[i][1] << " " << permutedMatrix[i][2];
    }

    std::vector<vtkRenderWindow*> vtkWindows = this->GetAllVtkWindows();
    for (unsigned int window = 0; window < vtkWindows.size(); window++)
    {
      vtkRenderWindow* vtkWindow = vtkWindows[window];
      mitk::BaseRenderer *baseRenderer = mitk::BaseRenderer::GetInstance(vtkWindow);
      int id = baseRenderer->GetMapperID();

      // Get access to slice navigation controller, as this sorts out most of the process.
      mitk::SliceNavigationController *sliceNavigationController = baseRenderer->GetSliceNavigationController();
      sliceNavigationController->SetViewDirectionToDefault();

      // Get the view/orientation flags.
      mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

      if (window < 3)
      {

        mitk::Point3D    originVoxels;
        mitk::Point3D    originMillimetres;
        mitk::Point3D    originOfSlice;
        mitk::VnlVector  rightDV(3);
        mitk::VnlVector  bottomDV(3);
        mitk::VnlVector  normal(3);
        int              width = 1;
        int              height = 1;
        mitk::ScalarType viewSpacing = 1;
        unsigned int     slices = 1;
        bool             isFlipped;

        float voxelOffset = 0;
        float boundingBoxOffset = 0;

        if(geometry->GetImageGeometry())
        {
          voxelOffset = 0.5;
          boundingBoxOffset = 0.5;
        }

        originVoxels[0] = 0;
        originVoxels[1] = 0;
        originVoxels[2] = 0;

        if (flippedAxes[0] < 0)
        {
          MITK_DEBUG << "Matt, flippedAxis[0] < 0, so flipping axis " << permutedAxes[0] << std::endl;
          originVoxels[permutedAxes[0]] = geometry->GetExtent(permutedAxes[0]) - 1;
        }
        if (flippedAxes[1] < 0)
        {
          MITK_DEBUG << "Matt, flippedAxis[1] < 0, so flipping axis " << permutedAxes[1] << std::endl;
          originVoxels[permutedAxes[1]] = geometry->GetExtent(permutedAxes[1]) - 1;
        }
        if (flippedAxes[2] < 0)
        {
          MITK_DEBUG << "Matt, flippedAxis[2] < 0, so flipping axis " << permutedAxes[2] << std::endl;
          originVoxels[permutedAxes[2]] = geometry->GetExtent(permutedAxes[2]) - 1;
        }

        geometry->IndexToWorld(originVoxels, originMillimetres);

        MITK_DEBUG << "Matt, originVoxels=" << originVoxels << ", originMillimetres=" << originMillimetres << std::endl;

        // Setting up the width, height, axis orientation.
        switch(viewDirection)
        {
        case mitk::SliceNavigationController::Sagittal:
          width  = permutedBoundingBox[1];
          height = permutedBoundingBox[2];
          originOfSlice[0] = originMillimetres[0];
          originOfSlice[1] = originMillimetres[1] - voxelOffset*permutedSpacing[1];
          originOfSlice[2] = originMillimetres[2] - voxelOffset*permutedSpacing[2];
          rightDV[0] = permutedSpacing[0]*permutedMatrix[0][1];
          rightDV[1] = permutedSpacing[1]*permutedMatrix[1][1];
          rightDV[2] = permutedSpacing[2]*permutedMatrix[2][1];
          bottomDV[0] = permutedSpacing[0]*permutedMatrix[0][2];
          bottomDV[1] = permutedSpacing[1]*permutedMatrix[1][2];
          bottomDV[2] = permutedSpacing[2]*permutedMatrix[2][2];
          normal[0] = permutedMatrix[0][0];
          normal[1] = permutedMatrix[1][0];
          normal[2] = permutedMatrix[2][0];
          viewSpacing = permutedSpacing[0];
          slices = permutedBoundingBox[0];
          isFlipped = false;
          break;
        case mitk::SliceNavigationController::Frontal:
          width  = permutedBoundingBox[0];
          height = permutedBoundingBox[2];
          originOfSlice[0] = originMillimetres[0] - voxelOffset*permutedSpacing[0];
          originOfSlice[1] = originMillimetres[1];
          originOfSlice[2] = originMillimetres[2] - voxelOffset*permutedSpacing[2];
          rightDV[0] = permutedSpacing[0]*permutedMatrix[0][0];
          rightDV[1] = permutedSpacing[1]*permutedMatrix[1][0];
          rightDV[2] = permutedSpacing[2]*permutedMatrix[2][0];
          bottomDV[0] = permutedSpacing[0]*permutedMatrix[0][2];
          bottomDV[1] = permutedSpacing[1]*permutedMatrix[1][2];
          bottomDV[2] = permutedSpacing[2]*permutedMatrix[2][2];
          normal[0] = permutedMatrix[0][1];
          normal[1] = permutedMatrix[1][1];
          normal[2] = permutedMatrix[2][1];
          viewSpacing = permutedSpacing[1];
          slices = permutedBoundingBox[1];
          isFlipped = true;
          break;
        default:
          // Default = Transverse.
          width  = permutedBoundingBox[0];
          height = permutedBoundingBox[1];
          originOfSlice[0] = originMillimetres[0] + permutedBoundingBox[0]*permutedSpacing[0]*permutedMatrix[0][1] - voxelOffset*permutedSpacing[0];
          originOfSlice[1] = originMillimetres[1] + permutedBoundingBox[1]*permutedSpacing[1]*permutedMatrix[1][1] + voxelOffset*permutedSpacing[1];
          originOfSlice[2] = originMillimetres[2] + permutedBoundingBox[2]*permutedSpacing[2]*permutedMatrix[2][1];
          rightDV[0] = permutedSpacing[0]*permutedMatrix[0][0];
          rightDV[1] = permutedSpacing[1]*permutedMatrix[1][0];
          rightDV[2] = permutedSpacing[2]*permutedMatrix[2][0];
          bottomDV[0] = -1.0 * permutedSpacing[0]*permutedMatrix[0][1];
          bottomDV[1] = -1.0 * permutedSpacing[1]*permutedMatrix[1][1];
          bottomDV[2] = -1.0 * permutedSpacing[2]*permutedMatrix[2][1];
          normal[0] = permutedMatrix[0][2];
          normal[1] = permutedMatrix[1][2];
          normal[2] = permutedMatrix[2][2];
          viewSpacing = permutedSpacing[2];
          slices = permutedBoundingBox[2];
          isFlipped = true;
          break;
        } // end switch

        MITK_DEBUG << "Matt, image=" << geometry->GetImageGeometry() << std::endl;
        MITK_DEBUG << "Matt, width=" << width << std::endl;
        MITK_DEBUG << "Matt, height=" << height << std::endl;
        MITK_DEBUG << "Matt, originOfSlice=" << originOfSlice << std::endl;
        MITK_DEBUG << "Matt, rightDV=" << rightDV << std::endl;
        MITK_DEBUG << "Matt, bottomDV=" << bottomDV << std::endl;
        MITK_DEBUG << "Matt, normal=" << normal << std::endl;
        MITK_DEBUG << "Matt, viewSpacing=" << viewSpacing << std::endl;
        MITK_DEBUG << "Matt, slices=" << slices << std::endl;
        MITK_DEBUG << "Matt, isFlipped=" << isFlipped << std::endl;

        unsigned int numberOfTimeSteps = 0;
        mitk::TimeSlicedGeometry::Pointer inputTimeSlicedGeometry = static_cast<mitk::TimeSlicedGeometry*>(geometry);
        if (inputTimeSlicedGeometry.IsNotNull())
        {
          numberOfTimeSteps = inputTimeSlicedGeometry->GetTimeSteps();
        }
        else
        {
          numberOfTimeSteps = 1;
        }

        mitk::TimeSlicedGeometry::Pointer createdTimeSlicedGeometry = mitk::TimeSlicedGeometry::New();
        createdTimeSlicedGeometry->InitializeEmpty(numberOfTimeSteps);
        createdTimeSlicedGeometry->SetImageGeometry(false);
        createdTimeSlicedGeometry->SetEvenlyTimed(true);

        if (inputTimeSlicedGeometry.IsNotNull())
        {
          createdTimeSlicedGeometry->SetEvenlyTimed(inputTimeSlicedGeometry->GetEvenlyTimed());
          createdTimeSlicedGeometry->SetTimeBounds(inputTimeSlicedGeometry->GetTimeBounds());
          createdTimeSlicedGeometry->SetBounds(inputTimeSlicedGeometry->GetBounds());
        }

        // For the PlaneGeometry.
        mitk::ScalarType bounds[6]= { 0, width, 0, height, 0, 1 };

        // A SlicedGeometry3D is initialised from a 2D PlaneGeometry, plus the number of slices.
        mitk::PlaneGeometry::Pointer planeGeometry = mitk::PlaneGeometry::New();
        planeGeometry->SetIdentity();
        planeGeometry->SetImageGeometry(false);
        planeGeometry->SetBounds(bounds);
        planeGeometry->SetOrigin(originOfSlice);
        planeGeometry->SetMatrixByVectors(rightDV, bottomDV, normal.two_norm());

        for (unsigned int i = 0; i < numberOfTimeSteps; i++)
        {
          // Then we create the SlicedGeometry3D from an initial plane, and a given number of slices.
          mitk::SlicedGeometry3D::Pointer slicedGeometry = mitk::SlicedGeometry3D::New();
          slicedGeometry->SetIdentity();
          slicedGeometry->SetReferenceGeometry(geometry);
          slicedGeometry->SetImageGeometry(false);
          slicedGeometry->InitializeEvenlySpaced(planeGeometry, viewSpacing, slices, isFlipped );

          if (inputTimeSlicedGeometry.IsNotNull())
          {
            slicedGeometry->SetTimeBounds(inputTimeSlicedGeometry->GetGeometry3D(i)->GetTimeBounds());
          }
          createdTimeSlicedGeometry->SetGeometry3D(slicedGeometry, i);
        }
        createdTimeSlicedGeometry->UpdateInformation();

        MITK_DEBUG << "Matt - final geometry=" << createdTimeSlicedGeometry << std::endl;
        MITK_DEBUG << "Matt - final geometry origin=" << createdTimeSlicedGeometry->GetOrigin() << std::endl;
        MITK_DEBUG << "Matt - final geometry center=" << createdTimeSlicedGeometry->GetCenter() << std::endl;
        for(int i = 0; i < 8; i++)
        {
          MITK_DEBUG << "Matt - final geometry i=" << i << ", p=" << createdTimeSlicedGeometry->GetCornerPoint(i) << std::endl;
        }
        m_CreatedGeometries[window] = createdTimeSlicedGeometry;
        sliceNavigationController->SetInputWorldGeometry(createdTimeSlicedGeometry);
        sliceNavigationController->Update(mitk::SliceNavigationController::Original, true, true, false);
        sliceNavigationController->SetViewDirection(viewDirection);

        // For 2D mappers only, set to middle slice (the 3D mapper simply follows by event listening).
        if ( id == 1 )
        {
          // Now geometry is established, set to middle slice.
          int sliceNumber = (int)((double)(sliceNavigationController->GetSlice()->GetSteps() - 1) / 2.0);
          sliceNavigationController->GetSlice()->SetPos(sliceNumber);
        }

        // Now geometry is established, get the display geometry to fit the picture to the window.
        baseRenderer->GetDisplayGeometry()->SetConstrainZoomingAndPanning(false);
        baseRenderer->GetDisplayGeometry()->Fit();

      } // if window < 3
    }
  }
}

void QmitkMIDASStdMultiWidget::SetMIDASView(MIDASView view, bool rebuildLayout)
{
  if (rebuildLayout)
  {
    if (m_GridLayout != NULL)
    {
      delete m_GridLayout;
    }
    if (QmitkStdMultiWidgetLayout != NULL)
    {
      delete QmitkStdMultiWidgetLayout;
    }

    m_GridLayout = new QGridLayout();
    m_GridLayout->setContentsMargins(0, 0, 0, 0);
    m_GridLayout->setSpacing(0);

    QmitkStdMultiWidgetLayout = new QHBoxLayout( this );
    QmitkStdMultiWidgetLayout->setContentsMargins(0, 0, 0, 0);
    QmitkStdMultiWidgetLayout->setSpacing(0);

    if (view == MIDAS_VIEW_3H)
    {
      m_GridLayout->addWidget(this->mitkWidget1Container, 0, 0);
      m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);
      m_GridLayout->addWidget(this->mitkWidget3Container, 0, 2);
      m_GridLayout->addWidget(this->mitkWidget4Container, 0, 3);
    }
    else if (view == MIDAS_VIEW_3V)
    {
      m_GridLayout->addWidget(this->mitkWidget1Container, 0, 0);
      m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);
      m_GridLayout->addWidget(this->mitkWidget3Container, 2, 0);
      m_GridLayout->addWidget(this->mitkWidget4Container, 3, 0);
    }
    else if (view == MIDAS_VIEW_SAG_COR_H)
    {
      m_GridLayout->addWidget(this->mitkWidget2Container, 0, 0);
      m_GridLayout->addWidget(this->mitkWidget3Container, 0, 1);
      m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);
      m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);
    }
    else if (view == MIDAS_VIEW_SAG_COR_V)
    {
      m_GridLayout->addWidget(this->mitkWidget2Container, 0, 0);
      m_GridLayout->addWidget(this->mitkWidget1Container, 0, 1);
      m_GridLayout->addWidget(this->mitkWidget3Container, 1, 0);
      m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);
    }
    else if (view == MIDAS_VIEW_AX_COR_H)
    {
      m_GridLayout->addWidget(this->mitkWidget1Container, 0, 0);
      m_GridLayout->addWidget(this->mitkWidget3Container, 0, 1);
      m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);
      m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);
    }
    else if (view == MIDAS_VIEW_AX_SAG_V)
    {
      m_GridLayout->addWidget(this->mitkWidget1Container, 0, 0);
      m_GridLayout->addWidget(this->mitkWidget3Container, 0, 1);
      m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);
      m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);
    }
    else
    {
      m_GridLayout->addWidget(this->mitkWidget1Container, 0, 0);
      m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);
      m_GridLayout->addWidget(this->mitkWidget3Container, 1, 0);
      m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);
    }

    QmitkStdMultiWidgetLayout->addLayout(m_GridLayout);
  }

  switch(view)
  {
  case MIDAS_VIEW_AXIAL:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_VIEW_SAGITTAL:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->hide();
    this->mitkWidget2->setFocus();
    break;
  case MIDAS_VIEW_CORONAL:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget3->setFocus();
    break;
  case MIDAS_VIEW_ORTHO:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->show();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_VIEW_3H:
  case MIDAS_VIEW_3V:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_VIEW_3D:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->show();
    this->mitkWidget4->setFocus();
    break;
  case MIDAS_VIEW_SAG_COR_H:
  case MIDAS_VIEW_SAG_COR_V:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget2->setFocus();
    break;
  case MIDAS_VIEW_AX_COR_H:
  case MIDAS_VIEW_AX_COR_V:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_VIEW_AX_SAG_H:
  case MIDAS_VIEW_AX_SAG_V:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  default:
    // die, this should never happen
    assert((m_View >= 0 && m_View <= 6) || (m_View >= 9 && m_View <= 14));
    break;
  }
  m_View = view;
  this->Update3DWindowVisibility();
  this->m_GridLayout->activate();
}

MIDASView QmitkMIDASStdMultiWidget::GetMIDASView() const
{
  return m_View;
}


bool QmitkMIDASStdMultiWidget::IsSingle2DView() const
{
  bool result = false;
  if (m_View == MIDAS_VIEW_AXIAL || m_View == MIDAS_VIEW_SAGITTAL || m_View == MIDAS_VIEW_CORONAL)
  {
    result = true;
  }
  return result;
}

mitk::SliceNavigationController::Pointer QmitkMIDASStdMultiWidget::GetSliceNavigationController(MIDASOrientation orientation) const
{
  mitk::SliceNavigationController::Pointer result = NULL;
  if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    result = mitkWidget1->GetSliceNavigationController();
  }
  else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    result = mitkWidget2->GetSliceNavigationController();
  }
  else if (orientation == MIDAS_ORIENTATION_CORONAL)
  {
    result = mitkWidget3->GetSliceNavigationController();
  }
  return result;
}

unsigned int QmitkMIDASStdMultiWidget::GetMinSlice(MIDASOrientation orientation) const
{
  return 0;
}

unsigned int QmitkMIDASStdMultiWidget::GetMaxSlice(MIDASOrientation orientation) const
{
  unsigned int result = 0;

  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(orientation);
  assert(snc);

  if (snc->GetSlice() != NULL)
  {
    if (snc->GetSlice()->GetSteps() >= 1)
    {
      result = snc->GetSlice()->GetSteps() -1;
    }
  }
  return result;
}

unsigned int QmitkMIDASStdMultiWidget::GetMinTime() const
{
  return 0;
}

unsigned int QmitkMIDASStdMultiWidget::GetMaxTime() const
{
  unsigned int result = 0;

  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  assert(snc);

  if (snc->GetTime() != NULL)
  {
    if (snc->GetTime()->GetSteps() >= 1)
    {
      result = snc->GetTime()->GetSteps() -1;
    }
  }
  return result;
}

void QmitkMIDASStdMultiWidget::OnScaleFactorChanged(QmitkRenderWindow *renderWindow)
{
  if (!m_BlockDisplayGeometryEvents)
  {
    double magnificationFactor = ComputeMagnificationFactor(renderWindow);
    if (magnificationFactor != m_MagnificationFactor)
    {
      MITK_DEBUG << "New magnification factor: " << magnificationFactor;
      m_MagnificationFactor = magnificationFactor;
      emit MagnificationFactorChanged(renderWindow, magnificationFactor);
    }
    else
    {
      MITK_DEBUG << "magnification factor not changed: " << magnificationFactor;
    }
  }
}

void QmitkMIDASStdMultiWidget::OnPositionChanged(MIDASOrientation orientation)
{
  const mitk::Geometry3D *geometry = m_Geometry;
  if (geometry != NULL)
  {
    int sliceNumber = 0;
    mitk::Index3D voxelPoint;
    mitk::Point3D millimetrePoint = this->GetCrossPosition();
    int axis = m_OrientationToAxisMap[orientation];

    geometry->WorldToIndex(millimetrePoint, voxelPoint);
    sliceNumber = voxelPoint[axis];

    QmitkRenderWindow *window = NULL;
    if (orientation == MIDAS_ORIENTATION_AXIAL)
    {
      window = this->mitkWidget1;
    }
    else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      window = this->mitkWidget2;
    }
    else if (orientation == MIDAS_ORIENTATION_CORONAL)
    {
      window = this->mitkWidget3;
    }
    emit PositionChanged(window, voxelPoint, millimetrePoint, sliceNumber, orientation);
  }
}

void QmitkMIDASStdMultiWidget::SetSliceNumber(MIDASOrientation orientation, unsigned int sliceNumber)
{
  const mitk::Geometry3D *geometry = m_Geometry;
  if (geometry != NULL)
  {
    mitk::Index3D voxelPoint;
    mitk::Point3D millimetrePoint = this->GetCrossPosition();

    geometry->WorldToIndex(millimetrePoint, voxelPoint);

    int axis = m_OrientationToAxisMap[orientation];
    voxelPoint[axis] = sliceNumber;

    mitk::Point3D tmp;
    tmp[0] = voxelPoint[0];
    tmp[1] = voxelPoint[1];
    tmp[2] = voxelPoint[2];

    geometry->IndexToWorld(tmp, millimetrePoint);

    // Does not work, as it relies on the StateMachine event broadcasting mechanism,
    // and if the widget is not listening, then it goes unnoticed.
    //this->MoveCrossToPosition(millimetrePoint);

    // This however, directly forces the SNC to the right place.
    mitkWidget1->GetSliceNavigationController()->SelectSliceByPoint(millimetrePoint);
    mitkWidget2->GetSliceNavigationController()->SelectSliceByPoint(millimetrePoint);
    mitkWidget3->GetSliceNavigationController()->SelectSliceByPoint(millimetrePoint);
  }
}

unsigned int QmitkMIDASStdMultiWidget::GetSliceNumber(const MIDASOrientation orientation) const
{
  int sliceNumber = 0;

  const mitk::Geometry3D *geometry = m_Geometry;
  if (geometry != NULL)
  {
    mitk::Index3D voxelPoint;
    mitk::Point3D millimetrePoint = this->GetCrossPosition();

    geometry->WorldToIndex(millimetrePoint, voxelPoint);

    int axis = m_OrientationToAxisMap[orientation];
    sliceNumber = voxelPoint[axis];
  }

  return sliceNumber;
}

void QmitkMIDASStdMultiWidget::SetTime(unsigned int timeSlice)
{
  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  snc->GetTime()->SetPos(timeSlice);

  snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_SAGITTAL);
  snc->GetTime()->SetPos(timeSlice);

  snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_CORONAL);
  snc->GetTime()->SetPos(timeSlice);
}

unsigned int QmitkMIDASStdMultiWidget::GetTime() const
{
  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  assert(snc);

  return snc->GetTime()->GetPos();
}

double QmitkMIDASStdMultiWidget::GetMagnificationFactor() const
{
  return m_MagnificationFactor;
}

void QmitkMIDASStdMultiWidget::SetMagnificationFactor(double magnificationFactor)
{
  if (m_MagnificationFactor == magnificationFactor)
  {
    return;
  }
  m_BlockDisplayGeometryEvents = true;

  // The aim of this method, is that when a magnificationFactor is passed in,
  // all 2D views update to an equivalent zoom, even if they were different beforehand.
  // The magnification factor is as it would be displayed in MIDAS, i.e. an integer
  // that corresponds to the rules given at the top of the header file.

  // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
  std::vector<QmitkRenderWindow*> windows = this->GetAllWindows();
  for (unsigned int i = 0; i < 3; i++)
  {
    QmitkRenderWindow *window = windows[i];

    double zoomScaleFactor = ComputeScaleFactor(window, magnificationFactor);
    this->ZoomDisplayAboutCentre(window, zoomScaleFactor);
  }

  m_MagnificationFactor = magnificationFactor;
  this->RequestUpdate();

  m_BlockDisplayGeometryEvents = false;
}

double QmitkMIDASStdMultiWidget::ComputeScaleFactor(QmitkRenderWindow* window, double magnificationFactor)
{
  mitk::Point2D scaleFactorPixPerVoxel;
  mitk::Point2D scaleFactorPixPerMillimetres;
  this->GetScaleFactors(window, scaleFactorPixPerVoxel, scaleFactorPixPerMillimetres);

  double effectiveMagnificationFactor = 0.0;
  if (magnificationFactor >= 0.0)
  {
    effectiveMagnificationFactor = magnificationFactor + 1.0;
  }
  else
  {
    effectiveMagnificationFactor = -1.0 / (magnificationFactor - 1.0);
  }

  mitk::Point2D targetScaleFactor;

  // Need to scale both of the current scaleFactorPixPerVoxel[i]
  for (int i = 0; i < 2; i++)
  {
    targetScaleFactor[i] = effectiveMagnificationFactor / scaleFactorPixPerVoxel[i];
  }

  // Pick the one that has changed the least
  int axisWithLeastDifference = -1;
  double leastDifference = std::numeric_limits<double>::max();
  for(int i = 0; i < 2; i++)
  {
    double difference = fabs(targetScaleFactor[i] - 1.0);
    if (difference < leastDifference)
    {
      leastDifference = difference;
      axisWithLeastDifference = i;
    }
  }

  return targetScaleFactor[axisWithLeastDifference];
}

double QmitkMIDASStdMultiWidget::ComputeMagnificationFactor(QmitkRenderWindow* window)
{
  if (this->GetOrientation() == MIDAS_ORIENTATION_UNKNOWN)
  {
    MITK_DEBUG << "if (this->GetOrientation() == MIDAS_ORIENTATION_UNKNOWN): true";
    return 0;
  }

  // We do this with mitk::Point2D, so we have different values in X and Y, as images can be anisotropic.
  mitk::Point2D scaleFactorPixPerVoxel;
  mitk::Point2D scaleFactorPixPerMillimetres;
  this->GetScaleFactors(window, scaleFactorPixPerVoxel, scaleFactorPixPerMillimetres);

  // Now we scale these values so we get an integer number of pixels per voxel.
  mitk::Point2D targetScaleFactorPixPerVoxel;
  mitk::Point2D targetScaleFactorPixPerMillimetres;

  // Need to round the scale factors.
  for (int i = 0; i < 2; i++)
  {
    // If they are >= than 1, we round down towards 1
    // so you have less pixels per voxel, so image will appear smaller.
    if (scaleFactorPixPerVoxel[i] >= 1)
    {
      targetScaleFactorPixPerVoxel[i] = scaleFactorPixPerVoxel[i];
    }
    else
    {
      // Otherwise, we still have to make image smaller to fit it on screen.
      //
      // For example, if the scale factor is 0.5, we invert it to get 2, which is an integer, so OK.
      // If however the scale factor is 0.4, we invert it to get 2.5 voxels per pixel, so we have
      // to round it up to 3, which means the image gets smaller (3 voxels fit into 1 pixel), and then
      // invert it to get the scale factor again.
      double tmp = 1.0 / scaleFactorPixPerVoxel[i];
      int roundedTmp = (int)(tmp + 0.5);
      targetScaleFactorPixPerVoxel[i] = 1.0 / roundedTmp;
    }
    targetScaleFactorPixPerMillimetres[i] = scaleFactorPixPerMillimetres[i] * (targetScaleFactorPixPerVoxel[i] / scaleFactorPixPerVoxel[i]);
  }

  // We may have anisotropic voxels, so find the axis that requires most scale factor change.
  int axisWithLargestDifference = 0;
  double largestDifference = -1.0 * std::numeric_limits<double>::max();
  for(int i = 0; i < 2; i++)
  {
    double difference = fabs(targetScaleFactorPixPerVoxel[i] - scaleFactorPixPerVoxel[i]);
    if (difference > largestDifference)
    {
      largestDifference = fabs(targetScaleFactorPixPerVoxel[i] - scaleFactorPixPerVoxel[i]);
      axisWithLargestDifference = i;
    }
  }

  double effectiveMagnificationFactor = targetScaleFactorPixPerVoxel[axisWithLargestDifference];

  /*
  * Note: The requirements specification for MIDAS style zoom basically says.
  *
  * magnification   : actual pixels per voxel.
  * on MIDAS widget :
  * 2               : 3
  * 1               : 2
  * 0               : 1 (i.e. no magnification).
  * -1              : 0.5 (i.e. 1 pixel covers 2 voxels).
  * -2              : 0.33 (i.e. 1 pixel covers 3 voxels).
  */

  // See comments at top of header file
  double magnificationFactor;
  if (effectiveMagnificationFactor >= 1.0)
  {
    // So, if pixels per voxel = 2, midas magnification = 1.
    // So, if pixels per voxel = 1, midas magnification = 0. etc.
    magnificationFactor = effectiveMagnificationFactor - 1.0;
  }
  else
  {
    magnificationFactor = (-1.0 / effectiveMagnificationFactor) + 1.0;
  }
  return magnificationFactor;
}

double QmitkMIDASStdMultiWidget::FitMagnificationFactor()
{
  double magnificationFactor = 0.0;

  MIDASOrientation orientation = this->GetOrientation();
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    // Note, that this method should only be called for MIDAS purposes, when the view is a 2D
    // view, so it will either be Axial, Coronal, Sagittal, and not 3D or OthoView.
    QmitkRenderWindow *window = this->GetRenderWindow(orientation);

    // Given the above comment, this means, we MUST have a window from this choice of 3.
    assert(window);

    //////////////////////////////////////////////////////////////////////////
    // Use the current window to work out a reasonable magnification factor.
    // This has to be done for each window, as they may be out of sync
    // due to the user manually (right click + mouse move) zooming the window.
    //////////////////////////////////////////////////////////////////////////

    magnificationFactor = ComputeMagnificationFactor(window);
    magnificationFactor = static_cast<int>(magnificationFactor);
  }
  return magnificationFactor;
}

void QmitkMIDASStdMultiWidget::GetScaleFactors(
    QmitkRenderWindow *window,
    mitk::Point2D &scaleFactorPixPerVoxel,
    mitk::Point2D &scaleFactorPixPerMillimetres)
{
  // Basic initialization - default value is arbitrarily 1 in both cases.
  scaleFactorPixPerVoxel[0] = 1;
  scaleFactorPixPerVoxel[1] = 1;
  scaleFactorPixPerMillimetres[0] = 1;
  scaleFactorPixPerMillimetres[1] = 1;

  if (window != NULL)
  {
    const mitk::Geometry3D *geometry = window->GetSliceNavigationController()->GetInputWorldGeometry();

    mitk::SliceNavigationController* sliceNavigationController = window->GetSliceNavigationController();
    assert(sliceNavigationController);

    mitk::BaseRenderer::Pointer baseRenderer = sliceNavigationController->GetRenderer();
    assert(baseRenderer);

    mitk::DisplayGeometry::Pointer displayGeometry = baseRenderer->GetDisplayGeometry();
    assert(displayGeometry);

    mitk::Point3D cornerPointsInImage[8];
    cornerPointsInImage[0] = geometry->GetCornerPoint(true, true, true);
    cornerPointsInImage[1] = geometry->GetCornerPoint(true, true, false);
    cornerPointsInImage[2] = geometry->GetCornerPoint(true, false, true);
    cornerPointsInImage[3] = geometry->GetCornerPoint(true, false, false);
    cornerPointsInImage[4] = geometry->GetCornerPoint(false, true, true);
    cornerPointsInImage[5] = geometry->GetCornerPoint(false, true, false);
    cornerPointsInImage[6] = geometry->GetCornerPoint(false, false, true);
    cornerPointsInImage[7] = geometry->GetCornerPoint(false, false, false);

    scaleFactorPixPerVoxel[0] = std::numeric_limits<float>::max();
    scaleFactorPixPerVoxel[1] = std::numeric_limits<float>::max();

    // Take every combination of pairs of 3D corner points taken from the 8 corners of the geometry.
    for (unsigned int i = 0; i < 8; i++)
    {
      mitk::Point3D pointsInVoxels[2];

      for (unsigned int j = 1; j < 8; j++)
      {
        geometry->WorldToIndex(cornerPointsInImage[i], pointsInVoxels[0]);
        geometry->WorldToIndex(cornerPointsInImage[j], pointsInVoxels[1]);

        // We only want to pick pairs of points where the points are different
        // and also differ in 3D space along exactly one axis (i.e. no diagonals).
        unsigned int differentVoxelIndexesCounter=0;

        for (unsigned int k = 0; k < 3; k++)
        {
          if (fabs(pointsInVoxels[1][k] - pointsInVoxels[0][k]) > 0.1)
          {
            differentVoxelIndexesCounter++;
          }
        }
        if (differentVoxelIndexesCounter == 1)
        {
          // So, for this pair (i,j) of points, project to 2D
          mitk::Point2D displayPointInMillimetreCoordinates[2];
          mitk::Point2D displayPointInPixelCoordinates[2];

          displayGeometry->Map(cornerPointsInImage[i], displayPointInMillimetreCoordinates[0]);
          displayGeometry->WorldToDisplay(displayPointInMillimetreCoordinates[0], displayPointInPixelCoordinates[0]);

          displayGeometry->Map(cornerPointsInImage[j], displayPointInMillimetreCoordinates[1]);
          displayGeometry->WorldToDisplay(displayPointInMillimetreCoordinates[1], displayPointInPixelCoordinates[1]);

          // Similarly, we only want to pick pairs of points where the projected 2D points
          // differ in 2D display coordinates along exactly one axis.
          unsigned int differentDisplayIndexesCounter=0;
          int differentDisplayAxis = -1;

          for (unsigned int k = 0; k < 2; k++)
          {
            if (fabs(displayPointInPixelCoordinates[1][k] - displayPointInPixelCoordinates[0][k]) > 0.1)
            {
              differentDisplayIndexesCounter++;
              differentDisplayAxis = k;
            }
          }
          if (differentDisplayIndexesCounter == 1)
          {
            // We now have i,j corresponding to a pair of points that are different in
            // 1 axis in voxel space, and different in one axis in diplay space, we can
            // use them to calculate scale factors.

            double distanceInMillimetres = cornerPointsInImage[i].EuclideanDistanceTo(cornerPointsInImage[j]);
            double distanceInVoxels = pointsInVoxels[0].EuclideanDistanceTo(pointsInVoxels[1]);
            double distanceInPixels = displayPointInPixelCoordinates[0].EuclideanDistanceTo(displayPointInPixelCoordinates[1]);
            double scaleFactorInDisplayPixelsPerImageVoxel = distanceInPixels / distanceInVoxels;
            double scaleFactorInDisplayPixelsPerMillimetres = distanceInPixels / distanceInMillimetres;

            if (scaleFactorInDisplayPixelsPerImageVoxel < scaleFactorPixPerVoxel[differentDisplayAxis])
            {
              scaleFactorPixPerVoxel[differentDisplayAxis] = scaleFactorInDisplayPixelsPerImageVoxel;
              scaleFactorPixPerMillimetres[differentDisplayAxis] = scaleFactorInDisplayPixelsPerMillimetres;
            }
          }
        }
      }
    }
  } // end window != NULL
}

void QmitkMIDASStdMultiWidget::ZoomDisplayAboutCentre(QmitkRenderWindow *window, double scaleFactor)
{
  if (window != NULL)
  {
    // I'm using assert statements, because fundamentally, if the window exists, so should all the other objects.
    mitk::SliceNavigationController* sliceNavigationController = window->GetSliceNavigationController();
    assert(sliceNavigationController);

    mitk::BaseRenderer* baseRenderer = sliceNavigationController->GetRenderer();
    assert(baseRenderer);

    mitk::DisplayGeometry* displayGeometry = baseRenderer->GetDisplayGeometry();
    assert(displayGeometry);

    mitk::Vector2D sizeInDisplayUnits = displayGeometry->GetSizeInDisplayUnits();
    mitk::Point2D centreOfDisplayInDisplayUnits;

    centreOfDisplayInDisplayUnits[0] = (sizeInDisplayUnits[0]-1.0)/2.0;
    centreOfDisplayInDisplayUnits[1] = (sizeInDisplayUnits[1]-1.0)/2.0;

    // Note that the scaleFactor is cumulative or multiplicative rather than absolute.
    displayGeometry->Zoom(scaleFactor, centreOfDisplayInDisplayUnits);

    // Now shift the origin, so the viewport is centred on the centre of the image.
    mitk::Geometry3D::ConstPointer geometry = sliceNavigationController->GetInputWorldGeometry();
    if (geometry.IsNotNull())
    {
      mitk::Point3D centreInVoxels;
      mitk::Point3D centreInMillimetres;
      mitk::Point2D projectedCentreInMillimetres;
      mitk::Point2D projectedCentreInPixels;

      centreInVoxels[0] = (geometry->GetExtent(0)-1)/2.0;
      centreInVoxels[1] = (geometry->GetExtent(1)-1)/2.0;
      centreInVoxels[2] = (geometry->GetExtent(2)-1)/2.0;

      geometry->IndexToWorld(centreInVoxels, centreInMillimetres);
      displayGeometry->Map(centreInMillimetres, projectedCentreInMillimetres);
      displayGeometry->WorldToDisplay(projectedCentreInMillimetres, projectedCentreInPixels);

      mitk::Vector2D difference;
      difference[0] = projectedCentreInPixels[0] - centreOfDisplayInDisplayUnits[0];
      difference[1] = projectedCentreInPixels[1] - centreOfDisplayInDisplayUnits[1];
      displayGeometry->MoveBy(difference);
    }
  }
}

void QmitkMIDASStdMultiWidget::StoreCameras()
{
  std::vector<QmitkRenderWindow*> windows = this->GetAllWindows();
  for (unsigned int i = 0; i < windows.size(); i++)
  {
    vtkCamera* camera = windows[i]->GetRenderer()->GetVtkRenderer()->GetActiveCamera();
    this->m_Cameras[i]->SetPosition(camera->GetPosition());
    this->m_Cameras[i]->SetFocalPoint(camera->GetFocalPoint());
    this->m_Cameras[i]->SetViewUp(camera->GetViewUp());
    this->m_Cameras[i]->SetClippingRange(camera->GetClippingRange());
  }
}

void QmitkMIDASStdMultiWidget::RestoreCameras()
{
  std::vector<QmitkRenderWindow*> windows = this->GetAllWindows();
  for (unsigned int i = 0; i < windows.size(); i++)
  {
    vtkCamera* camera = windows[i]->GetRenderer()->GetVtkRenderer()->GetActiveCamera();
    camera->SetPosition(this->m_Cameras[i]->GetPosition());
    camera->SetFocalPoint(this->m_Cameras[i]->GetFocalPoint());
    camera->SetViewUp(this->m_Cameras[i]->GetViewUp());
    camera->SetClippingRange(this->m_Cameras[i]->GetClippingRange());
  }
}

QmitkRenderWindow* QmitkMIDASStdMultiWidget::GetRenderWindow(const MIDASOrientation& orientation) const
{
  QmitkRenderWindow *window = NULL;
  if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    window = this->GetRenderWindow1();
  }
  else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    window = this->GetRenderWindow2();
  }
  else if (orientation == MIDAS_ORIENTATION_CORONAL)
  {
    window = this->GetRenderWindow3();
  }
  return window;
}

int QmitkMIDASStdMultiWidget::GetSliceUpDirection(MIDASOrientation orientation) const
{
  int result = 0;
  if (this->m_Geometry != NULL)
  {
    result = mitk::GetUpDirection(m_Geometry, orientation);
  }
  return result;
}
