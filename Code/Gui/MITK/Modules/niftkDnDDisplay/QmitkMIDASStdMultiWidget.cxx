/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASStdMultiWidget.h"

#include <cmath>
#include <itkMatrix.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <QmitkRenderWindow.h>
#include <QGridLayout>
#include <mitkMIDASOrientationUtils.h>

#include <usGetModuleContext.h>
#include <usModuleRegistry.h>
#include <mitkVtkLayerController.h>

#include "vtkSideAnnotation.h"

/**
 * This class is to notify the SingleViewWidget about the display geometry changes of a render window.
 */
class DisplayGeometryModificationCommand : public itk::Command
{
public:
  mitkNewMacro3Param(DisplayGeometryModificationCommand, QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, mitk::DisplayGeometry*);


  //-----------------------------------------------------------------------------
  DisplayGeometryModificationCommand(QmitkMIDASStdMultiWidget* stdMultiWidget, QmitkRenderWindow* renderWindow, mitk::DisplayGeometry* displayGeometry)
  : itk::Command()
  , m_StdMultiWidget(stdMultiWidget)
  , m_RenderWindow(renderWindow)
  , m_DisplayGeometry(displayGeometry)
  , m_Origin(displayGeometry->GetOriginInMM())
  , m_ScaleFactor(displayGeometry->GetScaleFactorMMPerDisplayUnit())
  {
  }


  //-----------------------------------------------------------------------------
  void Execute(itk::Object* caller, const itk::EventObject& event)
  {
    this->Execute((const itk::Object*) caller, event);
  }


  //-----------------------------------------------------------------------------
  void Execute(const itk::Object* object, const itk::EventObject& /*event*/)
  {
    // Note that the scaling changes the scale factor *and* the origin,
    // while the moving changes the origin only.

    bool beingPanned = true;

    double scaleFactor = m_DisplayGeometry->GetScaleFactorMMPerDisplayUnit();
    if (scaleFactor != m_ScaleFactor)
    {
      beingPanned = false;
      m_StdMultiWidget->OnScaleFactorChanged(m_RenderWindow, scaleFactor);
      m_ScaleFactor = scaleFactor;
    }

    mitk::Vector2D origin = m_DisplayGeometry->GetOriginInDisplayUnits();
    if (origin != m_Origin)
    {
      if (beingPanned)
      {
        m_StdMultiWidget->OnOriginChanged(m_RenderWindow, beingPanned);
      }
      m_Origin = origin;
    }
  }

private:
  QmitkMIDASStdMultiWidget* const m_StdMultiWidget;
  QmitkRenderWindow* const m_RenderWindow;
  mitk::DisplayGeometry* const m_DisplayGeometry;
  mitk::Vector2D m_Origin;
  double m_ScaleFactor;
};


//-----------------------------------------------------------------------------
QmitkMIDASStdMultiWidget::QmitkMIDASStdMultiWidget(
    QWidget* parent,
    Qt::WindowFlags flags,
    mitk::RenderingManager* renderingManager,
    mitk::DataStorage* dataStorage
    )
: QmitkStdMultiWidget(parent, flags, renderingManager)
, m_GridLayout(NULL)
, m_AxialSliceTag(0)
, m_SagittalSliceTag(0)
, m_CoronalSliceTag(0)
, m_IsSelected(false)
, m_IsEnabled(false)
, m_SelectedRenderWindow(0)
, m_Display2DCursorsLocally(true)
, m_Display2DCursorsGlobally(false)
, m_Show3DWindowInOrthoView(false)
, m_Layout(MIDAS_LAYOUT_ORTHO)
, m_Magnification(0.0)
, m_ScaleFactor(0.0)
, m_Geometry(NULL)
, m_BlockDisplayGeometryEvents(false)
, m_CursorPositionsAreBound(true)
, m_ScaleFactorsAreBound(true)
{
  m_RenderWindows[0] = this->GetRenderWindow1();
  m_RenderWindows[1] = this->GetRenderWindow2();
  m_RenderWindows[2] = this->GetRenderWindow3();
  m_RenderWindows[3] = this->GetRenderWindow4();

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

  // Set these off, as it wont matter until there is an image dropped, with a specific layout and orientation.
  m_CornerAnnotaions[0].cornerText->SetText(0, "");
  m_CornerAnnotaions[1].cornerText->SetText(0, "");
  m_CornerAnnotaions[2].cornerText->SetText(0, "");

  for (int i = 0; i < 3; ++i)
  {
    m_DirectionAnnotations[i] = vtkSideAnnotation::New();
    m_DirectionAnnotations[i]->SetMaximumFontSize(16);
    m_DirectionAnnotations[i]->GetTextProperty()->BoldOn();
    m_DirectionAnnotationRenderers[i] = vtkRenderer::New();
    m_DirectionAnnotationRenderers[i]->AddActor(m_DirectionAnnotations[i]);
    m_DirectionAnnotationRenderers[i]->InteractiveOff();
    mitk::VtkLayerController::GetInstance(m_RenderWindows[i]->GetRenderWindow())->InsertForegroundRenderer(m_DirectionAnnotationRenderers[i], true);
  }

  double axialColour[3] = {1.0, 0.0, 0.0};
  double sagittalColour[3] = {0.0, 1.0, 0.0};
  double coronalColour[3] = {0.295, 0.295, 1.0};

  m_DirectionAnnotations[0]->SetColour(0, coronalColour);
  m_DirectionAnnotations[0]->SetColour(1, sagittalColour);
  m_DirectionAnnotations[0]->SetColour(2, coronalColour);
  m_DirectionAnnotations[0]->SetColour(3, sagittalColour);
  m_DirectionAnnotations[1]->SetColour(0, axialColour);
  m_DirectionAnnotations[1]->SetColour(1, coronalColour);
  m_DirectionAnnotations[1]->SetColour(2, axialColour);
  m_DirectionAnnotations[1]->SetColour(3, coronalColour);
  m_DirectionAnnotations[2]->SetColour(0, axialColour);
  m_DirectionAnnotations[2]->SetColour(1, sagittalColour);
  m_DirectionAnnotations[2]->SetColour(2, axialColour);
  m_DirectionAnnotations[2]->SetColour(3, sagittalColour);

  // Set default layout. This must be ORTHO.
  this->SetLayout(MIDAS_LAYOUT_ORTHO);

  // Default to unselected, so borders are off.
  this->SetSelected(false);

  // Need each widget to signal when something is dropped, so connect signals to OnNodesDropped.
  QObject::connect(this->mitkWidget1, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  QObject::connect(this->mitkWidget2, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  QObject::connect(this->mitkWidget3, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  QObject::connect(this->mitkWidget4, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)));

  // Register to listen to SliceNavigators, slice changed events.
  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onAxialSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onAxialSliceChangedCommand->SetCallbackFunction(this, &QmitkMIDASStdMultiWidget::OnAxialSliceChanged);
  m_AxialSliceTag = mitkWidget1->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onAxialSliceChangedCommand);

  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onSagittalSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onSagittalSliceChangedCommand->SetCallbackFunction(this, &QmitkMIDASStdMultiWidget::OnSagittalSliceChanged);
  m_SagittalSliceTag = mitkWidget2->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSagittalSliceChangedCommand);

  itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::Pointer onCoronalSliceChangedCommand =
    itk::ReceptorMemberCommand<QmitkMIDASStdMultiWidget>::New();
  onCoronalSliceChangedCommand->SetCallbackFunction(this, &QmitkMIDASStdMultiWidget::OnCoronalSliceChanged);
  m_CoronalSliceTag = mitkWidget3->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onCoronalSliceChangedCommand);

  // The cursor is at the middle of the display at the beginning.
  m_CursorPosition[0] = 0.5;
  m_CursorPosition[1] = 0.5;
  m_CursorPosition[2] = 0.5;

  // The world position is unknown until the geometry is set. These values are invalid,
  // but still better then having undefined values.
  m_SelectedPosition[0] = 0.0;
  m_SelectedPosition[1] = 0.0;
  m_SelectedPosition[2] = 0.0;

  // Set the default voxel size to 1.0mm for each axes.
  m_MmPerVx[0] = 1.0;
  m_MmPerVx[1] = 1.0;
  m_MmPerVx[2] = 1.0;

  // Listen to the display geometry changes so we raise an event when
  // the geometry changes through the display interactor (e.g. zooming with the mouse).
  std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  for (int i = 0; i < 3; ++i)
  {
    AddDisplayGeometryModificationObserver(renderWindows[i]);
  }

  // The mouse mode switcher is declared and initialised in QmitkStdMultiWidget. It creates an
  // mitk::DisplayInteractor. This line decreases the reference counter of the mouse mode switcher
  // so that it is destructed and it unregisters and destructs its display interactor as well.
  m_MouseModeSwitcher = 0;
}


//-----------------------------------------------------------------------------
QmitkMIDASStdMultiWidget::~QmitkMIDASStdMultiWidget()
{
  // Release the display interactor.
  this->SetDisplayInteractionsEnabled(false);

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

  // Stop listening to the display geometry changes so we raise an event when
  // the geometry changes through the display interactor (e.g. zooming with the mouse).
  std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  for (unsigned i = 0; i < 3; ++i)
  {
    RemoveDisplayGeometryModificationObserver(renderWindows[i]);
  }

  for (int i = 0; i < 3; ++i)
  {
    mitk::VtkLayerController::GetInstance(this->m_RenderWindows[i]->GetRenderWindow())->RemoveRenderer(m_DirectionAnnotationRenderers[i]);
  }

  for (int i = 0; i < 3; ++i)
  {
    m_DirectionAnnotations[i]->Delete();
    m_DirectionAnnotationRenderers[i]->Delete();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::AddDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow)
{
  mitk::BaseRenderer* renderer = renderWindow->GetRenderer();
  assert(renderer);

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  assert(displayGeometry);

  DisplayGeometryModificationCommand::Pointer command = DisplayGeometryModificationCommand::New(this, renderWindow, displayGeometry);
  unsigned long observerTag = displayGeometry->AddObserver(itk::ModifiedEvent(), command);
  m_DisplayGeometryModificationObservers[renderWindow] = observerTag;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::RemoveDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow)
{
  mitk::BaseRenderer* renderer = renderWindow->GetRenderer();
  assert(renderer);

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  assert(displayGeometry);

  displayGeometry->RemoveObserver(m_DisplayGeometryModificationObservers[renderWindow]);
  m_DisplayGeometryModificationObservers.erase(renderWindow);
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::OnNodesDropped(QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes)
{
  // We have to block the display geometry events until the event is processed.
  // Therefore, it is important that the event is processed synchronuously, right after
  // emitting the signal. To achieve this, the signal has to be directly connected to the
  // slots (Qt::DirectConnection). Another way of achieving this could be calling
  // QApplication::processEvents() but that would process other pending signals as well
  // what we might not want.
  m_BlockDisplayGeometryEvents = true;
  emit NodesDropped(this, renderWindow, nodes);
  m_BlockDisplayGeometryEvents = false;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::OnAxialSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(MIDAS_ORIENTATION_AXIAL);
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::OnSagittalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(MIDAS_ORIENTATION_SAGITTAL);
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::OnCoronalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(MIDAS_ORIENTATION_CORONAL);
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetBackgroundColor(QColor colour)
{
  m_BackgroundColor = colour;
  m_GradientBackground1->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground1->Enable();
  m_GradientBackground2->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground2->Enable();
  m_GradientBackground3->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground3->Enable();
  m_GradientBackground4->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground4->Enable();
}


//-----------------------------------------------------------------------------
QColor QmitkMIDASStdMultiWidget::GetBackgroundColor() const
{
  return m_BackgroundColor;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetSelected(bool selected)
{
  m_IsSelected = selected;

  if (selected)
  {
    this->EnableColoredRectangles();
  }
  else
  {
    this->DisableColoredRectangles();
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::IsSelected() const
{
  return m_IsSelected;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASStdMultiWidget::GetSelectedRenderWindow() const
{
  return m_SelectedRenderWindow;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetSelectedRenderWindow(QmitkRenderWindow* renderWindow)
{
  // When we "Select", the selection is at the level of the QmitkMIDASStdMultiWidget
  // so the whole of this widget is selected. However, we may have clicked in
  // a specific view, so it still helps to highlight the most recently clicked on view.
  // Also, if you are displaying ortho layout (2x2) then you actually have 4 windows present,
  // then highlighting them all starts to look a bit confusing, so we just highlight the
  // most recently focused window, (eg. axial, sagittal, coronal or 3D).

  m_IsSelected = true;
  m_SelectedRenderWindow = renderWindow;

  if (renderWindow == this->GetRenderWindow1())
  {
    m_RectangleRendering1->Enable(1.0, 0.0, 0.0);
    m_RectangleRendering2->Disable();
    m_RectangleRendering3->Disable();
    m_RectangleRendering4->Disable();
  }
  else if (renderWindow == this->GetRenderWindow2())
  {
    m_IsSelected = true;
    m_RectangleRendering1->Disable();
    m_RectangleRendering2->Enable(0.0, 1.0, 0.0);
    m_RectangleRendering3->Disable();
    m_RectangleRendering4->Disable();
  }
  else if (renderWindow == this->GetRenderWindow3())
  {
    m_IsSelected = true;
    m_RectangleRendering1->Disable();
    m_RectangleRendering2->Disable();
    m_RectangleRendering3->Enable(0.0, 0.0, 1.0);
    m_RectangleRendering4->Disable();
  }
  else if (renderWindow == this->GetRenderWindow4())
  {
    m_IsSelected = true;
    m_RectangleRendering1->Disable();
    m_RectangleRendering2->Disable();
    m_RectangleRendering3->Disable();
    m_RectangleRendering4->Enable(1.0, 1.0, 0.0);
  }
  else
  {
    this->SetSelected(false);
  }
  this->ForceImmediateUpdate();

  if (renderWindow && !this->AreScaleFactorsBound())
  {
    m_Magnification = this->GetMagnification(renderWindow);
    m_ScaleFactor = this->GetScaleFactor(renderWindow);
  }
}


//-----------------------------------------------------------------------------
std::vector<QmitkRenderWindow*> QmitkMIDASStdMultiWidget::GetVisibleRenderWindows() const
{
  std::vector<QmitkRenderWindow*> renderWindows;

  if (this->mitkWidget1Container->isVisible())
  {
    renderWindows.push_back(this->GetRenderWindow1());
  }
  if (this->mitkWidget2Container->isVisible())
  {
    renderWindows.push_back(this->GetRenderWindow2());
  }
  if (this->mitkWidget3Container->isVisible())
  {
    renderWindows.push_back(this->GetRenderWindow3());
  }
  if (this->mitkWidget4Container->isVisible())
  {
    renderWindows.push_back(this->GetRenderWindow4());
  }
  return renderWindows;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::RequestUpdate()
{
  // The point of all this is to minimise the number of Updates.
  // So, ONLY call RequestUpdate on the specific window that is shown.

  if (this->isVisible())
  {
    switch(m_Layout)
    {
    case MIDAS_LAYOUT_AXIAL:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      break;
    case MIDAS_LAYOUT_SAGITTAL:
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      break;
    case MIDAS_LAYOUT_CORONAL:
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      break;
    case MIDAS_LAYOUT_ORTHO:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    case MIDAS_LAYOUT_3H:
    case MIDAS_LAYOUT_3V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      break;
    case MIDAS_LAYOUT_3D:
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    case MIDAS_LAYOUT_COR_SAG_H:
    case MIDAS_LAYOUT_COR_SAG_V:
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
    break;
    case MIDAS_LAYOUT_COR_AX_H:
    case MIDAS_LAYOUT_COR_AX_V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
    break;
    case MIDAS_LAYOUT_SAG_AX_H:
    case MIDAS_LAYOUT_SAG_AX_V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
    break;
    default:
      // die, this should never happen
      assert((m_Layout >= 0 && m_Layout <= 6) || (m_Layout >= 9 && m_Layout <= 14));
      break;
    }
  }
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::IsEnabled() const
{
  return m_IsEnabled;
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::GetDisplay2DCursorsLocally() const
{
  return m_Display2DCursorsLocally;
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::GetDisplay2DCursorsGlobally() const
{
  return m_Display2DCursorsGlobally;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::AreDirectionAnnotationsVisible() const
{
  return m_DirectionAnnotations[0]->GetVisibility()
      && m_DirectionAnnotations[1]->GetVisibility()
      && m_DirectionAnnotations[2]->GetVisibility();
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetDirectionAnnotationsVisible(bool visible)
{
  for (int i = 0; i < 3; ++i)
  {
    m_DirectionAnnotations[i]->SetVisibility(visible);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetShow3DWindowInOrthoView(bool visible)
{
  m_Show3DWindowInOrthoView = visible;
  this->Update3DWindowVisibility();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::GetShow3DWindowInOrthoView() const
{
  return m_Show3DWindowInOrthoView;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::Update3DWindowVisibility()
{
  if (m_DataStorage.IsNotNull())
  {
    mitk::BaseRenderer* axialRenderer = this->mitkWidget1->GetRenderer();

    bool show3DPlanes = false;

    mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
    for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
    {
      if (it->Value().IsNull())
      {
        continue;
      }

      bool visibleIn3DWindow = false;
      if ((m_Layout == MIDAS_LAYOUT_ORTHO && m_Show3DWindowInOrthoView)
          || m_Layout == MIDAS_LAYOUT_3D)
      {
        visibleIn3DWindow = true;
      }

      bool visibleInAxialWindow = false;
      if (it->Value()->GetBoolProperty("visible", visibleInAxialWindow, axialRenderer))
      {
        if (!visibleInAxialWindow)
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


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetVisibility(QmitkRenderWindow* renderWindow, mitk::DataNode* node, bool visible)
{
  if (renderWindow != NULL && node != NULL)
  {
    mitk::BaseRenderer* renderer = renderWindow->GetRenderer();
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


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::ContainsRenderWindow(QmitkRenderWindow* renderWindow) const
{
  return mitkWidget1 == renderWindow
      || mitkWidget2 == renderWindow
      || mitkWidget3 == renderWindow
      || mitkWidget4 == renderWindow;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASStdMultiWidget::GetRenderWindow(vtkRenderWindow* vtkRenderWindow) const
{
  QmitkRenderWindow* renderWindow = 0;
  if (mitkWidget1->GetVtkRenderWindow() == vtkRenderWindow)
  {
    renderWindow = mitkWidget1;
  }
  else if (mitkWidget2->GetVtkRenderWindow() == vtkRenderWindow)
  {
    renderWindow = mitkWidget2;
  }
  else if (mitkWidget3->GetVtkRenderWindow() == vtkRenderWindow)
  {
    renderWindow = mitkWidget3;
  }
  else if (mitkWidget4->GetVtkRenderWindow() == vtkRenderWindow)
  {
    renderWindow = mitkWidget4;
  }
  return renderWindow;
}


//-----------------------------------------------------------------------------
std::vector<QmitkRenderWindow*> QmitkMIDASStdMultiWidget::GetRenderWindows() const
{
  return std::vector<QmitkRenderWindow*>(m_RenderWindows, m_RenderWindows + 4);
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASStdMultiWidget::GetOrientation()
{
  MIDASOrientation result = MIDAS_ORIENTATION_UNKNOWN;
  if (m_Layout == MIDAS_LAYOUT_AXIAL)
  {
    result = MIDAS_ORIENTATION_AXIAL;
  }
  else if (m_Layout == MIDAS_LAYOUT_SAGITTAL)
  {
    result = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (m_Layout == MIDAS_LAYOUT_CORONAL)
  {
    result = MIDAS_ORIENTATION_CORONAL;
  }
  else if (m_Layout == MIDAS_LAYOUT_ORTHO
           || m_Layout == MIDAS_LAYOUT_3H
           || m_Layout == MIDAS_LAYOUT_3V
           || m_Layout == MIDAS_LAYOUT_COR_SAG_H
           || m_Layout == MIDAS_LAYOUT_COR_SAG_V
           || m_Layout == MIDAS_LAYOUT_COR_AX_H
           || m_Layout == MIDAS_LAYOUT_COR_AX_V
           || m_Layout == MIDAS_LAYOUT_SAG_AX_H
           || m_Layout == MIDAS_LAYOUT_SAG_AX_V
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


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::FitToDisplay()
{
  std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  for (unsigned int i = 0; i < renderWindows.size(); i++)
  {
    m_BlockDisplayGeometryEvents = true;
    renderWindows[i]->GetRenderer()->GetDisplayGeometry()->Fit();
    m_BlockDisplayGeometryEvents = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetGeometry(mitk::Geometry3D* geometry)
{
  if (geometry != NULL)
  {
    m_Geometry = geometry;

    // Calculating the voxel size. This is needed for the conversion between the
    // magnification and the scale factors.
    for (int i = 0; i < 3; ++i)
    {
      m_MmPerVx[i] = m_Geometry->GetExtentInMM(i) / m_Geometry->GetExtent(i);
    }

    // Add these annotations the first time we have a real geometry.
    m_CornerAnnotaions[0].cornerText->SetText(0, "Axial");
    m_CornerAnnotaions[1].cornerText->SetText(0, "Sagittal");
    m_CornerAnnotaions[2].cornerText->SetText(0, "Coronal");

    {
      int axialDirection = this->GetSliceUpDirection(MIDAS_ORIENTATION_AXIAL);
      int sagittalDirection = this->GetSliceUpDirection(MIDAS_ORIENTATION_SAGITTAL);
      int coronalDirection = this->GetSliceUpDirection(MIDAS_ORIENTATION_CORONAL);

      // Neurological convention. Note that this is different from the radiological convention.
      m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(0, coronalDirection > 0 ? "A" : "P");
      m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(1, sagittalDirection > 0 ? "L" : "R");
      m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(2, coronalDirection > 0 ? "P" : "A");
      m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(3, sagittalDirection > 0 ? "R" : "L");
      m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(0, axialDirection > 0 ? "I" : "S");
      m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(1, coronalDirection > 0 ? "P" : "A");
      m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(2, axialDirection > 0 ? "S" : "I");
      m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(3, coronalDirection > 0 ? "A" : "P");
      m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(0, axialDirection > 0 ? "I" : "S");
      m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(1, sagittalDirection > 0 ? "L" : "R");
      m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(2, axialDirection > 0 ? "S" : "I");
      m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(3, sagittalDirection > 0 ? "R" : "L");
    }

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

    permutedBoundingBox[0] = static_cast<int>(geometry->GetExtent(dominantAxisRL));
    permutedBoundingBox[1] = static_cast<int>(geometry->GetExtent(dominantAxisAP));
    permutedBoundingBox[2] = static_cast<int>(geometry->GetExtent(dominantAxisSI));

    permutedSpacing[0] = geometry->GetSpacing()[permutedAxes[0]];
    permutedSpacing[1] = geometry->GetSpacing()[permutedAxes[1]];
    permutedSpacing[2] = geometry->GetSpacing()[permutedAxes[2]];

    mitk::AffineTransform3D::MatrixType::InternalMatrixType permutedMatrix;
    permutedMatrix.set_identity();

    // permutedMatrix(column) = inverseTransformMatrix(row) * flippedAxes
    permutedMatrix[0][0] = inverseTransformMatrix[permutedAxes[0]][0] * flippedAxes[0];
    permutedMatrix[1][0] = inverseTransformMatrix[permutedAxes[0]][1] * flippedAxes[0];
    permutedMatrix[2][0] = inverseTransformMatrix[permutedAxes[0]][2] * flippedAxes[0];
    permutedMatrix[0][1] = inverseTransformMatrix[permutedAxes[1]][0] * flippedAxes[1];
    permutedMatrix[1][1] = inverseTransformMatrix[permutedAxes[1]][1] * flippedAxes[1];
    permutedMatrix[2][1] = inverseTransformMatrix[permutedAxes[1]][2] * flippedAxes[1];
    permutedMatrix[0][2] = inverseTransformMatrix[permutedAxes[2]][0] * flippedAxes[2];
    permutedMatrix[1][2] = inverseTransformMatrix[permutedAxes[2]][1] * flippedAxes[2];
    permutedMatrix[2][2] = inverseTransformMatrix[permutedAxes[2]][2] * flippedAxes[2];

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

    std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
    for (unsigned int i = 0; i < renderWindows.size(); i++)
    {
      QmitkRenderWindow* renderWindow = renderWindows[i];
      mitk::BaseRenderer* renderer = renderWindow->GetRenderer();
      int id = renderer->GetMapperID();

      // Get access to slice navigation controller, as this sorts out most of the process.
      mitk::SliceNavigationController* sliceNavigationController = renderer->GetSliceNavigationController();
      sliceNavigationController->SetViewDirectionToDefault();

      // Get the view/orientation flags.
      mitk::SliceNavigationController::ViewDirection viewDirection = const_cast<const mitk::SliceNavigationController*>(sliceNavigationController)->GetViewDirection();

      if (i < 3)
      {

        mitk::Point3D originInVx;
        mitk::Point3D originInMm;
        mitk::Point3D originOfSlice;
        mitk::VnlVector rightDV(3);
        mitk::VnlVector bottomDV(3);
        mitk::VnlVector normal(3);
        int width = 1;
        int height = 1;
        mitk::ScalarType viewSpacing = 1;
        unsigned int slices = 1;
        bool isFlipped;

        float voxelOffset = 0;
        if(geometry->GetImageGeometry())
        {
          voxelOffset = 0.5;
        }

        originInVx[0] = 0;
        originInVx[1] = 0;
        originInVx[2] = 0;

        if (flippedAxes[0] < 0)
        {
          MITK_DEBUG << "Matt, flippedAxis[0] < 0, so flipping axis " << permutedAxes[0] << std::endl;
          originInVx[permutedAxes[0]] = geometry->GetExtent(permutedAxes[0]) - 1;
        }
        if (flippedAxes[1] < 0)
        {
          MITK_DEBUG << "Matt, flippedAxis[1] < 0, so flipping axis " << permutedAxes[1] << std::endl;
          originInVx[permutedAxes[1]] = geometry->GetExtent(permutedAxes[1]) - 1;
        }
        if (flippedAxes[2] < 0)
        {
          MITK_DEBUG << "Matt, flippedAxis[2] < 0, so flipping axis " << permutedAxes[2] << std::endl;
          originInVx[permutedAxes[2]] = geometry->GetExtent(permutedAxes[2]) - 1;
        }

        geometry->IndexToWorld(originInVx, originInMm);

        MITK_DEBUG << "Matt, originInVx: " << originInVx << ", originInMm: " << originInMm << std::endl;

        // Setting up the width, height, axis orientation.
        switch (viewDirection)
        {
        case mitk::SliceNavigationController::Sagittal:
          width  = permutedBoundingBox[1];
          height = permutedBoundingBox[2];
          originOfSlice[0] = originInMm[0];
          originOfSlice[1] = originInMm[1] - voxelOffset * permutedSpacing[1];
          originOfSlice[2] = originInMm[2] - voxelOffset * permutedSpacing[2];
          rightDV[0] = permutedSpacing[0] * permutedMatrix[0][1];
          rightDV[1] = permutedSpacing[1] * permutedMatrix[1][1];
          rightDV[2] = permutedSpacing[2] * permutedMatrix[2][1];
          bottomDV[0] = permutedSpacing[0] * permutedMatrix[0][2];
          bottomDV[1] = permutedSpacing[1] * permutedMatrix[1][2];
          bottomDV[2] = permutedSpacing[2] * permutedMatrix[2][2];
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
          originOfSlice[0] = originInMm[0] - voxelOffset * permutedSpacing[0];
          originOfSlice[1] = originInMm[1];
          originOfSlice[2] = originInMm[2] - voxelOffset * permutedSpacing[2];
          rightDV[0] = permutedSpacing[0] * permutedMatrix[0][0];
          rightDV[1] = permutedSpacing[1] * permutedMatrix[1][0];
          rightDV[2] = permutedSpacing[2] * permutedMatrix[2][0];
          bottomDV[0] = permutedSpacing[0] * permutedMatrix[0][2];
          bottomDV[1] = permutedSpacing[1] * permutedMatrix[1][2];
          bottomDV[2] = permutedSpacing[2] * permutedMatrix[2][2];
          normal[0] = permutedMatrix[0][1];
          normal[1] = permutedMatrix[1][1];
          normal[2] = permutedMatrix[2][1];
          viewSpacing = permutedSpacing[1];
          slices = permutedBoundingBox[1];
          isFlipped = true;
          break;
        default:
          width  = permutedBoundingBox[0];
          height = permutedBoundingBox[1];
          originOfSlice[0] = originInMm[0] + permutedBoundingBox[0] * permutedSpacing[0] * permutedMatrix[0][1] - voxelOffset * permutedSpacing[0];
          originOfSlice[1] = originInMm[1] + permutedBoundingBox[1] * permutedSpacing[1] * permutedMatrix[1][1] + voxelOffset * permutedSpacing[1];
          originOfSlice[2] = originInMm[2] + permutedBoundingBox[2] * permutedSpacing[2] * permutedMatrix[2][1];
          rightDV[0] = permutedSpacing[0] * permutedMatrix[0][0];
          rightDV[1] = permutedSpacing[1] * permutedMatrix[1][0];
          rightDV[2] = permutedSpacing[2] * permutedMatrix[2][0];
          bottomDV[0] = -1.0 * permutedSpacing[0] * permutedMatrix[0][1];
          bottomDV[1] = -1.0 * permutedSpacing[1] * permutedMatrix[1][1];
          bottomDV[2] = -1.0 * permutedSpacing[2] * permutedMatrix[2][1];
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
        mitk::ScalarType bounds[6]= { 0, static_cast<float>(width), 0, static_cast<float>(height), 0, 1 };

        // A SlicedGeometry3D is initialised from a 2D PlaneGeometry, plus the number of slices.
        mitk::PlaneGeometry::Pointer planeGeometry = mitk::PlaneGeometry::New();
        planeGeometry->SetIdentity();
        planeGeometry->SetImageGeometry(false);
        planeGeometry->SetBounds(bounds);
        planeGeometry->SetOrigin(originOfSlice);
        planeGeometry->SetMatrixByVectors(rightDV, bottomDV, normal.two_norm());

        for (unsigned int j = 0; j < numberOfTimeSteps; j++)
        {
          // Then we create the SlicedGeometry3D from an initial plane, and a given number of slices.
          mitk::SlicedGeometry3D::Pointer slicedGeometry = mitk::SlicedGeometry3D::New();
          slicedGeometry->SetIdentity();
          slicedGeometry->SetReferenceGeometry(geometry);
          slicedGeometry->SetImageGeometry(false);
          slicedGeometry->InitializeEvenlySpaced(planeGeometry, viewSpacing, slices, isFlipped);

          if (inputTimeSlicedGeometry.IsNotNull())
          {
            slicedGeometry->SetTimeBounds(inputTimeSlicedGeometry->GetGeometry3D(j)->GetTimeBounds());
          }
          createdTimeSlicedGeometry->SetGeometry3D(slicedGeometry, j);
        }
        createdTimeSlicedGeometry->UpdateInformation();

        MITK_DEBUG << "Matt - final geometry=" << createdTimeSlicedGeometry << std::endl;
        MITK_DEBUG << "Matt - final geometry origin=" << createdTimeSlicedGeometry->GetOrigin() << std::endl;
        MITK_DEBUG << "Matt - final geometry center=" << createdTimeSlicedGeometry->GetCenter() << std::endl;
        for (int j = 0; j < 8; j++)
        {
          MITK_DEBUG << "Matt - final geometry j=" << j << ", p=" << createdTimeSlicedGeometry->GetCornerPoint(j) << std::endl;
        }
        sliceNavigationController->SetInputWorldGeometry(createdTimeSlicedGeometry);
        sliceNavigationController->Update(mitk::SliceNavigationController::Original, true, true, false);
        sliceNavigationController->SetViewDirection(viewDirection);

        // For 2D mappers only, set to middle slice (the 3D mapper simply follows by event listening).
        if (id == 1)
        {
          // Now geometry is established, set to middle slice.
          int sliceNumber = (int)((sliceNavigationController->GetSlice()->GetSteps() - 1) / 2.0);
          sliceNavigationController->GetSlice()->SetPos(sliceNumber);
        }

        // Now geometry is established, get the display geometry to fit the picture to the window.
        renderer->GetDisplayGeometry()->SetConstrainZoomingAndPanning(false);
        renderer->GetDisplayGeometry()->Fit();

      } // if window < 3
    }
  }
  else
  {
    // Probably not necessary, but we restore the default voxel size if there is no geometry.
    m_MmPerVx[0] = 1.0;
    m_MmPerVx[1] = 1.0;
    m_MmPerVx[2] = 1.0;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetLayout(MIDASLayout layout)
{
  m_BlockDisplayGeometryEvents = true;

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

  QmitkStdMultiWidgetLayout = new QHBoxLayout(this);
  QmitkStdMultiWidgetLayout->setContentsMargins(0, 0, 0, 0);
  QmitkStdMultiWidgetLayout->setSpacing(0);

  if (layout == MIDAS_LAYOUT_3H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 2);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 0, 3);  // 3D:       off
  }
  else if (layout == MIDAS_LAYOUT_3V)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 2, 0);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 3, 0);  // 3D:       off
  }
  else if (layout == MIDAS_LAYOUT_COR_SAG_H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);  // axial:    off
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (layout == MIDAS_LAYOUT_COR_SAG_V)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 1);  // axial:    off
    m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (layout == MIDAS_LAYOUT_COR_AX_H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 1);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);  // sagittal: off
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (layout == MIDAS_LAYOUT_COR_AX_V)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: off
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (layout == MIDAS_LAYOUT_SAG_AX_H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 1);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 0);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 1, 0);  // coronal:  off
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (layout == MIDAS_LAYOUT_SAG_AX_V)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 0);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 1);  // coronal:  off
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       on
  }

  QmitkStdMultiWidgetLayout->addLayout(m_GridLayout);

  switch(layout)
  {
  case MIDAS_LAYOUT_AXIAL:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_LAYOUT_SAGITTAL:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->hide();
    this->mitkWidget2->setFocus();
    break;
  case MIDAS_LAYOUT_CORONAL:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget3->setFocus();
    break;
  case MIDAS_LAYOUT_ORTHO:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->show();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_LAYOUT_3H:
  case MIDAS_LAYOUT_3V:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_LAYOUT_3D:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->show();
    this->mitkWidget4->setFocus();
    break;
  case MIDAS_LAYOUT_COR_SAG_H:
  case MIDAS_LAYOUT_COR_SAG_V:
    this->mitkWidget1Container->hide();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget2->setFocus();
    break;
  case MIDAS_LAYOUT_COR_AX_H:
  case MIDAS_LAYOUT_COR_AX_V:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->hide();
    this->mitkWidget3Container->show();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  case MIDAS_LAYOUT_SAG_AX_H:
  case MIDAS_LAYOUT_SAG_AX_V:
    this->mitkWidget1Container->show();
    this->mitkWidget2Container->show();
    this->mitkWidget3Container->hide();
    this->mitkWidget4Container->hide();
    this->mitkWidget1->setFocus();
    break;
  default:
    // die, this should never happen
    assert((m_Layout >= 0 && m_Layout <= 6) || (m_Layout >= 9 && m_Layout <= 14));
    break;
  }

  m_Layout = layout;

  this->Update3DWindowVisibility();
  m_GridLayout->activate();

  m_BlockDisplayGeometryEvents = false;
}


//-----------------------------------------------------------------------------
MIDASLayout QmitkMIDASStdMultiWidget::GetLayout() const
{
  return m_Layout;
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* QmitkMIDASStdMultiWidget::GetSliceNavigationController(MIDASOrientation orientation) const
{
  mitk::SliceNavigationController* result = NULL;
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


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASStdMultiWidget::GetMaxSliceIndex(MIDASOrientation orientation) const
{
  unsigned int result = 0;

  mitk::SliceNavigationController* snc = this->GetSliceNavigationController(orientation);
  assert(snc);

  if (snc->GetSlice() != NULL && snc->GetSlice()->GetSteps() > 0)
  {
    result = snc->GetSlice()->GetSteps() - 1;
  }

  return result;
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASStdMultiWidget::GetMaxTimeStep() const
{
  unsigned int result = 0;

  mitk::SliceNavigationController* snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  assert(snc);

  if (snc->GetTime() != NULL && snc->GetTime()->GetSteps() >= 1)
  {
    result = snc->GetTime()->GetSteps() -1;
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::OnOriginChanged(QmitkRenderWindow* renderWindow, bool beingPanned)
{
  if (m_Geometry && !m_BlockDisplayGeometryEvents)
  {
    mitk::Vector2D cursorPosition = this->GetCursorPosition(renderWindow);

    // cursor[0] <-> axial[0] <-> coronal[0]
    // cursor[1] <-> axial[1] <-> 1.0 - sagittal[0]
    // cursor[2] <-> sagittal[1] <-> coronal[1]

    if (renderWindow == m_RenderWindows[MIDAS_ORIENTATION_AXIAL])
    {
      m_CursorPosition[0] = cursorPosition[0];
      m_CursorPosition[1] = cursorPosition[1];
    }
    else if (renderWindow == m_RenderWindows[MIDAS_ORIENTATION_SAGITTAL])
    {
      m_CursorPosition[1] = 1.0 - cursorPosition[0];
      m_CursorPosition[2] = cursorPosition[1];
    }
    else if (renderWindow == m_RenderWindows[MIDAS_ORIENTATION_CORONAL])
    {
      m_CursorPosition[0] = cursorPosition[0];
      m_CursorPosition[2] = cursorPosition[1];
    }

    if (beingPanned && this->AreCursorPositionsBound())
    {
      // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
      for (int i = 0; i < 3; ++i)
      {
        QmitkRenderWindow* otherRenderWindow = m_RenderWindows[i];
        if (otherRenderWindow != renderWindow)
        {
          mitk::Vector2D origin = this->ComputeOriginFromCursorPosition(otherRenderWindow, m_CursorPosition);
          this->SetOrigin(otherRenderWindow, origin);
        }
      }
    }

    this->RequestUpdate();
    if (beingPanned)
    {
      emit CursorPositionChanged(m_CursorPosition);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetOrigin(QmitkRenderWindow* renderWindow, const mitk::Vector2D& origin)
{
  mitk::Vector2D originInMm;

  mitk::DisplayGeometry* displayGeometry = renderWindow->GetRenderer()->GetDisplayGeometry();

  displayGeometry->DisplayToWorld(origin, originInMm);

  m_BlockDisplayGeometryEvents = true;
  displayGeometry->SetOriginInMM(originInMm);
  m_BlockDisplayGeometryEvents = false;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::OnScaleFactorChanged(QmitkRenderWindow* renderWindow, double scaleFactor)
{
  if (m_Geometry && !m_BlockDisplayGeometryEvents)
  {
    double magnification = this->GetMagnification(renderWindow);

    if (scaleFactor != m_ScaleFactor)
    {
      if (this->AreScaleFactorsBound())
      {
        // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
        for (int i = 0; i < 3; ++i)
        {
          QmitkRenderWindow* otherRenderWindow = m_RenderWindows[i];
          if (otherRenderWindow != renderWindow && otherRenderWindow->isVisible())
          {
            this->SetScaleFactor(otherRenderWindow, scaleFactor);
          }
        }
      }

      m_Magnification = magnification;
      m_ScaleFactor = scaleFactor;
      this->RequestUpdate();
      emit ScaleFactorChanged(scaleFactor);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::OnSelectedPositionChanged(MIDASOrientation orientation)
{
  if (m_Geometry != NULL && !m_BlockDisplayGeometryEvents && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    int sliceIndex = 0;
    mitk::Index3D selectedPositionInVx;
    mitk::Point3D selectedPosition = this->GetSelectedPosition();
    int axis = m_OrientationToAxisMap[orientation];

    m_Geometry->WorldToIndex(selectedPosition, selectedPositionInVx);
    sliceIndex = selectedPositionInVx[axis];

    emit SelectedPositionChanged(m_RenderWindows[orientation], sliceIndex);

    // cursor[0] <-> axial[0] <-> coronal[0]
    // cursor[1] <-> axial[1] <-> 1.0 - sagittal[0]
    // cursor[2] <-> sagittal[1] <-> coronal[1]

    mitk::Vector2D cursorPositionOnAxialDisplay = this->GetCursorPosition(m_RenderWindows[MIDAS_ORIENTATION_AXIAL]);
    mitk::Vector2D cursorPositionOnSagittalDisplay = this->GetCursorPosition(m_RenderWindows[MIDAS_ORIENTATION_SAGITTAL]);
    mitk::Vector2D cursorPositionOnCoronalDisplay = this->GetCursorPosition(m_RenderWindows[MIDAS_ORIENTATION_CORONAL]);

    QmitkRenderWindow* renderWindow = this->GetSelectedRenderWindow();
    if (renderWindow == this->mitkWidget1)
    {
      if (orientation == MIDAS_ORIENTATION_AXIAL)
      {
        m_CursorPosition[2] = cursorPositionOnCoronalDisplay[1];
      }
      else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        m_CursorPosition[0] = cursorPositionOnAxialDisplay[0];
      }
      else// if (orientation == MIDAS_ORIENTATION_CORONAL)
      {
        m_CursorPosition[1] = cursorPositionOnAxialDisplay[1];
      }
    }
    else if (renderWindow == this->mitkWidget2)
    {
      if (orientation == MIDAS_ORIENTATION_AXIAL)
      {
        m_CursorPosition[2] = cursorPositionOnSagittalDisplay[1];
      }
      else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        m_CursorPosition[0] = cursorPositionOnAxialDisplay[0];
      }
      else// if (orientation == MIDAS_ORIENTATION_CORONAL)
      {
        m_CursorPosition[1] = 1.0 - cursorPositionOnSagittalDisplay[0];
      }
    }
    else// if (renderWindow == this->mitkWidget3)
    {
      if (orientation == MIDAS_ORIENTATION_AXIAL)
      {
        m_CursorPosition[2] = cursorPositionOnCoronalDisplay[1];
      }
      else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        m_CursorPosition[0] = cursorPositionOnCoronalDisplay[0];
      }
      else// if (orientation == MIDAS_ORIENTATION_CORONAL)
      {
        m_CursorPosition[1] = cursorPositionOnAxialDisplay[1];
      }
    }

    if (this->AreCursorPositionsBound() && !this->AreScaleFactorsBound())
    {
      // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
      for (int i = 0; i < 3; ++i)
      {
        QmitkRenderWindow* otherRenderWindow = m_RenderWindows[i];
        if (otherRenderWindow != renderWindow)
        {
          mitk::Vector2D origin = this->ComputeOriginFromCursorPosition(otherRenderWindow, m_CursorPosition);
          this->SetOrigin(otherRenderWindow, origin);
        }
      }
      this->RequestUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetSliceIndex(MIDASOrientation orientation, unsigned int sliceIndex)
{
  const mitk::Geometry3D* geometry = m_Geometry;
  if (geometry != NULL)
  {
    mitk::Index3D selectedPositionInVx;
    mitk::Point3D selectedPosition = this->GetSelectedPosition();

    geometry->WorldToIndex(selectedPosition, selectedPositionInVx);

    int axis = m_OrientationToAxisMap[orientation];
    selectedPositionInVx[axis] = sliceIndex;

    mitk::Point3D tmp;
    tmp[0] = selectedPositionInVx[0];
    tmp[1] = selectedPositionInVx[1];
    tmp[2] = selectedPositionInVx[2];

    geometry->IndexToWorld(tmp, selectedPosition);

    // Does not work, as it relies on the StateMachine event broadcasting mechanism,
    // and if the widget is not listening, then it goes unnoticed.
    //this->MoveCrossToPosition(selectedPosition);

    // This however, directly forces the SNC to the right place.
    mitkWidget1->GetSliceNavigationController()->SelectSliceByPoint(selectedPosition);
    mitkWidget2->GetSliceNavigationController()->SelectSliceByPoint(selectedPosition);
    mitkWidget3->GetSliceNavigationController()->SelectSliceByPoint(selectedPosition);
  }
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASStdMultiWidget::GetSliceIndex(MIDASOrientation orientation) const
{
  int sliceIndex = 0;

  if (m_Geometry != NULL)
  {
    mitk::Index3D selectedPositionInVx;
    const mitk::Point3D selectedPosition = this->GetSelectedPosition();

    m_Geometry->WorldToIndex(selectedPosition, selectedPositionInVx);

    int axis = m_OrientationToAxisMap[orientation];
    sliceIndex = selectedPositionInVx[axis];
  }

  return sliceIndex;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetTimeStep(unsigned int timeStep)
{
  mitk::SliceNavigationController* snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  snc->GetTime()->SetPos(timeStep);

  snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_SAGITTAL);
  snc->GetTime()->SetPos(timeStep);

  snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_CORONAL);
  snc->GetTime()->SetPos(timeStep);
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASStdMultiWidget::GetTimeStep() const
{
  mitk::SliceNavigationController* snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  assert(snc);

  return snc->GetTime()->GetPos();
}


//-----------------------------------------------------------------------------
const mitk::Point3D QmitkMIDASStdMultiWidget::GetSelectedPosition() const
{
  return this->GetCrossPosition();
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition)
{
  mitk::SliceNavigationController* snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_AXIAL);
  // Check if the slice navigation controller has a valid geometry.
  m_BlockDisplayGeometryEvents = true;
  if (snc->GetCreatedWorldGeometry())
  {
    snc->SelectSliceByPoint(selectedPosition);

    snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_SAGITTAL);
    snc->SelectSliceByPoint(selectedPosition);

    snc = this->GetSliceNavigationController(MIDAS_ORIENTATION_CORONAL);
    snc->SelectSliceByPoint(selectedPosition);
  }
  m_BlockDisplayGeometryEvents = false;
}


//-----------------------------------------------------------------------------
const mitk::Vector3D& QmitkMIDASStdMultiWidget::GetCursorPosition() const
{
  return m_CursorPosition;
}


//-----------------------------------------------------------------------------
const mitk::Vector2D QmitkMIDASStdMultiWidget::GetCursorPosition(QmitkRenderWindow* renderWindow) const
{
  const mitk::Point3D selectedPosition = this->GetSelectedPosition();

  mitk::BaseRenderer* renderer = renderWindow->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  mitk::Vector2D displaySize = displayGeometry->GetSizeInDisplayUnits();

  mitk::Vector2D cursorPosition;

  mitk::Point2D cursorPositionInMm;
  mitk::Point2D cursorPositionInPx;

  displayGeometry->Map(selectedPosition, cursorPositionInMm);
  displayGeometry->WorldToDisplay(cursorPositionInMm, cursorPositionInPx);

  cursorPosition[0] = cursorPositionInPx[0] / displaySize[0];
  cursorPosition[1] = cursorPositionInPx[1] / displaySize[1];

  return cursorPosition;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetCursorPosition(const mitk::Vector3D& cursorPosition)
{
  m_CursorPosition = cursorPosition;

  bool hasChanged = false;

  if (this->AreCursorPositionsBound())
  {
    // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
    for (int i = 0; i < 3; ++i)
    {
      QmitkRenderWindow* renderWindow = m_RenderWindows[i];
      if (renderWindow->isVisible())
      {
        mitk::Vector2D origin = this->ComputeOriginFromCursorPosition(renderWindow, cursorPosition);
        this->SetOrigin(renderWindow, origin);
        hasChanged = true;
      }
    }
  }
  else
  {
    QmitkRenderWindow* renderWindow = this->GetSelectedRenderWindow();
    if (renderWindow && renderWindow->isVisible() && renderWindow != m_RenderWindows[3])
    {
      mitk::Vector2D origin = this->ComputeOriginFromCursorPosition(renderWindow, cursorPosition);
      this->SetOrigin(renderWindow, origin);
      hasChanged = true;
    }
  }

  if (hasChanged)
  {
    this->RequestUpdate();
  }
}


//-----------------------------------------------------------------------------
mitk::Vector2D QmitkMIDASStdMultiWidget::ComputeOriginFromCursorPosition(QmitkRenderWindow* renderWindow, const mitk::Vector3D& cursorPosition)
{
  mitk::Vector2D cursorPosition2D;
  if (renderWindow == m_RenderWindows[MIDAS_ORIENTATION_AXIAL])
  {
    cursorPosition2D[0] = cursorPosition[0];
    cursorPosition2D[1] = cursorPosition[1];
  }
  else if (renderWindow == m_RenderWindows[MIDAS_ORIENTATION_SAGITTAL])
  {
    cursorPosition2D[0] = 1.0 - cursorPosition[1];
    cursorPosition2D[1] = cursorPosition[2];
  }
  else if (renderWindow == m_RenderWindows[MIDAS_ORIENTATION_CORONAL])
  {
    cursorPosition2D[0] = cursorPosition[0];
    cursorPosition2D[1] = cursorPosition[2];
  }

  return this->ComputeOriginFromCursorPosition(renderWindow, cursorPosition2D);
}


//-----------------------------------------------------------------------------
mitk::Vector2D QmitkMIDASStdMultiWidget::ComputeOriginFromCursorPosition(QmitkRenderWindow* renderWindow, const mitk::Vector2D& cursorPosition)
{
  mitk::DisplayGeometry* displayGeometry = renderWindow->GetRenderer()->GetDisplayGeometry();
  mitk::Point3D selectedPosition = this->GetSelectedPosition();

  mitk::Vector2D displaySize = displayGeometry->GetSizeInDisplayUnits();

  mitk::Vector2D cursorPositionInPx;
  cursorPositionInPx[0] = cursorPosition[0] * displaySize[0];
  cursorPositionInPx[1] = cursorPosition[1] * displaySize[1];

  mitk::Point2D selectedPosition2D;
  displayGeometry->Map(selectedPosition, selectedPosition2D);
  double scaleFactor = displayGeometry->GetScaleFactorMMPerDisplayUnit();
  mitk::Vector2D selectedPositionInPx;
  selectedPositionInPx[0] = selectedPosition2D[0] / scaleFactor;
  selectedPositionInPx[1] = selectedPosition2D[1] / scaleFactor;

  return selectedPositionInPx - cursorPositionInPx;
}


//-----------------------------------------------------------------------------
double QmitkMIDASStdMultiWidget::GetScaleFactor() const
{
  return m_ScaleFactor;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetScaleFactor(double scaleFactor)
{
  QmitkRenderWindow* selectedRenderWindow = this->GetSelectedRenderWindow();

  if (selectedRenderWindow)
  {
    this->SetScaleFactor(selectedRenderWindow, scaleFactor);

    m_ScaleFactor = scaleFactor;
    m_Magnification = this->GetMagnification(selectedRenderWindow);
    this->RequestUpdate();
  }

  if (this->AreScaleFactorsBound())
  {
    // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
    for (int i = 0; i < 3; ++i)
    {
      QmitkRenderWindow* otherRenderWindow = m_RenderWindows[i];
      if (otherRenderWindow != selectedRenderWindow && otherRenderWindow->isVisible())
      {
        this->SetScaleFactor(otherRenderWindow, scaleFactor);
      }
    }
  }
}


//-----------------------------------------------------------------------------
double QmitkMIDASStdMultiWidget::GetMagnification() const
{
  return this->GetMagnification(this->GetSelectedRenderWindow());
//  return m_Magnification;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetMagnification(double magnification)
{
  this->SetMagnification(this->GetSelectedRenderWindow(), magnification);
}


//-----------------------------------------------------------------------------
int QmitkMIDASStdMultiWidget::GetDominantAxis(MIDASOrientation orientation) const
{
  int axisWithLongerSide = 0;
  if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    axisWithLongerSide = m_MmPerVx[0] > m_MmPerVx[1] ? 0 : 1;
  }
  else if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    axisWithLongerSide = m_MmPerVx[1] > m_MmPerVx[2] ? 1 : 2;
  }
  else // if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    axisWithLongerSide = m_MmPerVx[0] > m_MmPerVx[2] ? 0 : 2;
  }
  return axisWithLongerSide;
}


//-----------------------------------------------------------------------------
double QmitkMIDASStdMultiWidget::GetMagnification(QmitkRenderWindow* renderWindow) const
{
  double scaleFactorMmPerPx = this->GetScaleFactor(renderWindow);

  MIDASOrientation orientation = this->GetOrientation(renderWindow);
  int dominantAxis = this->GetDominantAxis(orientation);

  // Note that we take the scale factor from the first axis always, consequently.
  // Other strategies could be followed as well, e.g. taking the axes with the smallest
  // voxel size.
  double scaleFactorPxPerVx = m_MmPerVx[dominantAxis] / scaleFactorMmPerPx;

  // Finally, we calculate the magnification from the scale factor.
  double magnification = scaleFactorPxPerVx - 1.0;
  if (magnification < 0.0)
  {
    magnification /= scaleFactorPxPerVx;
  }

  return magnification;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetMagnification(QmitkRenderWindow* renderWindow, double magnification)
{
  MIDASOrientation orientation = this->GetOrientation(renderWindow);
  int dominantAxis = this->GetDominantAxis(orientation);

  mitk::Vector3D scaleFactors = this->ComputeScaleFactors(magnification);
  double scaleFactor = scaleFactors[dominantAxis];

  this->SetScaleFactor(scaleFactor);
}


//-----------------------------------------------------------------------------
mitk::Vector3D QmitkMIDASStdMultiWidget::ComputeScaleFactors(double magnification)
{
  double scaleFactorVxPerPx;
  if (magnification >= 0.0)
  {
    scaleFactorVxPerPx = 1.0 / (magnification + 1.0);
  }
  else
  {
    scaleFactorVxPerPx = -magnification + 1.0;
  }

  return m_MmPerVx * scaleFactorVxPerPx;
}


//-----------------------------------------------------------------------------
double QmitkMIDASStdMultiWidget::GetScaleFactor(QmitkRenderWindow* renderWindow) const
{
  if (renderWindow == 0)
  {
    return 0.0;
  }

  mitk::DisplayGeometry* displayGeometry = renderWindow->GetRenderer()->GetDisplayGeometry();
  return displayGeometry->GetScaleFactorMMPerDisplayUnit();
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetScaleFactor(QmitkRenderWindow* renderWindow, double scaleFactor)
{
  if (renderWindow != NULL)
  {
    mitk::DisplayGeometry* displayGeometry = renderWindow->GetRenderer()->GetDisplayGeometry();

    const mitk::Point3D& selectedPosition = this->GetSelectedPosition();

    mitk::Point2D focusInMm;
    mitk::Point2D focusInPx;

    displayGeometry->Map(selectedPosition, focusInMm);
    displayGeometry->WorldToDisplay(focusInMm, focusInPx);

    double previousScaleFactor = displayGeometry->GetScaleFactorMMPerDisplayUnit();
    m_BlockDisplayGeometryEvents = true;
    if (displayGeometry->SetScaleFactor(scaleFactor))
    {
      mitk::Vector2D originInMm = displayGeometry->GetOriginInMM();
      displayGeometry->SetOriginInMM(originInMm - focusInPx.GetVectorFromOrigin() * (scaleFactor - previousScaleFactor));
    }
    m_BlockDisplayGeometryEvents = false;
  }
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASStdMultiWidget::GetRenderWindow(const MIDASOrientation& orientation) const
{
  QmitkRenderWindow* renderWindow;
  if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    renderWindow = this->GetRenderWindow1();
  }
  else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    renderWindow = this->GetRenderWindow2();
  }
  else if (orientation == MIDAS_ORIENTATION_CORONAL)
  {
    renderWindow = this->GetRenderWindow3();
  }
//  else if (orientation == MIDAS_ORIENTATION_UNKNOWN)
//  {
//    renderWindow = this->GetRenderWindow4();
//  }
  else
  {
    renderWindow = 0;
  }
  return renderWindow;
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASStdMultiWidget::GetOrientation(const QmitkRenderWindow* renderWindow) const
{
  MIDASOrientation orientation;
  if (renderWindow == this->GetRenderWindow1())
  {
    orientation = MIDAS_ORIENTATION_AXIAL;
  }
  else if (renderWindow == this->GetRenderWindow2())
  {
    orientation = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (renderWindow == this->GetRenderWindow3())
  {
    orientation = MIDAS_ORIENTATION_CORONAL;
  }
  else
  {
    orientation = MIDAS_ORIENTATION_UNKNOWN;
  }
  return orientation;
}


//-----------------------------------------------------------------------------
int QmitkMIDASStdMultiWidget::GetSliceUpDirection(MIDASOrientation orientation) const
{
  int result = 0;
  if (m_Geometry != NULL)
  {
    result = mitk::GetUpDirection(m_Geometry, orientation);
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetDisplayInteractionsEnabled(bool enabled)
{
  if (enabled == this->AreDisplayInteractionsEnabled())
  {
    // Already enabled/disabled.
    return;
  }

  if (enabled)
  {
    std::vector<mitk::BaseRenderer*> renderers(4);
    for (unsigned i = 0; i < 4; ++i)
    {
      renderers[i] = m_RenderWindows[i]->GetRenderer();
    }

    // Here we create our own display interactor...
    m_DisplayInteractor = mitk::MIDASDisplayInteractor::New(renderers);

    us::Module* niftkDnDDisplayModule = us::ModuleRegistry::GetModule("niftkDnDDisplay");
    m_DisplayInteractor->LoadStateMachine("DisplayInteraction.xml", niftkDnDDisplayModule);
    m_DisplayInteractor->SetEventConfig("DisplayConfigMITK.xml");

    // ... and register it as listener via the micro services.
    us::ServiceProperties props;
    props["name"] = std::string("DisplayInteractor");

    us::ModuleContext* moduleContext = us::GetModuleContext();
    m_DisplayInteractorService = moduleContext->RegisterService<mitk::InteractionEventObserver>(m_DisplayInteractor.GetPointer(), props);
  }
  else
  {
    // Unregister the display interactor service.
    m_DisplayInteractorService.Unregister();
    // Release the display interactor to let it be desctructed.
    m_DisplayInteractor = 0;
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::AreDisplayInteractionsEnabled() const
{
  return m_DisplayInteractor.IsNotNull();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::AreCursorPositionsBound() const
{
  return m_CursorPositionsAreBound;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetCursorPositionsBound(bool bound)
{
  if (bound == this->AreCursorPositionsBound())
  {
    // Already bound/unbound.
    return;
  }

  m_CursorPositionsAreBound = bound;

  if (bound)
  {
    this->SetCursorPosition(this->GetCursorPosition());
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASStdMultiWidget::AreScaleFactorsBound() const
{
  return m_ScaleFactorsAreBound;
}


//-----------------------------------------------------------------------------
void QmitkMIDASStdMultiWidget::SetScaleFactorsBound(bool bound)
{
  if (bound == this->AreScaleFactorsBound())
  {
    // Already bound/unbound.
    return;
  }

  m_ScaleFactorsAreBound = bound;

  if (bound)
  {
    this->SetScaleFactor(this->GetScaleFactor());
  }
}
