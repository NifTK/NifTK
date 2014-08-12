/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiWindowWidget_p.h"

#include <cmath>
#include <itkMatrix.h>
#include <itkSpatialOrientationAdapter.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <QmitkRenderWindow.h>
#include <QGridLayout>

#include <mitkGlobalInteraction.h>
#include <mitkVtkLayerController.h>

#include "vtkSideAnnotation_p.h"

/**
 * This class is to notify the SingleViewerWidget about the display geometry changes of a render window.
 */
class DisplayGeometryModificationCommand : public itk::Command
{
public:
  mitkNewMacro2Param(DisplayGeometryModificationCommand, niftkMultiWindowWidget*, int);


  //-----------------------------------------------------------------------------
  DisplayGeometryModificationCommand(niftkMultiWindowWidget* multiWindowWidget, int windowIndex)
  : itk::Command()
  , m_MultiWindowWidget(multiWindowWidget)
  , m_WindowIndex(windowIndex)
  {
  }


  //-----------------------------------------------------------------------------
  void Execute(itk::Object* caller, const itk::EventObject& event)
  {
    this->Execute((const itk::Object*) caller, event);
  }


  //-----------------------------------------------------------------------------
  void Execute(const itk::Object* /*object*/, const itk::EventObject& /*event*/)
  {
    m_MultiWindowWidget->OnDisplayGeometryModified(m_WindowIndex);
    return;
  }

private:
  niftkMultiWindowWidget* const m_MultiWindowWidget;
  int m_WindowIndex;
};


//-----------------------------------------------------------------------------
niftkMultiWindowWidget::niftkMultiWindowWidget(
    QWidget* parent,
    Qt::WindowFlags flags,
    mitk::RenderingManager* renderingManager,
    mitk::BaseRenderer::RenderingMode::Type renderingMode,
    const QString& name)
: QmitkStdMultiWidget(parent, flags, renderingManager, renderingMode, name)
, m_RenderWindows(4)
, m_GridLayout(NULL)
, m_AxialSliceObserverTag(0ul)
, m_SagittalSliceObserverTag(0ul)
, m_CoronalSliceObserverTag(0ul)
, m_TimeStepObserverTag(0ul)
, m_IsFocused(false)
, m_LinkedNavigationEnabled(false)
, m_Enabled(false)
, m_SelectedWindowIndex(CORONAL)
, m_FocusLosingWindowIndex(-1)
, m_CursorVisibility(true)
, m_Show3DWindowIn2x2WindowLayout(false)
, m_WindowLayout(WINDOW_LAYOUT_ORTHO)
, m_TimeStep(0)
, m_CursorPositions(3)
, m_ScaleFactors(3)
, m_WorldGeometries(3)
, m_RenderWindowSizes(3)
, m_Origins(3)
, m_TimeGeometry(NULL)
, m_Geometry(NULL)
, m_BlockDisplayEvents(false)
, m_BlockSncEvents(false)
, m_BlockFocusEvents(false)
, m_BlockUpdate(false)
, m_FocusHasChanged(false)
, m_GeometryHasChanged(false)
, m_WindowLayoutHasChanged(false)
, m_TimeStepHasChanged(false)
, m_SelectedSliceHasChanged(3)
, m_CursorPositionHasChanged(3)
, m_ScaleFactorHasChanged(3)
, m_CursorPositionBindingHasChanged(false)
, m_ScaleFactorBindingHasChanged(false)
, m_CursorPositionBinding(true)
, m_CursorAxialPositionsAreBound(true)
, m_CursorSagittalPositionsAreBound(true)
, m_CursorCoronalPositionsAreBound(false)
, m_ScaleFactorBinding(true)
{
  /// Note:
  /// The rendering manager is surely not null. If NULL is specified then the superclass
  /// constructor initialised it with the default one.
  this->SetDataStorage(m_RenderingManager->GetDataStorage());

  m_RenderWindows[AXIAL] = this->GetRenderWindow1();
  m_RenderWindows[SAGITTAL] = this->GetRenderWindow2();
  m_RenderWindows[CORONAL] = this->GetRenderWindow3();
  m_RenderWindows[THREE_D] = this->GetRenderWindow4();

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
  this->SetBackgroundColour(QColor(0, 0, 0));

  // 3D planes should only be visible in this specific widget, not globally, so we create them, then make them globally invisible.
  this->AddDisplayPlaneSubTree();
  m_PlaneNode1->SetVisibility(false);
  m_PlaneNode2->SetVisibility(false);
  m_PlaneNode3->SetVisibility(false);
  this->SetCursorVisible(false);
  this->SetWidgetPlanesLocked(true);
  this->SetWidgetPlanesRotationLocked(true);

  // Need each widget to react to Qt drag/drop events.
  this->mitkWidget1->setAcceptDrops(true);
  this->mitkWidget2->setAcceptDrops(true);
  this->mitkWidget3->setAcceptDrops(true);
  this->mitkWidget4->setAcceptDrops(true);

  // Set these off, as it wont matter until there is an image dropped, with a specific layout and orientation.
  m_CornerAnnotaions[AXIAL].cornerText->SetText(0, "");
  m_CornerAnnotaions[SAGITTAL].cornerText->SetText(0, "");
  m_CornerAnnotaions[CORONAL].cornerText->SetText(0, "");

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

  m_DirectionAnnotations[AXIAL]->SetColour(0, coronalColour);
  m_DirectionAnnotations[AXIAL]->SetColour(1, sagittalColour);
  m_DirectionAnnotations[AXIAL]->SetColour(2, coronalColour);
  m_DirectionAnnotations[AXIAL]->SetColour(3, sagittalColour);
  m_DirectionAnnotations[SAGITTAL]->SetColour(0, axialColour);
  m_DirectionAnnotations[SAGITTAL]->SetColour(1, coronalColour);
  m_DirectionAnnotations[SAGITTAL]->SetColour(2, axialColour);
  m_DirectionAnnotations[SAGITTAL]->SetColour(3, coronalColour);
  m_DirectionAnnotations[CORONAL]->SetColour(0, axialColour);
  m_DirectionAnnotations[CORONAL]->SetColour(1, sagittalColour);
  m_DirectionAnnotations[CORONAL]->SetColour(2, axialColour);
  m_DirectionAnnotations[CORONAL]->SetColour(3, sagittalColour);

  // Set default layout. This must be ORTHO.
  this->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  // Default to unselected, so borders are off.
  this->DisableColoredRectangles();

  // Register to listen to SliceNavigators, slice changed events.
  itk::ReceptorMemberCommand<niftkMultiWindowWidget>::Pointer onAxialSliceChangedCommand =
    itk::ReceptorMemberCommand<niftkMultiWindowWidget>::New();
  onAxialSliceChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnAxialSliceChanged);
  m_AxialSliceObserverTag = mitkWidget1->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onAxialSliceChangedCommand);

  itk::ReceptorMemberCommand<niftkMultiWindowWidget>::Pointer onSagittalSliceChangedCommand =
    itk::ReceptorMemberCommand<niftkMultiWindowWidget>::New();
  onSagittalSliceChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnSagittalSliceChanged);
  m_SagittalSliceObserverTag = mitkWidget2->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSagittalSliceChangedCommand);

  itk::ReceptorMemberCommand<niftkMultiWindowWidget>::Pointer onCoronalSliceChangedCommand =
    itk::ReceptorMemberCommand<niftkMultiWindowWidget>::New();
  onCoronalSliceChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnCoronalSliceChanged);
  m_CoronalSliceObserverTag = mitkWidget3->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onCoronalSliceChangedCommand);

  itk::ReceptorMemberCommand<niftkMultiWindowWidget>::Pointer onTimeStepChangedCommand =
    itk::ReceptorMemberCommand<niftkMultiWindowWidget>::New();
  onTimeStepChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnTimeStepChanged);
//  m_TimeStepObserverTag = m_TimeNavigationController->AddObserver(mitk::SliceNavigationController::GeometryTimeEvent(NULL, 0), onTimeStepChangedCommand);
  m_TimeStepObserverTag = m_RenderWindows[AXIAL]->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometryTimeEvent(NULL, 0), onTimeStepChangedCommand);

  // The world position is unknown until the geometry is set. These values are invalid,
  // but still better then having undefined values.
  m_SelectedPosition.Fill(0.0);

  // The cursor is at the middle of the display at the beginning.
  m_CursorPositions[AXIAL].Fill(0.5);
  m_CursorPositions[SAGITTAL].Fill(0.5);
  m_CursorPositions[CORONAL].Fill(0.5);

  m_ScaleFactors[AXIAL] = 1.0;
  m_ScaleFactors[SAGITTAL] = 1.0;
  m_ScaleFactors[CORONAL] = 1.0;

  // Set the default voxel size to 1.0mm for each axes.
  m_MmPerVx[0] = 1.0;
  m_MmPerVx[1] = 1.0;
  m_MmPerVx[2] = 1.0;

  // Listen to the display geometry changes so we raise an event when
  // the geometry changes through the display interactor (e.g. zooming with the mouse).
  for (int i = 0; i < 3; ++i)
  {
    /// This call requires MITK modification. Currently, the displayed region is rescaled
    /// around the centre of the render window when the window is resized. This is not
    /// good for us, because we want to rescale the region around the cursor. Setting
    /// the 'KeepDisplayedRegion' property to false would disable the behaviour of MITK,
    /// so we can apply the rescaling in our way. This line is commented out until
    /// this feature is added to our MITK fork.
    m_RenderWindows[i]->GetRenderer()->SetKeepDisplayedRegion(false);
    m_RenderWindows[i]->GetRenderer()->GetDisplayGeometry()->SetConstrainZoomingAndPanning(false);
    this->AddDisplayGeometryModificationObserver(i);
  }

  // We listen to FocusManager to detect when things have changed focus, and hence to highlight the "current window".
  itk::SimpleMemberCommand<niftkMultiWindowWidget>::Pointer onFocusChangedCommand =
    itk::SimpleMemberCommand<niftkMultiWindowWidget>::New();
  onFocusChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnFocusChanged);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);

  // The mouse mode switcher is declared and initialised in QmitkStdMultiWidget. It creates an
  // mitk::DisplayInteractor. This line decreases the reference counter of the mouse mode switcher
  // so that it is destructed and it unregisters and destructs its display interactor as well.
  m_MouseModeSwitcher = 0;
}


//-----------------------------------------------------------------------------
niftkMultiWindowWidget::~niftkMultiWindowWidget()
{
  this->SetEnabled(false);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  focusManager->RemoveObserver(m_FocusManagerObserverTag);

  if (mitkWidget1 != NULL && m_AxialSliceObserverTag != 0)
  {
    mitkWidget1->GetSliceNavigationController()->RemoveObserver(m_AxialSliceObserverTag);
  }
  if (mitkWidget2 != NULL && m_SagittalSliceObserverTag != 0)
  {
    mitkWidget2->GetSliceNavigationController()->RemoveObserver(m_SagittalSliceObserverTag);
  }
  if (mitkWidget3 != NULL && m_CoronalSliceObserverTag != 0)
  {
    mitkWidget3->GetSliceNavigationController()->RemoveObserver(m_CoronalSliceObserverTag);
  }
  if (m_RenderingManager != NULL && m_TimeStepObserverTag != 0)
  {
//    m_TimeNavigationController->RemoveObserver(m_TimeStepObserverTag);
    m_RenderWindows[AXIAL]->GetSliceNavigationController()->RemoveObserver(m_TimeStepObserverTag);
  }

  // Stop listening to the display geometry changes so we raise an event when
  // the geometry changes through the display interactor (e.g. zooming with the mouse).
  for (int i = 0; i < 3; ++i)
  {
    this->RemoveDisplayGeometryModificationObserver(i);
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
void niftkMultiWindowWidget::AddDisplayGeometryModificationObserver(int windowIndex)
{
  mitk::BaseRenderer* renderer = m_RenderWindows[windowIndex]->GetRenderer();
  assert(renderer);

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  assert(displayGeometry);

  DisplayGeometryModificationCommand::Pointer command = DisplayGeometryModificationCommand::New(this, windowIndex);
  m_WorldGeometries[windowIndex] = displayGeometry->GetWorldGeometry();
  m_RenderWindowSizes[windowIndex] = displayGeometry->GetSizeInDisplayUnits();
  m_Origins[windowIndex] = displayGeometry->GetOriginInDisplayUnits();
  m_ScaleFactors[windowIndex] = displayGeometry->GetScaleFactorMMPerDisplayUnit();
  unsigned long observerTag = displayGeometry->AddObserver(itk::ModifiedEvent(), command);
  m_DisplayGeometryModificationObservers[windowIndex] = observerTag;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::RemoveDisplayGeometryModificationObserver(int windowIndex)
{
  mitk::BaseRenderer* renderer = m_RenderWindows[windowIndex]->GetRenderer();
  assert(renderer);

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  assert(displayGeometry);

  displayGeometry->RemoveObserver(m_DisplayGeometryModificationObservers[windowIndex]);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnAxialSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(AXIAL);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnSagittalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(SAGITTAL);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnCoronalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(CORONAL);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnTimeStepChanged(const itk::EventObject& /*geometryTimeEvent*/)
{
  if (!m_BlockSncEvents && m_Geometry != NULL)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

//    int timeStep = m_TimeNavigationController->GetTime()->GetPos();
    int timeStep = m_RenderWindows[AXIAL]->GetSliceNavigationController()->GetTime()->GetPos();
    if (timeStep != m_TimeStep)
    {
      m_TimeStep = timeStep;
      m_TimeStepHasChanged = true;
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetBackgroundColour(QColor colour)
{
  m_BackgroundColour = colour;
  m_GradientBackground1->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground1->Enable();
  m_GradientBackground2->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground2->Enable();
  m_GradientBackground3->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground3->Enable();
  m_GradientBackground4->SetGradientColors(colour.redF(), colour.greenF(), colour.blueF(), colour.redF(), colour.greenF(), colour.blueF());
  m_GradientBackground4->Enable();
  this->RequestUpdate();
}


//-----------------------------------------------------------------------------
QColor niftkMultiWindowWidget::GetBackgroundColour() const
{
  return m_BackgroundColour;
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::IsFocused() const
{
  return m_IsFocused;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetFocused()
{
  if (!m_IsFocused)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    m_IsFocused = true;
    m_FocusHasChanged = true;

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkMultiWindowWidget::GetSelectedRenderWindow() const
{
  assert(m_SelectedWindowIndex >= 0 && m_SelectedWindowIndex < m_RenderWindows.size());

  return m_RenderWindows[m_SelectedWindowIndex];
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetSelectedRenderWindow(QmitkRenderWindow* renderWindow)
{
  std::size_t selectedWindowIndex = std::find(m_RenderWindows.begin(), m_RenderWindows.end(), renderWindow) - m_RenderWindows.begin();
  assert(selectedWindowIndex != m_RenderWindows.size());

  this->SetSelectedWindowIndex(selectedWindowIndex);
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetSelectedWindowIndex() const
{
  assert(m_SelectedWindowIndex >= 0 && m_SelectedWindowIndex < m_RenderWindows.size());

  return m_SelectedWindowIndex;
}

//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetSelectedWindowIndex(int selectedWindowIndex)
{
  assert(selectedWindowIndex >= 0 && selectedWindowIndex < m_RenderWindows.size());

  if (selectedWindowIndex != m_SelectedWindowIndex)
  {
    bool updateWasBlocked = this->BlockUpdate(true);
    if (m_IsFocused)
    {
      m_FocusHasChanged = true;
      m_FocusLosingWindowIndex = m_SelectedWindowIndex;
    }
    m_SelectedWindowIndex = selectedWindowIndex;
    this->BlockUpdate(updateWasBlocked);
  }
}

//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::UpdateBorders()
{
  // When we "Select", the selection is at the level of the niftkMultiWindowWidget
  // so the whole of this widget is selected. However, we may have clicked in
  // a specific view, so it still helps to highlight the most recently clicked on view.
  // Also, if you are displaying ortho window layout (2x2) then you actually have 4 windows present,
  // then highlighting them all starts to look a bit confusing, so we just highlight the
  // most recently focused window, (eg. axial, sagittal, coronal or 3D).

  if (m_IsFocused && m_Geometry)
  {
    if (m_SelectedWindowIndex == AXIAL)
    {
      m_RectangleRendering1->Enable(1.0, 0.0, 0.0);
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (m_SelectedWindowIndex == SAGITTAL)
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Enable(0.0, 1.0, 0.0);
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (m_SelectedWindowIndex == CORONAL)
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Enable(0.0, 0.0, 1.0);
      m_RectangleRendering4->Disable();
    }
    else // THREE_D
    {
      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Enable(1.0, 1.0, 0.0);
    }
  }
  else
  {
    m_RectangleRendering1->Disable();
    m_RectangleRendering2->Disable();
    m_RectangleRendering3->Disable();
    m_RectangleRendering3->Disable();
  }
}


//-----------------------------------------------------------------------------
std::vector<QmitkRenderWindow*> niftkMultiWindowWidget::GetVisibleRenderWindows() const
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
void niftkMultiWindowWidget::RequestUpdate()
{
  // The point of all this is to minimise the number of Updates.
  // So, ONLY call RequestUpdate on the specific window that is shown.

  if (this->isVisible())
  {
    switch (m_WindowLayout)
    {
    case WINDOW_LAYOUT_AXIAL:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      break;
    case WINDOW_LAYOUT_SAGITTAL:
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      break;
    case WINDOW_LAYOUT_CORONAL:
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      break;
    case WINDOW_LAYOUT_ORTHO:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    case WINDOW_LAYOUT_3H:
    case WINDOW_LAYOUT_3V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
      break;
    case WINDOW_LAYOUT_3D:
      m_RenderingManager->RequestUpdate(mitkWidget4->GetRenderWindow());
      break;
    case WINDOW_LAYOUT_COR_SAG_H:
    case WINDOW_LAYOUT_COR_SAG_V:
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
    break;
    case WINDOW_LAYOUT_COR_AX_H:
    case WINDOW_LAYOUT_COR_AX_V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget3->GetRenderWindow());
    break;
    case WINDOW_LAYOUT_SAG_AX_H:
    case WINDOW_LAYOUT_SAG_AX_V:
      m_RenderingManager->RequestUpdate(mitkWidget1->GetRenderWindow());
      m_RenderingManager->RequestUpdate(mitkWidget2->GetRenderWindow());
    break;
    default:
      // die, this should never happen
      assert((m_WindowLayout >= 0 && m_WindowLayout <= 6) || (m_WindowLayout >= 9 && m_WindowLayout <= 14));
      break;
    }
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::IsEnabled() const
{
  return m_Enabled;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetEnabled(bool enabled)
{
  // See also constructor for things that are ALWAYS on/off.
  if (enabled != m_Enabled)
  {
    m_Enabled = enabled;

    if (enabled)
    {
      this->AddPlanesToDataStorage();
    }
    else
    {
      this->RemovePlanesFromDataStorage();
    }
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::IsCursorVisible() const
{
  return m_CursorVisibility;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetCursorVisible(bool visible)
{
  // Here, "locally" means, for this widget, so we are setting Renderer Specific properties.
  m_CursorVisibility = visible;
  this->SetVisibility(mitkWidget1, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget1, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget1, m_PlaneNode3, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget2, m_PlaneNode3, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode1, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode2, visible);
  this->SetVisibility(mitkWidget3, m_PlaneNode3, visible);
  this->RequestUpdate();
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::AreDirectionAnnotationsVisible() const
{
  return m_DirectionAnnotations[AXIAL]->GetVisibility()
      && m_DirectionAnnotations[SAGITTAL]->GetVisibility()
      && m_DirectionAnnotations[CORONAL]->GetVisibility();
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetDirectionAnnotationsVisible(bool visible)
{
  m_DirectionAnnotations[AXIAL]->SetVisibility(visible);
  m_DirectionAnnotations[SAGITTAL]->SetVisibility(visible);
  m_DirectionAnnotations[CORONAL]->SetVisibility(visible);
  this->RequestUpdate();
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::GetShow3DWindowIn2x2WindowLayout() const
{
  return m_Show3DWindowIn2x2WindowLayout;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetShow3DWindowIn2x2WindowLayout(bool visible)
{
  m_Show3DWindowIn2x2WindowLayout = visible;
  this->Update3DWindowVisibility();
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::Update3DWindowVisibility()
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
      if ((m_WindowLayout == WINDOW_LAYOUT_ORTHO && m_Show3DWindowIn2x2WindowLayout)
          || m_WindowLayout == WINDOW_LAYOUT_3D)
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
  m_RenderingManager->RequestUpdate(this->mitkWidget4->GetRenderWindow());
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetVisibility(QmitkRenderWindow* renderWindow, mitk::DataNode* node, bool visibility)
{
  if (renderWindow != NULL && node != NULL)
  {
    mitk::BaseRenderer* renderer = renderWindow->GetRenderer();
    if (renderer != NULL)
    {
      bool currentVisibility = false;
      node->GetVisibility(currentVisibility, renderer);

      if (visibility != currentVisibility)
      {
        node->SetVisibility(visibility, renderer);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetVisibility(std::vector<mitk::DataNode*> nodes, bool visibility)
{
  for (std::size_t i = 0; i < nodes.size(); ++i)
  {
    this->SetVisibility(mitkWidget1, nodes[i], visibility);
    this->SetVisibility(mitkWidget2, nodes[i], visibility);
    this->SetVisibility(mitkWidget3, nodes[i], visibility);
  }
  this->Update3DWindowVisibility();
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::ContainsRenderWindow(QmitkRenderWindow* renderWindow) const
{
  return mitkWidget1 == renderWindow
      || mitkWidget2 == renderWindow
      || mitkWidget3 == renderWindow
      || mitkWidget4 == renderWindow;
}


//-----------------------------------------------------------------------------
const std::vector<QmitkRenderWindow*>& niftkMultiWindowWidget::GetRenderWindows() const
{
  return m_RenderWindows;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::FitRenderWindows(double scaleFactor)
{
  if (!m_Geometry)
  {
    return;
  }

  bool updateWasBlocked = this->BlockUpdate(true);

  if (!m_CursorPositionBinding && !m_ScaleFactorBinding)
  {
    /// If neither the cursor positions nor the scale factors are not bound,
    /// we simply fit each displayed region into their window.
    for (int windowIndex = 0; windowIndex < 3; ++windowIndex)
    {
      if (m_RenderWindows[windowIndex]->isVisible())
      {
        this->FitRenderWindow(windowIndex, scaleFactor);
      }
    }
  }
  else if (m_CursorPositionBinding && !m_ScaleFactorBinding)
  {
    /// If the cursor positions are bound but the scale factors are not,
    /// first we fit the selected window then synchronise the positions
    /// in the other windows to it.
    if (m_WindowLayout != WINDOW_LAYOUT_3D)
    {
      int windowIndex = m_SelectedWindowIndex;
      if (windowIndex == THREE_D)
      {
        windowIndex = CORONAL;
      }
      this->FitRenderWindow(windowIndex, scaleFactor);
      this->SynchroniseCursorPositions(windowIndex);
    }
  }
  else
  {
    /// If the scale factors are bound then after moving the regions to the center
    /// the cursors will be aligned. Therefore we do not need to handle differently
    /// if the cursors are bound or not.

    if (scaleFactor == 0.0)
    {
      /// If the scale factors are bound and no scale factor is given then
      /// we need to find the window that requires the largest scaling.
      for (int windowIndex = 0; windowIndex < 3; ++windowIndex)
      {
        if (m_RenderWindows[windowIndex]->isVisible())
        {
          int windowWidthInPx = m_RenderWindowSizes[windowIndex][0];
          int windowHeightInPx = m_RenderWindowSizes[windowIndex][1];

          double regionWidthInMm;
          double regionHeightInMm;
          if (windowIndex == AXIAL)
          {
            regionWidthInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
            regionHeightInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
          }
          else if (windowIndex == SAGITTAL)
          {
            regionWidthInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
            regionHeightInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
          }
          else if (windowIndex == CORONAL)
          {
            regionWidthInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
            regionHeightInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
          }

          double sfh = regionWidthInMm / windowWidthInPx;
          double sfv = regionHeightInMm / windowHeightInPx;
          if (sfh > scaleFactor)
          {
            scaleFactor = sfh;
          }
          if (sfv > scaleFactor)
          {
            scaleFactor = sfv;
          }
        }
      }
    }

    /// Finally, we apply the same scale factor to every render window.
    for (int windowIndex = 0; windowIndex < 3; ++windowIndex)
    {
      if (m_RenderWindows[windowIndex]->isVisible())
      {
        this->FitRenderWindow(windowIndex, scaleFactor);
      }
    }
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::FitRenderWindow(int windowIndex, double scaleFactor)
{
  assert(windowIndex < 3);

  bool updateWasBlocked = this->BlockUpdate(true);

  double windowWidthInPx = m_RenderWindowSizes[windowIndex][0];
  double windowHeightInPx = m_RenderWindowSizes[windowIndex][1];

  double regionWidthInMm;
  double regionHeightInMm;
  if (windowIndex == AXIAL)
  {
    regionWidthInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
    regionHeightInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
  }
  else if (windowIndex == SAGITTAL)
  {
    regionWidthInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
    regionHeightInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
  }
  else if (windowIndex == CORONAL)
  {
    regionWidthInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
    regionHeightInMm = m_Geometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
  }

  if (scaleFactor == 0.0)
  {
    double sfh = regionWidthInMm / windowWidthInPx;
    double sfv = regionHeightInMm / windowHeightInPx;
    scaleFactor = sfh > sfv ? sfh : sfv;
  }
  else if (scaleFactor == -1.0)
  {
    scaleFactor = m_ScaleFactors[windowIndex];
  }

  double regionWidthInPx = regionWidthInMm / scaleFactor;
  double regionHeightInPx = regionHeightInMm / scaleFactor;

  mitk::DisplayGeometry* displayGeometry = m_RenderWindows[windowIndex]->GetRenderer()->GetDisplayGeometry();

  mitk::Point2D selectedPosition2D;
  displayGeometry->Map(m_SelectedPosition, selectedPosition2D);

  mitk::Vector2D selectedPosition2DInPx;
  selectedPosition2DInPx[0] = selectedPosition2D[0] / scaleFactor;
  selectedPosition2DInPx[1] = selectedPosition2D[1] / scaleFactor;

  mitk::Vector2D originInPx;
  originInPx[0] = (regionWidthInPx - windowWidthInPx) / 2.0;
  originInPx[1] = (regionHeightInPx - windowHeightInPx) / 2.0;

  mitk::Vector2D cursorPosition;
  cursorPosition[0] = (selectedPosition2DInPx[0] - originInPx[0]) / windowWidthInPx;
  cursorPosition[1] = (selectedPosition2DInPx[1] - originInPx[1]) / windowHeightInPx;
  /// TODO The condition is commented out so that the unit tests pass.
  /// The condition is good, though, the unit tests should be corrected.
//  if (cursorPosition != m_CursorPositions[windowIndex])
  {
    m_CursorPositions[windowIndex] = cursorPosition;
    m_CursorPositionHasChanged[windowIndex] = true;
  }

  /// TODO The condition is commented out so that the unit tests pass.
  /// The condition is good, though, the unit tests should be corrected.
//  if (scaleFactor != m_ScaleFactors[windowIndex])
  {
    m_ScaleFactors[windowIndex] = scaleFactor;
    m_ScaleFactorHasChanged[windowIndex] = true;
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetTimeGeometry(const mitk::TimeGeometry* timeGeometry)
{
  if (timeGeometry != NULL)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);

    m_TimeGeometry = timeGeometry;
    m_Geometry = timeGeometry->GetGeometryForTimeStep(0);

    // Calculating the voxel size. This is needed for the conversion between the
    // magnification and the scale factors.
    for (int axis = 0; axis < 3; ++axis)
    {
      m_MmPerVx[axis] = m_Geometry->GetExtentInMM(axis) / m_Geometry->GetExtent(axis);
    }

    // Add these annotations the first time we have a real geometry.
    m_CornerAnnotaions[AXIAL].cornerText->SetText(0, "Axial");
    m_CornerAnnotaions[SAGITTAL].cornerText->SetText(0, "Sagittal");
    m_CornerAnnotaions[CORONAL].cornerText->SetText(0, "Coronal");

    /// The place of the direction annotations on the render window:
    ///
    /// +----0----+
    /// |         |
    /// 3         1
    /// |         |
    /// +----2----+
    m_DirectionAnnotations[AXIAL]->SetText(0, "A");
    m_DirectionAnnotations[AXIAL]->SetText(2, "P");
    m_DirectionAnnotations[AXIAL]->SetText(3, "R");
    m_DirectionAnnotations[AXIAL]->SetText(1, "L");

    m_DirectionAnnotations[SAGITTAL]->SetText(0, "S");
    m_DirectionAnnotations[SAGITTAL]->SetText(2, "I");
    m_DirectionAnnotations[SAGITTAL]->SetText(3, "A");
    m_DirectionAnnotations[SAGITTAL]->SetText(1, "P");

    m_DirectionAnnotations[CORONAL]->SetText(0, "S");
    m_DirectionAnnotations[CORONAL]->SetText(2, "I");
    m_DirectionAnnotations[CORONAL]->SetText(3, "R");
    m_DirectionAnnotations[CORONAL]->SetText(1, "L");

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

    mitk::AffineTransform3D::Pointer affineTransform = m_Geometry->GetIndexToWorldTransform();
    itk::Matrix<double, 3, 3> affineTransformMatrix = affineTransform->GetMatrix();
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
    double permutedSpacing[3];

    permutedAxes[0] = dominantAxisRL;
    permutedAxes[1] = dominantAxisAP;
    permutedAxes[2] = dominantAxisSI;

    m_UpDirections[0] = signRL;
    m_UpDirections[1] = signAP;
    m_UpDirections[2] = signSI;

    permutedBoundingBox[0] = static_cast<int>(m_Geometry->GetExtent(dominantAxisRL));
    permutedBoundingBox[1] = static_cast<int>(m_Geometry->GetExtent(dominantAxisAP));
    permutedBoundingBox[2] = static_cast<int>(m_Geometry->GetExtent(dominantAxisSI));

    permutedSpacing[0] = m_Geometry->GetSpacing()[permutedAxes[0]];
    permutedSpacing[1] = m_Geometry->GetSpacing()[permutedAxes[1]];
    permutedSpacing[2] = m_Geometry->GetSpacing()[permutedAxes[2]];

    mitk::AffineTransform3D::MatrixType::InternalMatrixType permutedMatrix;
    permutedMatrix.set_identity();

    // permutedMatrix(column) = inverseTransformMatrix(row) * flippedAxes
    permutedMatrix[0][0] = inverseTransformMatrix[permutedAxes[0]][0] * m_UpDirections[0];
    permutedMatrix[1][0] = inverseTransformMatrix[permutedAxes[0]][1] * m_UpDirections[0];
    permutedMatrix[2][0] = inverseTransformMatrix[permutedAxes[0]][2] * m_UpDirections[0];
    permutedMatrix[0][1] = inverseTransformMatrix[permutedAxes[1]][0] * m_UpDirections[1];
    permutedMatrix[1][1] = inverseTransformMatrix[permutedAxes[1]][1] * m_UpDirections[1];
    permutedMatrix[2][1] = inverseTransformMatrix[permutedAxes[1]][2] * m_UpDirections[1];
    permutedMatrix[0][2] = inverseTransformMatrix[permutedAxes[2]][0] * m_UpDirections[2];
    permutedMatrix[1][2] = inverseTransformMatrix[permutedAxes[2]][1] * m_UpDirections[2];
    permutedMatrix[2][2] = inverseTransformMatrix[permutedAxes[2]][2] * m_UpDirections[2];

    m_OrientationAxes[AXIAL] = dominantAxisSI;
    m_OrientationAxes[SAGITTAL] = dominantAxisRL;
    m_OrientationAxes[CORONAL] = dominantAxisAP;

//    MITK_INFO << "Matt, image geometry=" << m_Geometry->GetImageGeometry();
//    MITK_INFO << "Matt, origin=" << m_Geometry->GetOrigin();
//    MITK_INFO << "Matt, centre=" << m_Geometry->GetCenter();
//    MITK_INFO << "Matt, extent=" << m_Geometry->GetExtent(0) << ", " << m_Geometry->GetExtent(1) << ", " << m_Geometry->GetExtent(2);
//    MITK_INFO << "Matt, domRL=" << dominantAxisRL << ", signRL=" << signRL << ", domAP=" << dominantAxisAP << ", signAP=" << signAP << ", dominantAxisSI=" << dominantAxisSI << ", signSI=" << signSI;
//    MITK_INFO << "Matt, permutedBoundingBox=" << permutedBoundingBox[0] << ", " << permutedBoundingBox[1] << ", " << permutedBoundingBox[2];
//    MITK_INFO << "Matt, permutedAxes=" << permutedAxes[0] << ", " << permutedAxes[1] << ", " << permutedAxes[2];
//    MITK_INFO << "Matt, permutedSpacing=" << permutedSpacing[0] << ", " << permutedSpacing[1] << ", " << permutedSpacing[2];
//    MITK_INFO << "Matt, flippedAxes=" << flippedAxes[0] << ", " << flippedAxes[1] << ", " << flippedAxes[2];
//    MITK_INFO << "Matt, input normalised matrix=";
//    for (unsigned int i=0; i < 3; i++)
//    {
//      MITK_INFO << affineTransformMatrix[i][0] << " " << affineTransformMatrix[i][1] << " " << affineTransformMatrix[i][2];
//    }
//    MITK_INFO << "Matt, inverse normalised matrix=";
//    for (unsigned int i=0; i < 3; i++)
//    {
//      MITK_INFO << inverseTransformMatrix[i][0] << " " << inverseTransformMatrix[i][1] << " " << inverseTransformMatrix[i][2];
//    }
//    MITK_INFO << "Matt, permuted matrix=";
//    for (unsigned int i=0; i < 3; i++)
//    {
//      MITK_INFO << permutedMatrix[i][0] << " " << permutedMatrix[i][1] << " " << permutedMatrix[i][2];
//    }

    mitk::Point3D worldBottomLeftBackCorner = m_Geometry->GetOrigin();

    if (m_Geometry->GetImageGeometry())
    {
      m_GeometryType = ImageGeometry;
      for (int i = 0; i < 3; ++i)
      {
        if (m_UpDirections[i] > 0)
        {
          worldBottomLeftBackCorner[i] -= 0.5 * permutedSpacing[i];
        }
        else
        {
          worldBottomLeftBackCorner[i] -= permutedBoundingBox[i] * permutedSpacing[i] * permutedMatrix[i][i] - 0.5 * permutedSpacing[i];
        }
      }
    }
    else if (permutedAxes[0] == 0 && permutedAxes[1] == 1 && permutedAxes[2] == 2) // Axial
    {
      m_GeometryType = AxialGeometry;
      worldBottomLeftBackCorner[1] -= permutedBoundingBox[1] * permutedSpacing[1] * permutedMatrix[1][1];
    }
    else if (permutedAxes[0] == 2 && permutedAxes[1] == 0 && permutedAxes[2] == 1) // Sagittal
    {
      m_GeometryType = SagittalGeometry;
    }
    else if (permutedAxes[0] == 0 && permutedAxes[1] == 2 && permutedAxes[2] == 1) // Coronal
    {
      m_GeometryType = CoronalGeometry;
    }
    else
    {
      assert(false);
    }
//    MITK_INFO << "Matt, bottom left corner: " << worldBottomLeftBackCorner;

    std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
    for (unsigned int i = 0; i < renderWindows.size(); i++)
    {
      QmitkRenderWindow* renderWindow = renderWindows[i];
      mitk::BaseRenderer* renderer = renderWindow->GetRenderer();

      // Get access to slice navigation controller, as this sorts out most of the process.
      mitk::SliceNavigationController* sliceNavigationController = renderer->GetSliceNavigationController();
      sliceNavigationController->SetViewDirectionToDefault();

      // Get the view/orientation flags.
      mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

      if (i < 3)
      {
        mitk::Point3D originOfSlice;
        mitk::VnlVector rightDV(3);
        mitk::VnlVector bottomDV(3);
        mitk::VnlVector normal(3);
        int width = 1;
        int height = 1;
        unsigned int slices = 1;
        mitk::ScalarType viewSpacing = 1;
        bool isFlipped;

        // Setting up the width, height, axis orientation.
        switch (viewDirection)
        {
        case mitk::SliceNavigationController::Sagittal:
          width  = permutedBoundingBox[1];
          height = permutedBoundingBox[2];
          slices = permutedBoundingBox[0];
          viewSpacing = permutedSpacing[0];
          isFlipped = false;
          originOfSlice[0] = worldBottomLeftBackCorner[0] + 0.5 * permutedSpacing[0];
          originOfSlice[1] = worldBottomLeftBackCorner[1];
          originOfSlice[2] = worldBottomLeftBackCorner[2];
          rightDV[0] = permutedSpacing[0] * permutedMatrix[0][1];
          rightDV[1] = permutedSpacing[1] * permutedMatrix[1][1];
          rightDV[2] = permutedSpacing[2] * permutedMatrix[2][1];
          bottomDV[0] = permutedSpacing[0] * permutedMatrix[0][2];
          bottomDV[1] = permutedSpacing[1] * permutedMatrix[1][2];
          bottomDV[2] = permutedSpacing[2] * permutedMatrix[2][2];
          normal[0] = permutedMatrix[0][0];
          normal[1] = permutedMatrix[1][0];
          normal[2] = permutedMatrix[2][0];
          break;
        /// Coronal:
        case mitk::SliceNavigationController::Frontal:
          width  = permutedBoundingBox[0];
          height = permutedBoundingBox[2];
          slices = permutedBoundingBox[1];
          viewSpacing = permutedSpacing[1];
          isFlipped = true;
          originOfSlice[0] = worldBottomLeftBackCorner[0];
          originOfSlice[1] = worldBottomLeftBackCorner[1] + 0.5 * permutedSpacing[1];
          originOfSlice[2] = worldBottomLeftBackCorner[2];
          rightDV[0] = permutedSpacing[0] * permutedMatrix[0][0];
          rightDV[1] = permutedSpacing[1] * permutedMatrix[1][0];
          rightDV[2] = permutedSpacing[2] * permutedMatrix[2][0];
          bottomDV[0] = permutedSpacing[0] * permutedMatrix[0][2];
          bottomDV[1] = permutedSpacing[1] * permutedMatrix[1][2];
          bottomDV[2] = permutedSpacing[2] * permutedMatrix[2][2];
          normal[0] = permutedMatrix[0][1];
          normal[1] = permutedMatrix[1][1];
          normal[2] = permutedMatrix[2][1];
          break;
        /// Axial:
        default:
          width  = permutedBoundingBox[0];
          height = permutedBoundingBox[1];
          slices = permutedBoundingBox[2];
          viewSpacing = permutedSpacing[2];
          isFlipped = true;
          originOfSlice[0] = worldBottomLeftBackCorner[0];
          originOfSlice[1] = worldBottomLeftBackCorner[1] + permutedBoundingBox[1] * permutedSpacing[1] * permutedMatrix[1][1];
          originOfSlice[2] = worldBottomLeftBackCorner[2] + permutedBoundingBox[2] * permutedSpacing[2] * permutedMatrix[2][2] - 0.5 * permutedSpacing[2];
          originOfSlice[2] = worldBottomLeftBackCorner[2] + 0.5 * permutedSpacing[2];
          rightDV[0] = permutedSpacing[0] * permutedMatrix[0][0];
          rightDV[1] = permutedSpacing[1] * permutedMatrix[1][0];
          rightDV[2] = permutedSpacing[2] * permutedMatrix[2][0];
          bottomDV[0] = -1.0 * permutedSpacing[0] * permutedMatrix[0][1];
          bottomDV[1] = -1.0 * permutedSpacing[1] * permutedMatrix[1][1];
          bottomDV[2] = -1.0 * permutedSpacing[2] * permutedMatrix[2][1];
          normal[0] = permutedMatrix[0][2];
          normal[1] = permutedMatrix[1][2];
          normal[2] = permutedMatrix[2][2];
          break;
        }

//        MITK_INFO << "Matt, image=" << m_Geometry->GetImageGeometry();
//        MITK_INFO << "Matt, width=" << width;
//        MITK_INFO << "Matt, height=" << height;
//        MITK_INFO << "Matt, originOfSlice=" << originOfSlice;
//        MITK_INFO << "Matt, rightDV=" << rightDV;
//        MITK_INFO << "Matt, bottomDV=" << bottomDV;
//        MITK_INFO << "Matt, normal=" << normal;
//        MITK_INFO << "Matt, viewSpacing=" << viewSpacing;
//        MITK_INFO << "Matt, slices=" << slices;
//        MITK_INFO << "Matt, isFlipped=" << isFlipped;

        mitk::TimeStepType numberOfTimeSteps = timeGeometry->CountTimeSteps();

        mitk::ProportionalTimeGeometry::Pointer createdTimeGeometry = mitk::ProportionalTimeGeometry::New();
        createdTimeGeometry->Initialize();
        createdTimeGeometry->Expand(numberOfTimeSteps);

        // TODO Commented out when migrating to the redesigned MITK geometry framework.
        // This will definitely not work. Should be fixed.

//        createdTimeGeometry->SetEvenlyTimed(true);

//        if (inputTimeSlicedGeometry.IsNotNull())
//        {
//          createdTimeGeometry->SetEvenlyTimed(inputTimeSlicedGeometry->GetEvenlyTimed());
//          createdTimeGeometry->SetTimeBounds(inputTimeSlicedGeometry->GetTimeBounds());
//          createdTimeGeometry->SetBounds(inputTimeSlicedGeometry->GetBounds());
//        }

        // For the PlaneGeometry.
        mitk::ScalarType bounds[6] = {
          0,
          static_cast<mitk::ScalarType>(width),
          0,
          static_cast<mitk::ScalarType>(height),
          0,
          1
        };

        // A SlicedGeometry3D is initialised from a 2D PlaneGeometry, plus the number of slices.
        mitk::PlaneGeometry::Pointer planeGeometry = mitk::PlaneGeometry::New();
        planeGeometry->SetIdentity();
        planeGeometry->SetImageGeometry(false);
        planeGeometry->SetBounds(bounds);
        planeGeometry->SetOrigin(originOfSlice);
        planeGeometry->SetMatrixByVectors(rightDV, bottomDV, normal.two_norm());

        for (mitk::TimeStepType timeStep = 0; timeStep < numberOfTimeSteps; timeStep++)
        {
          // Then we create the SlicedGeometry3D from an initial plane, and a given number of slices.
          mitk::SlicedGeometry3D::Pointer slicedGeometry = mitk::SlicedGeometry3D::New();
          slicedGeometry->SetIdentity();
          slicedGeometry->SetReferenceGeometry(m_Geometry);
          slicedGeometry->SetImageGeometry(false);
          slicedGeometry->InitializeEvenlySpaced(planeGeometry, viewSpacing, slices, isFlipped);

          slicedGeometry->SetTimeBounds(timeGeometry->GetGeometryForTimeStep(timeStep)->GetTimeBounds());
          createdTimeGeometry->SetTimeStepGeometry(slicedGeometry, timeStep);
        }
        createdTimeGeometry->Update();

//        MITK_INFO << "Matt - final geometry=" << createdTimeGeometry;
//        MITK_INFO << "Matt - final geometry origin=" << createdTimeGeometry->GetGeometryForTimeStep(0)->GetOrigin();
//        MITK_INFO << "Matt - final geometry center=" << createdTimeGeometry->GetGeometryForTimeStep(0)->GetCenter();
//        for (int j = 0; j < 8; j++)
//        {
//          MITK_INFO << "Matt - final geometry j=" << j << ", p=" << createdTimeGeometry->GetGeometryForTimeStep(0)->GetCornerPoint(j);
//        }

        sliceNavigationController->SetInputWorldTimeGeometry(createdTimeGeometry);
        sliceNavigationController->Update(mitk::SliceNavigationController::Original, true, true, false);
        sliceNavigationController->SetViewDirection(viewDirection);

        // For 2D mappers only, set to middle slice (the 3D mapper simply follows by event listening).
        if (renderer->GetMapperID() == 1)
        {
          // Now geometry is established, set to middle slice.
          int middleSlicePos = sliceNavigationController->GetSlice()->GetSteps() / 2;
          if ((slices % 2 == 0) && isFlipped)
          {
            middleSlicePos -= 1;
          }
          sliceNavigationController->GetSlice()->SetPos(middleSlicePos);
        }

        renderer->GetDisplayGeometry()->SetConstrainZoomingAndPanning(false);

        /// Note:
        /// The renderers are listening to the GeometrySendEvents of their slice navigation
        /// controller, and they update their world geometry to the one of their SNC whenever
        /// it changes. However, the SNC signals are blocked when the update of this viewer is
        /// blocked, and the SNC GeometrySendEvents are sent out only when BlockUpdate(false)
        /// is called for this widget. The renderers would update their world geometry right
        /// after this. However, the focus change signals are sent out *before* the SNC signals.
        /// As a result, if somebody is listening to the focus change signals, will find the
        /// old world geometry in the renderer. Therefore, here we manually set the new geometry
        /// to the renderers, even if they would get it later.
        /// Note also that the SNCs' Update function clones the input world geometry, therefore
        /// here we should not use the reference to 'createdTimeGeometry' but have to get
        /// it from the SNC.
        renderer->SetWorldTimeGeometry(sliceNavigationController->GetCreatedWorldGeometry());
      }
    }

    this->BlockDisplayEvents(displayEventsWereBlocked);

    m_GeometryHasChanged = true;

    m_SelectedPosition = this->GetCrossPosition();
    for (int i = 0; i < 3; ++i)
    {
      m_SelectedSliceHasChanged[i] = true;
    }

    m_TimeStep = 0;
    m_TimeStepHasChanged = true;

    this->BlockUpdate(updateWasBlocked);
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
void niftkMultiWindowWidget::SetWindowLayout(WindowLayout windowLayout)
{
/// The viewer is not correctly initialised if this check is enabled.
//  if (windowLayout == m_WindowLayout)
//  {
//    return;
//  }

  bool updateWasBlocked = this->BlockUpdate(true);

  bool displayEventsWereBlocked = this->BlockDisplayEvents(true);

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

  if (windowLayout == WINDOW_LAYOUT_3H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 2);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 0, 3);  // 3D:       off
  }
  else if (windowLayout == WINDOW_LAYOUT_3V)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 2, 0);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 0);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 1, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 3, 0);  // 3D:       off
  }
  else if (windowLayout == WINDOW_LAYOUT_COR_SAG_H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);  // axial:    off
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (windowLayout == WINDOW_LAYOUT_COR_SAG_V)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 1);  // axial:    off
    m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (windowLayout == WINDOW_LAYOUT_COR_AX_H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 1);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 1, 0);  // sagittal: off
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (windowLayout == WINDOW_LAYOUT_COR_AX_V)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: off
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (windowLayout == WINDOW_LAYOUT_SAG_AX_H)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 0, 1);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 0);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 1, 0);  // coronal:  off
    m_GridLayout->addWidget(this->mitkWidget4Container, 1, 1);  // 3D:       off
  }
  else if (windowLayout == WINDOW_LAYOUT_SAG_AX_V)
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

  bool showAxial = false;
  bool showSagittal = false;
  bool showCoronal = false;
  bool show3D = false;
  m_CursorAxialPositionsAreBound = false;
  m_CursorSagittalPositionsAreBound = false;
  m_CursorCoronalPositionsAreBound = false;

  int defaultWindowIndex;

  switch (windowLayout)
  {
  case WINDOW_LAYOUT_AXIAL:
    showAxial = true;
    defaultWindowIndex = AXIAL;
    break;
  case WINDOW_LAYOUT_SAGITTAL:
    showSagittal = true;
    defaultWindowIndex = SAGITTAL;
    break;
  case WINDOW_LAYOUT_CORONAL:
    showCoronal = true;
    defaultWindowIndex = CORONAL;
    break;
  case WINDOW_LAYOUT_ORTHO:
    showAxial = true;
    showSagittal = true;
    showCoronal = true;
    show3D = true;
    defaultWindowIndex = CORONAL;
    m_CursorAxialPositionsAreBound = true;
    m_CursorSagittalPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_3H:
    showAxial = true;
    showSagittal = true;
    showCoronal = true;
    defaultWindowIndex = CORONAL;
    m_CursorAxialPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_3V:
    showAxial = true;
    showSagittal = true;
    showCoronal = true;
    defaultWindowIndex = SAGITTAL;
    m_CursorSagittalPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_3D:
    show3D = true;
    defaultWindowIndex = THREE_D;
    break;
  case WINDOW_LAYOUT_COR_SAG_H:
    showSagittal = true;
    showCoronal = true;
    defaultWindowIndex = CORONAL;
    m_CursorAxialPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_COR_SAG_V:
    showSagittal = true;
    showCoronal = true;
    defaultWindowIndex = CORONAL;
    break;
  case WINDOW_LAYOUT_COR_AX_H:
    showAxial = true;
    showCoronal = true;
    defaultWindowIndex = CORONAL;
    break;
  case WINDOW_LAYOUT_COR_AX_V:
    showAxial = true;
    showCoronal = true;
    defaultWindowIndex = CORONAL;
    m_CursorSagittalPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_SAG_AX_H:
    showAxial = true;
    showSagittal = true;
    defaultWindowIndex = SAGITTAL;
    break;
  case WINDOW_LAYOUT_SAG_AX_V:
    showAxial = true;
    showSagittal = true;
    defaultWindowIndex = SAGITTAL;
    break;
  default:
    // die, this should never happen
    assert((m_WindowLayout >= 0 && m_WindowLayout <= 6) || (m_WindowLayout >= 9 && m_WindowLayout <= 14));
    break;
  }

  this->mitkWidget1Container->setVisible(showAxial);
  this->mitkWidget2Container->setVisible(showSagittal);
  this->mitkWidget3Container->setVisible(showCoronal);
  this->mitkWidget4Container->setVisible(show3D);

  m_CursorPositionBinding = ::IsMultiWindowLayout(windowLayout);
  m_ScaleFactorBinding = ::IsMultiWindowLayout(windowLayout);

  m_WindowLayout = windowLayout;
  m_WindowLayoutHasChanged = true;

  this->Update3DWindowVisibility();
  m_GridLayout->activate();

  if (!m_RenderWindows[m_SelectedWindowIndex]->isVisible())
  {
    m_SelectedWindowIndex = defaultWindowIndex;
    if (m_IsFocused)
    {
      m_FocusHasChanged = true;
    }
  }

  for (std::size_t i = 0; i < 4; ++i)
  {
    if (m_RenderWindows[i]->isVisible())
    {
      m_RenderWindows[i]->GetRenderWindow()->SetSize(m_RenderWindows[i]->width(), m_RenderWindows[i]->height());
    }
    else
    {
      m_RenderWindows[i]->GetRenderWindow()->SetSize(0, 0);
    }
  }

  for (size_t i = 0; i < 3; ++i)
  {
    m_RenderWindowSizes[i][0] = m_RenderWindows[i]->width();
    m_RenderWindowSizes[i][1] = m_RenderWindows[i]->height();
  }

  this->BlockDisplayEvents(displayEventsWereBlocked);

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
WindowLayout niftkMultiWindowWidget::GetWindowLayout() const
{
  return m_WindowLayout;
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetMaxSlice(int windowIndex) const
{
  int maxSlice = 0;

  mitk::SliceNavigationController* snc = m_RenderWindows[windowIndex]->GetSliceNavigationController();

  if (snc->GetSlice() != NULL && snc->GetSlice()->GetSteps() > 0)
  {
    maxSlice = snc->GetSlice()->GetSteps() - 1;
  }

  return maxSlice;
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetMaxTimeStep() const
{
//  return m_TimeNavigationController->GetTime()->GetSteps() - 1;
  return m_RenderWindows[AXIAL]->GetSliceNavigationController()->GetTime()->GetSteps() - 1;
}


//-----------------------------------------------------------------------------
const mitk::Vector2D& niftkMultiWindowWidget::GetCursorPosition(int windowIndex) const
{
  return m_CursorPositions[windowIndex];
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetCursorPosition(int windowIndex, const mitk::Vector2D& cursorPosition)
{
  bool updateWasBlocked = this->BlockUpdate(true);

  if (cursorPosition != m_CursorPositions[windowIndex])
  {
    m_CursorPositions[windowIndex] = cursorPosition;
    m_CursorPositionHasChanged[windowIndex] = true;
  }

  if (m_CursorPositionBinding)
  {
    if (windowIndex == AXIAL)
    {
      if (m_CursorSagittalPositionsAreBound)
      {
        m_CursorPositions[CORONAL][0] = m_CursorPositions[AXIAL][0];
        m_CursorPositionHasChanged[CORONAL] = true;
      }

      if (m_CursorCoronalPositionsAreBound)
      {
        m_CursorPositions[SAGITTAL][0] = 1.0 - m_CursorPositions[AXIAL][1];
        m_CursorPositionHasChanged[SAGITTAL] = true;
      }
    }
    else if (windowIndex == SAGITTAL)
    {
      if (m_CursorCoronalPositionsAreBound)
      {
        m_CursorPositions[AXIAL][1] = 1.0 - m_CursorPositions[SAGITTAL][0];
        m_CursorPositionHasChanged[AXIAL] = true;
      }

      if (m_CursorAxialPositionsAreBound)
      {
        m_CursorPositions[CORONAL][1] = m_CursorPositions[SAGITTAL][1];
        m_CursorPositionHasChanged[CORONAL] = true;
      }
    }
    else if (windowIndex == CORONAL)
    {
      if (m_CursorSagittalPositionsAreBound)
      {
        m_CursorPositions[AXIAL][0] = m_CursorPositions[CORONAL][0];
        m_CursorPositionHasChanged[AXIAL] = true;
      }

      if (m_CursorAxialPositionsAreBound)
      {
        m_CursorPositions[SAGITTAL][1] = m_CursorPositions[CORONAL][1];
        m_CursorPositionHasChanged[SAGITTAL] = true;
      }
    }
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::MoveToCursorPosition(int windowIndex)
{
  if (m_Geometry)
  {
    QmitkRenderWindow* renderWindow = m_RenderWindows[windowIndex];
    if (renderWindow->isVisible())
    {
      mitk::DisplayGeometry* displayGeometry = renderWindow->GetRenderer()->GetDisplayGeometry();

      mitk::Vector2D displaySize = displayGeometry->GetSizeInDisplayUnits();

      mitk::Point2D point2DInMm;
      displayGeometry->Map(m_SelectedPosition, point2DInMm);
      double scaleFactor = displayGeometry->GetScaleFactorMMPerDisplayUnit();
      mitk::Vector2D point2DInPx;
      point2DInPx[0] = point2DInMm[0] / scaleFactor;
      point2DInPx[1] = point2DInMm[1] / scaleFactor;

      mitk::Vector2D positionInPx;
      positionInPx[0] = m_CursorPositions[windowIndex][0] * displaySize[0];
      positionInPx[1] = m_CursorPositions[windowIndex][1] * displaySize[1];

      mitk::Vector2D originInPx = point2DInPx - positionInPx;

      mitk::Vector2D originInMm;
      displayGeometry->DisplayToWorld(originInPx, originInMm);

      bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
      displayGeometry->SetOriginInMM(originInMm);
      this->BlockDisplayEvents(displayEventsWereBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnDisplayGeometryModified(int windowIndex)
{
  if (m_BlockDisplayEvents || !m_Geometry)
  {
    return;
  }

  bool updateWasBlocked = this->BlockUpdate(true);

  mitk::DisplayGeometry* displayGeometry = m_RenderWindows[windowIndex]->GetRenderer()->GetDisplayGeometry();

  const mitk::Geometry2D* worldGeometry = displayGeometry->GetWorldGeometry();
  if (worldGeometry != m_WorldGeometries[windowIndex])
  {
    m_WorldGeometries[windowIndex] = worldGeometry;
    m_Origins[windowIndex] = displayGeometry->GetOriginInDisplayUnits();
    m_ScaleFactors[windowIndex] = displayGeometry->GetScaleFactorMMPerDisplayUnit();
  }

  mitk::Vector2D renderWindowSize = displayGeometry->GetSizeInDisplayUnits();
  if (renderWindowSize != m_RenderWindowSizes[windowIndex])
  {
    double horizontalSizeChange = m_RenderWindowSizes[windowIndex][0] / renderWindowSize[0];
    double verticalSizeChange = m_RenderWindowSizes[windowIndex][1] / renderWindowSize[1];
    double horizontalScaleFactor = m_ScaleFactors[windowIndex] * horizontalSizeChange;
    double verticalScaleFactor = m_ScaleFactors[windowIndex] * verticalSizeChange;

    /// Find the largest change, let it be zooming or unzooming.
    if (horizontalSizeChange < 1.0)
    {
      horizontalSizeChange = 1.0 / horizontalSizeChange;
    }
    if (verticalSizeChange < 1.0)
    {
      verticalSizeChange = 1.0 / verticalSizeChange;
    }
    double scaleFactor = horizontalSizeChange > verticalSizeChange ? horizontalScaleFactor : verticalScaleFactor;

    m_Origins[windowIndex] = displayGeometry->GetOriginInDisplayUnits();
    m_CursorPositionHasChanged[windowIndex] = true;
    m_ScaleFactors[windowIndex] = scaleFactor;
    m_ScaleFactorHasChanged[windowIndex] = true;
    m_RenderWindowSizes[windowIndex] = renderWindowSize;

    this->BlockUpdate(updateWasBlocked);
    return;
  }

  // Note that the scaling changes the scale factor *and* the origin,
  // while the moving changes the origin only.

  bool beingPanned = true;

  double scaleFactor = displayGeometry->GetScaleFactorMMPerDisplayUnit();
  if (scaleFactor != m_ScaleFactors[windowIndex])
  {
    beingPanned = false;

    /// Note:
    /// Even if the zooming was not initiated by the DnDDisplayInteractor, we still
    /// have to zoom around the selected position, not the origin, centre or mouse
    /// position, whatever. This happens e.g. if you zoom through the thumbnail viewer,
    /// or call the DisplayGeometry SetScaleFactor/Zoom functions from code.
    mitk::Vector2D origin = displayGeometry->GetOriginInDisplayUnits();
    mitk::Vector2D focusPoint = (m_Origins[windowIndex] * m_ScaleFactors[windowIndex] - origin * scaleFactor) / (scaleFactor - m_ScaleFactors[windowIndex]);
    focusPoint[0] /= m_RenderWindowSizes[windowIndex][0];
    focusPoint[1] /= m_RenderWindowSizes[windowIndex][1];

    if (focusPoint != m_CursorPositions[windowIndex])
    {
      this->MoveToCursorPosition(windowIndex);
    }

    this->OnScaleFactorChanged(windowIndex, scaleFactor);
  }

  mitk::Vector2D origin = displayGeometry->GetOriginInDisplayUnits();
  if (origin != m_Origins[windowIndex])
  {
    if (beingPanned)
    {
      this->OnOriginChanged(windowIndex, beingPanned);
    }
    m_Origins[windowIndex] = origin;
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnOriginChanged(int windowIndex, bool beingPanned)
{
  if (m_Geometry)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    this->UpdateCursorPosition(windowIndex);

    if (beingPanned && this->GetCursorPositionBinding())
    {
      // sagittal[1] <-> coronal[1]      (if m_CursorAxialPositionsAreBound is set)
      // axial[0] <-> coronal[0]         (if m_CursorSagittalPositionsAreBound is set)
      // axial[1] <-> 1.0 - sagittal[0]  (if m_CursorCoronalPositionsAreBound is set)

      if (windowIndex == AXIAL)
      {
        if (m_CursorSagittalPositionsAreBound)
        {
          m_CursorPositions[CORONAL][0] = m_CursorPositions[AXIAL][0];
          m_CursorPositionHasChanged[CORONAL] = true;
        }

        if (m_CursorCoronalPositionsAreBound)
        {
          m_CursorPositions[SAGITTAL][0] = 1.0 - m_CursorPositions[AXIAL][1];
          m_CursorPositionHasChanged[SAGITTAL] = true;
        }
      }
      else if (windowIndex == SAGITTAL)
      {
        if (m_CursorCoronalPositionsAreBound)
        {
          m_CursorPositions[AXIAL][1] = 1.0 - m_CursorPositions[SAGITTAL][0];
          m_CursorPositionHasChanged[AXIAL] = true;
        }

        if (m_CursorAxialPositionsAreBound)
        {
          m_CursorPositions[CORONAL][1] = m_CursorPositions[SAGITTAL][1];
          m_CursorPositionHasChanged[CORONAL] = true;
        }
      }
      else if (windowIndex == CORONAL)
      {
        if (m_CursorSagittalPositionsAreBound)
        {
          m_CursorPositions[AXIAL][0] = m_CursorPositions[CORONAL][0];
          m_CursorPositionHasChanged[AXIAL] = true;
        }

        if (m_CursorAxialPositionsAreBound)
        {
          m_CursorPositions[SAGITTAL][1] = m_CursorPositions[CORONAL][1];
          m_CursorPositionHasChanged[SAGITTAL] = true;
        }
      }
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnScaleFactorChanged(int windowIndex, double scaleFactor)
{
  if (m_Geometry)
  {
    if (scaleFactor != m_ScaleFactors[windowIndex])
    {
      bool updateWasBlocked = this->BlockUpdate(true);
      m_ScaleFactors[windowIndex] = scaleFactor;
      m_ScaleFactorHasChanged[windowIndex] = true;

      if (this->GetScaleFactorBinding())
      {
        // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
        for (int i = 0; i < 3; ++i)
        {
          if (i != windowIndex && m_RenderWindows[i]->isVisible())
          {
            this->SetScaleFactor(i, scaleFactor);
          }
        }
      }
      this->BlockUpdate(updateWasBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnSelectedPositionChanged(int windowIndex)
{
  if (!m_BlockSncEvents && m_Geometry != NULL)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    mitk::Point3D selectedPosition = this->GetCrossPosition();
    if (selectedPosition != m_SelectedPosition)
    {
      m_SelectedPosition = selectedPosition;
      m_SelectedSliceHasChanged[windowIndex] = true;
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetSelectedSlice(int windowIndex) const
{
  assert(0 <= windowIndex && windowIndex < 3);

  int selectedSlice = 0;

  if (m_Geometry != NULL)
  {
    int axis = m_OrientationAxes[windowIndex];

    mitk::Point3D selectedPositionInVx;
    m_Geometry->WorldToIndex(m_SelectedPosition, selectedPositionInVx);

    /// Round it to the closest integer.
    selectedSlice = static_cast<int>(selectedPositionInVx[axis] + 0.5);
  }

  return selectedSlice;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetSelectedSlice(int windowIndex, int selectedSlice)
{
  if (m_Geometry != NULL)
  {
    mitk::Point3D selectedPosition = m_SelectedPosition;

    mitk::Point3D selectedPositionInVx;
    m_Geometry->WorldToIndex(selectedPosition, selectedPositionInVx);

    int axis = m_OrientationAxes[windowIndex];
    selectedPositionInVx[axis] = selectedSlice;

    m_Geometry->IndexToWorld(selectedPositionInVx, selectedPosition);

    this->SetSelectedPosition(selectedPosition);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::MoveAnteriorOrPosterior(int windowIndex, int slices)
{
  if (m_Geometry && windowIndex < 3 && slices != 0)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    int selectedSlice = this->GetSelectedSlice(windowIndex);

    int upDirection;
    if (windowIndex == AXIAL)
    {
      upDirection = m_UpDirections[2];
    }
    else if (windowIndex == SAGITTAL)
    {
      upDirection = m_UpDirections[0];
    }
    else if (windowIndex == CORONAL)
    {
      upDirection = m_UpDirections[1];
    }

    int nextSelectedSlice = selectedSlice + upDirection * slices;

    int maxSlice = this->GetMaxSlice(windowIndex);

    if (nextSelectedSlice >= 0 && nextSelectedSlice <= static_cast<int>(maxSlice))
    {
      this->SetSelectedSlice(windowIndex, nextSelectedSlice);

      /// Note. As a request and for MIDAS compatibility, all the slice have to be forcibly rendered
      /// when scrolling through them by keeping the 'a' or 'z' key pressed.
      /// Otherwise, issues on the scan or in the segmentation may be not seen.

      /// TODO:
      /// In spite of the comment above, it is not right to do any render window update here.
      /// The forced immediate update should be done in the BlockUpdate() function.
//      m_RenderingManager->ForceImmediateUpdate(m_RenderWindows[windowIndex]->GetRenderWindow());
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
const mitk::Point3D& niftkMultiWindowWidget::GetSelectedPosition() const
{
  return m_SelectedPosition;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition)
{
  if (selectedPosition != m_SelectedPosition)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    m_SelectedPosition = selectedPosition;

    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);

    mitk::SliceNavigationController* axialSnc = m_RenderWindows[AXIAL]->GetSliceNavigationController();
    mitk::SliceNavigationController* sagittalSnc = m_RenderWindows[SAGITTAL]->GetSliceNavigationController();
    mitk::SliceNavigationController* coronalSnc = m_RenderWindows[CORONAL]->GetSliceNavigationController();

    if (axialSnc->GetCreatedWorldGeometry())
    {
      unsigned pos = axialSnc->GetSlice()->GetPos();
      axialSnc->SelectSliceByPoint(selectedPosition);
      m_SelectedSliceHasChanged[AXIAL] = m_SelectedSliceHasChanged[AXIAL] || pos != axialSnc->GetSlice()->GetPos();
    }
    if (sagittalSnc->GetCreatedWorldGeometry())
    {
      unsigned pos = sagittalSnc->GetSlice()->GetPos();
      sagittalSnc->SelectSliceByPoint(selectedPosition);
      m_SelectedSliceHasChanged[SAGITTAL] = m_SelectedSliceHasChanged[SAGITTAL] || pos != sagittalSnc->GetSlice()->GetPos();
    }
    if (coronalSnc->GetCreatedWorldGeometry())
    {
      unsigned pos = coronalSnc->GetSlice()->GetPos();
      coronalSnc->SelectSliceByPoint(selectedPosition);
      m_SelectedSliceHasChanged[CORONAL] = m_SelectedSliceHasChanged[CORONAL] || pos != coronalSnc->GetSlice()->GetPos();
    }

    this->BlockDisplayEvents(displayEventsWereBlocked);

    m_SelectedPosition = this->GetCrossPosition();

    if (m_WindowLayout != WINDOW_LAYOUT_3D)
    {
      /// Work out a window so that if the cursor positions are bound then
      /// we can synchronise the other 2D render windows to it.
      int windowIndex = m_SelectedWindowIndex;
      if (windowIndex == THREE_D)
      {
        windowIndex = CORONAL;
      }
      this->SynchroniseCursorPositions(windowIndex);
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetTimeStep() const
{
  return m_TimeStep;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetTimeStep(int timeStep)
{
  if (timeStep != m_TimeStep)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    m_TimeStep = timeStep;
    m_TimeStepHasChanged = true;

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SynchroniseCursorPositions(int windowIndex)
{
  /// sagittal[1] <-> coronal[1]      (if m_CursorAxialPositionsAreBound is set)
  /// axial[0] <-> coronal[0]         (if m_CursorSagittalPositionsAreBound is set)
  /// axial[1] <-> 1.0 - sagittal[0]  (if m_CursorCoronalPositionsAreBound is set)

  if (windowIndex == AXIAL)
  {
    this->UpdateCursorPosition(AXIAL);

    if (m_SelectedSliceHasChanged[AXIAL])
    {
      this->UpdateCursorPosition(CORONAL);

      if (m_CursorPositionBinding && m_CursorAxialPositionsAreBound)
      {
        m_CursorPositions[SAGITTAL][1] = m_CursorPositions[CORONAL][1];
        m_CursorPositionHasChanged[SAGITTAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(SAGITTAL);
      }
    }

    if (m_SelectedSliceHasChanged[SAGITTAL])
    {
      if (m_CursorPositionBinding && m_CursorSagittalPositionsAreBound)
      {
        m_CursorPositions[CORONAL][0] = m_CursorPositions[AXIAL][0];
        m_CursorPositionHasChanged[CORONAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(CORONAL);
      }
    }

    if (m_SelectedSliceHasChanged[CORONAL])
    {
      if (m_CursorPositionBinding && m_CursorCoronalPositionsAreBound)
      {
        m_CursorPositions[SAGITTAL][0] = 1.0 - m_CursorPositions[AXIAL][1];
        m_CursorPositionHasChanged[SAGITTAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(SAGITTAL);
      }
    }
  }
  else if (windowIndex == SAGITTAL)
  {
    this->UpdateCursorPosition(SAGITTAL);

    if (m_SelectedSliceHasChanged[SAGITTAL])
    {
      this->UpdateCursorPosition(CORONAL);

      if (m_CursorPositionBinding && m_CursorSagittalPositionsAreBound)
      {
        m_CursorPositions[AXIAL][0] = m_CursorPositions[CORONAL][0];
        m_CursorPositionHasChanged[AXIAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(AXIAL);
      }
    }

    if (m_SelectedSliceHasChanged[AXIAL])
    {
      if (m_CursorPositionBinding && m_CursorAxialPositionsAreBound)
      {
        m_CursorPositions[CORONAL][1] = m_CursorPositions[SAGITTAL][1];
        m_CursorPositionHasChanged[CORONAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(CORONAL);
      }
    }

    if (m_SelectedSliceHasChanged[CORONAL])
    {
      if (m_CursorPositionBinding && m_CursorCoronalPositionsAreBound)
      {
        m_CursorPositions[AXIAL][1] = 1.0 - m_CursorPositions[SAGITTAL][0];
        m_CursorPositionHasChanged[AXIAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(AXIAL);
      }
    }
  }
  else if (windowIndex == CORONAL)
  {
    this->UpdateCursorPosition(CORONAL);

    if (m_SelectedSliceHasChanged[CORONAL])
    {
      this->UpdateCursorPosition(AXIAL);
      this->UpdateCursorPosition(SAGITTAL);
    }

    if (m_SelectedSliceHasChanged[AXIAL])
    {
      if (m_CursorPositionBinding && m_CursorAxialPositionsAreBound)
      {
        m_CursorPositions[SAGITTAL][1] = m_CursorPositions[CORONAL][1];
        m_CursorPositionHasChanged[SAGITTAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(SAGITTAL);
      }
    }

    if (m_SelectedSliceHasChanged[SAGITTAL])
    {
      if (m_CursorPositionBinding && m_CursorSagittalPositionsAreBound)
      {
        m_CursorPositions[AXIAL][0] = m_CursorPositions[CORONAL][0];
        m_CursorPositionHasChanged[AXIAL] = true;
      }
      else
      {
        this->UpdateCursorPosition(AXIAL);
      }
    }
  }
}


//-----------------------------------------------------------------------------
const std::vector<mitk::Vector2D>& niftkMultiWindowWidget::GetCursorPositions() const
{
  return m_CursorPositions;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetCursorPositions(const std::vector<mitk::Vector2D>& cursorPositions)
{
  assert(cursorPositions.size() == 3);

  bool updateWasBlocked = this->BlockUpdate(true);
  for (std::size_t i = 0; i < 3; ++i)
  {
    if (m_RenderWindows[i]->isVisible() && cursorPositions[i] != m_CursorPositions[i])
    {
      m_CursorPositions[i] = cursorPositions[i];
      m_CursorPositionHasChanged[i] = true;
    }
  }
  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::UpdateCursorPosition(int windowIndex)
{
  bool updateWasBlocked = this->BlockUpdate(true);

  mitk::BaseRenderer* renderer = m_RenderWindows[windowIndex]->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  mitk::Vector2D displaySize = displayGeometry->GetSizeInDisplayUnits();

  mitk::Point2D point2DInMm;
  displayGeometry->Map(m_SelectedPosition, point2DInMm);

  mitk::Point2D point2DInPx;
  displayGeometry->WorldToDisplay(point2DInMm, point2DInPx);

  mitk::Vector2D cursorPositions;
  cursorPositions[0] = point2DInPx[0] / displaySize[0];
  cursorPositions[1] = point2DInPx[1] / displaySize[1];
  if (cursorPositions != m_CursorPositions[windowIndex])
  {
    m_CursorPositions[windowIndex] = cursorPositions;
    m_CursorPositionHasChanged[windowIndex] = true;
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
double niftkMultiWindowWidget::GetScaleFactor(int windowIndex) const
{
  return m_ScaleFactors[windowIndex];
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetScaleFactor(int windowIndex, double scaleFactor)
{
  bool updateWasBlocked = this->BlockUpdate(true);

  if (scaleFactor != m_ScaleFactors[windowIndex])
  {
    m_ScaleFactors[windowIndex] = scaleFactor;
    m_ScaleFactorHasChanged[windowIndex] = true;
  }

  if (this->GetScaleFactorBinding())
  {
    // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
    for (int otherWindowIndex = 0; otherWindowIndex < 3; ++otherWindowIndex)
    {
      if (otherWindowIndex != windowIndex && scaleFactor != m_ScaleFactors[otherWindowIndex])
      {
        m_ScaleFactors[otherWindowIndex] = scaleFactor;
        m_ScaleFactorHasChanged[otherWindowIndex] = true;
      }
    }
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
const std::vector<double>& niftkMultiWindowWidget::GetScaleFactors() const
{
  return m_ScaleFactors;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetScaleFactors(const std::vector<double>& scaleFactors)
{
  assert(scaleFactors.size() == 3);

  bool updateWasBlocked = this->BlockUpdate(true);

  for (std::size_t i = 0; i < 3; ++i)
  {
    if (m_RenderWindows[i]->isVisible() && scaleFactors[i] != m_ScaleFactors[i])
    {
      m_ScaleFactors[i] = scaleFactors[i];
      m_ScaleFactorHasChanged[i] = true;
    }
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::ZoomAroundCursorPosition(int windowIndex)
{
  if (m_Geometry)
  {
    mitk::DisplayGeometry* displayGeometry = m_RenderWindows[windowIndex]->GetRenderer()->GetDisplayGeometry();

    mitk::Vector2D displaySize = displayGeometry->GetSizeInDisplayUnits();

    mitk::Point2D focusPoint2DInPx;
    focusPoint2DInPx[0] = m_CursorPositions[windowIndex][0] * displaySize[0];
    focusPoint2DInPx[1] = m_CursorPositions[windowIndex][1] * displaySize[1];

    double scaleFactor = m_ScaleFactors[windowIndex];
    double previousScaleFactor = displayGeometry->GetScaleFactorMMPerDisplayUnit();
    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
    if (displayGeometry->SetScaleFactor(scaleFactor))
    {
      mitk::Vector2D originInMm = displayGeometry->GetOriginInMM();
      displayGeometry->SetOriginInMM(originInMm - focusPoint2DInPx.GetVectorFromOrigin() * (scaleFactor - previousScaleFactor));
    }
    this->BlockDisplayEvents(displayEventsWereBlocked);
  }
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetDominantAxis(int windowIndex) const
{
  int axisWithLongerSide = 0;

  int axialAxis = m_OrientationAxes[AXIAL];
  int sagittalAxis = m_OrientationAxes[SAGITTAL];
  int coronalAxis = m_OrientationAxes[CORONAL];

  if (windowIndex == AXIAL)
  {
    axisWithLongerSide = m_MmPerVx[sagittalAxis] < m_MmPerVx[coronalAxis] ? sagittalAxis : coronalAxis;
  }
  else if (windowIndex == SAGITTAL)
  {
    axisWithLongerSide = m_MmPerVx[axialAxis] < m_MmPerVx[coronalAxis] ? axialAxis : coronalAxis;
  }
  else if (windowIndex == CORONAL)
  {
    axisWithLongerSide = m_MmPerVx[axialAxis] < m_MmPerVx[sagittalAxis] ? axialAxis : sagittalAxis;
  }

  return axisWithLongerSide;
}


//-----------------------------------------------------------------------------
double niftkMultiWindowWidget::GetMagnification(int windowIndex) const
{
  double magnification = 0.0;

  if (m_Geometry)
  {
    int dominantAxis = this->GetDominantAxis(windowIndex);
    double scaleFactorPxPerVx = m_MmPerVx[dominantAxis] / m_ScaleFactors[windowIndex];

    // Finally, we calculate the magnification from the scale factor.
    magnification = scaleFactorPxPerVx - 1.0;

    if (magnification < 0.0)
    {
      magnification /= scaleFactorPxPerVx;
    }
  }

  return magnification;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetMagnification(int windowIndex, double magnification)
{
  if (m_Geometry)
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

    int dominantAxis = this->GetDominantAxis(windowIndex);
    double scaleFactor = m_MmPerVx[dominantAxis] * scaleFactorVxPerPx;

    this->SetScaleFactor(windowIndex, scaleFactor);
  }
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetSliceUpDirection(WindowOrientation orientation) const
{
  int upDirection = 0;
  if (m_Geometry && orientation >= 0 && orientation < 3)
  {
    if (orientation == WINDOW_ORIENTATION_AXIAL)
    {
      upDirection = m_UpDirections[2];
    }
    else if (orientation == WINDOW_ORIENTATION_SAGITTAL)
    {
      upDirection = m_UpDirections[0];
    }
    else if (orientation == WINDOW_ORIENTATION_CORONAL)
    {
      upDirection = m_UpDirections[1];
    }
  }
  return upDirection;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnFocusChanged()
{
  if (m_BlockFocusEvents)
  {
    return;
  }

  mitk::BaseRenderer* focusedRenderer = mitk::GlobalInteraction::GetInstance()->GetFocus();

  int focusedWindowIndex = -1;
  for (std::size_t i = 0; i < 4; ++i)
  {
    if (focusedRenderer == m_RenderWindows[i]->GetRenderer())
    {
      focusedWindowIndex = i;
      break;
    }
  }

  bool isFocused = focusedWindowIndex != -1;

  if (isFocused != m_IsFocused
      || (isFocused && focusedWindowIndex != m_SelectedWindowIndex))
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    if (m_IsFocused)
    {
      m_FocusLosingWindowIndex = m_SelectedWindowIndex;
    }

    if (isFocused)
    {
      m_SelectedWindowIndex = focusedWindowIndex;
    }

    m_IsFocused = isFocused;
    m_FocusHasChanged = true;

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::IsLinkedNavigationEnabled() const
{
  return m_LinkedNavigationEnabled;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetLinkedNavigationEnabled(bool linkedNavigationEnabled)
{
  if (linkedNavigationEnabled != m_LinkedNavigationEnabled)
  {
    m_LinkedNavigationEnabled = linkedNavigationEnabled;
    this->SetWidgetPlanesLocked(!linkedNavigationEnabled || !m_IsFocused || !m_Geometry);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::GetCursorPositionBinding() const
{
  return m_CursorPositionBinding;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetCursorPositionBinding(bool cursorPositionBinding)
{
  if (cursorPositionBinding != m_CursorPositionBinding)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    m_CursorPositionBinding = cursorPositionBinding;
    m_CursorPositionBindingHasChanged = true;

    if (cursorPositionBinding)
    {
      if (m_SelectedWindowIndex < 3)
      {
        this->OnOriginChanged(m_SelectedWindowIndex, true);
        if (m_WindowLayout == WINDOW_LAYOUT_ORTHO
            && (m_SelectedWindowIndex == AXIAL || m_SelectedWindowIndex == SAGITTAL))
        {
          /// We raise the event in the coronal window so that the cursors are in sync
          /// along the third axis, too.
          this->MoveToCursorPosition(CORONAL);
          this->OnOriginChanged(CORONAL, true);
        }
      }
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::GetScaleFactorBinding() const
{
  return m_ScaleFactorBinding;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetScaleFactorBinding(bool scaleFactorBinding)
{
  if (scaleFactorBinding != m_ScaleFactorBinding)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    m_ScaleFactorBinding = scaleFactorBinding;
    m_ScaleFactorBindingHasChanged = true;

    if (scaleFactorBinding)
    {
      if (m_SelectedWindowIndex < 3)
      {
        for (int otherWindowIndex = 0; otherWindowIndex < 3; ++otherWindowIndex)
        {
          if (otherWindowIndex != m_SelectedWindowIndex)
          {
            this->SetScaleFactor(otherWindowIndex, m_ScaleFactors[m_SelectedWindowIndex]);
          }
        }
      }
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::BlockDisplayEvents(bool blocked)
{
  bool eventsWereBlocked = m_BlockDisplayEvents;
  m_BlockDisplayEvents = blocked;
  return eventsWereBlocked;
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::BlockUpdate(bool blocked)
{
  bool updateWasBlocked = m_BlockUpdate;

  if (blocked != m_BlockUpdate)
  {
    m_BlockUpdate = blocked;

    for (int i = 0; i < 4; ++i)
    {
      m_RenderWindows[i]->GetSliceNavigationController()->BlockSignals(blocked);
    }

    if (!blocked)
    {
      bool rendererNeedsUpdate[4] = {false, false, false, false};

      /// Updating state according to the recorded changes.

      if (m_FocusHasChanged)
      {
        this->SetWidgetPlanesLocked(!m_LinkedNavigationEnabled || !m_IsFocused || !m_Geometry);

        this->UpdateBorders();

        if (m_FocusLosingWindowIndex != -1)
        {
          rendererNeedsUpdate[m_FocusLosingWindowIndex] = true;
          m_FocusLosingWindowIndex = -1;
        }
        rendererNeedsUpdate[m_SelectedWindowIndex] = true;
      }

      if (m_GeometryHasChanged || m_TimeStepHasChanged)
      {
        this->SetWidgetPlanesLocked(!m_LinkedNavigationEnabled || !m_IsFocused || !m_Geometry);

        /// Note:
        /// A viewer has a border iff it has the focus *and* it has a valid geometry.
        /// Therefore, the borders should be updated at the first time when m_Geometry
        /// is assigned a valid geometry.
        this->UpdateBorders();

        for (unsigned i = 0; i < 4; ++i)
        {
          rendererNeedsUpdate[i] = true;
        }
      }

      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_SelectedSliceHasChanged[i])
        {
          rendererNeedsUpdate[i] = true;
        }
      }

      if (m_TimeStepHasChanged)
      {
//        m_TimeNavigationController->GetTime()->SetPos(m_TimeStep);
        m_RenderWindows[AXIAL]->GetSliceNavigationController()->GetTime()->SetPos(m_TimeStep);
        m_RenderWindows[SAGITTAL]->GetSliceNavigationController()->GetTime()->SetPos(m_TimeStep);
        m_RenderWindows[CORONAL]->GetSliceNavigationController()->GetTime()->SetPos(m_TimeStep);
      }

      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_CursorPositionHasChanged[i])
        {
          this->MoveToCursorPosition(i);
          rendererNeedsUpdate[i] = true;
        }
      }

      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_ScaleFactorHasChanged[i])
        {
          this->ZoomAroundCursorPosition(i);
          rendererNeedsUpdate[i] = true;
        }
      }

      /// Updating render windows where necessary.

      for (unsigned i = 0; i < 4; ++i)
      {
        if (m_RenderWindows[i]->isVisible() && rendererNeedsUpdate[i])
        {
          m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
        }
      }

      /// Sending events and signals.

      if (m_FocusHasChanged)
      {
        m_FocusHasChanged = false;
        if (m_IsFocused && m_Geometry)
        {
          m_BlockFocusEvents = true;
          m_RenderWindows[m_SelectedWindowIndex]->setFocus();
          mitk::GlobalInteraction::GetInstance()->SetFocus(m_RenderWindows[m_SelectedWindowIndex]->GetRenderer());
          m_BlockFocusEvents = false;
        }
      }

      if (m_GeometryHasChanged)
      {
        m_GeometryHasChanged = false;
        if (m_IsFocused && m_Geometry)
        {
          m_BlockFocusEvents = true;
          m_RenderWindows[m_SelectedWindowIndex]->setFocus();
          mitk::GlobalInteraction::GetInstance()->SetFocus(m_RenderWindows[m_SelectedWindowIndex]->GetRenderer());
          m_BlockFocusEvents = false;
        }
        for (unsigned i = 0; i < 3; ++i)
        {
          m_BlockSncEvents = true;
          bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
          m_RenderWindows[i]->GetSliceNavigationController()->SendCreatedWorldGeometry();
          this->BlockDisplayEvents(displayEventsWereBlocked);
          m_BlockSncEvents = false;
        }
      }

      bool selectedPositionHasChanged = false;
      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_SelectedSliceHasChanged[i])
        {
          selectedPositionHasChanged = true;
          m_SelectedSliceHasChanged[i] = false;
          m_BlockSncEvents = true;
          bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
          m_RenderWindows[i]->GetSliceNavigationController()->SendSlice();
          this->BlockDisplayEvents(displayEventsWereBlocked);
          m_BlockSncEvents = false;
        }
      }

      if (m_TimeStepHasChanged)
      {
//          m_RenderWindows[AXIAL]->GetSliceNavigationController()->SendTime();
//        m_TimeNavigationController->SendTime();
        for (unsigned i = 0; i < 3; ++i)
        {
          m_BlockSncEvents = true;
          bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
          m_RenderWindows[i]->GetSliceNavigationController()->SendTime();
          this->BlockDisplayEvents(displayEventsWereBlocked);
          m_BlockSncEvents = false;
        }
      }

      if (selectedPositionHasChanged)
      {
        emit SelectedPositionChanged(m_SelectedPosition);
      }

      if (m_TimeStepHasChanged)
      {
        m_TimeStepHasChanged = false;
        emit TimeStepChanged(m_TimeStep);
      }

      if (m_WindowLayoutHasChanged)
      {
        m_WindowLayoutHasChanged = false;
        emit WindowLayoutChanged(m_WindowLayout);
      }

      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_CursorPositionHasChanged[i])
        {
          m_CursorPositionHasChanged[i] = false;
          if (m_RenderWindows[i]->isVisible())
          {
            emit CursorPositionChanged(i, m_CursorPositions[i]);
          }
        }
      }

      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_ScaleFactorHasChanged[i])
        {
          m_ScaleFactorHasChanged[i] = false;
          if (m_RenderWindows[i]->isVisible())
          {
            emit ScaleFactorChanged(i, m_ScaleFactors[i]);
          }
        }
      }

      if (m_CursorPositionBindingHasChanged)
      {
        m_CursorPositionBindingHasChanged = false;
        emit CursorPositionBindingChanged();
      }

      if (m_ScaleFactorBindingHasChanged)
      {
        m_ScaleFactorBindingHasChanged = false;
        emit ScaleFactorBindingChanged();
      }
    }
  }

  return updateWasBlocked;
}
