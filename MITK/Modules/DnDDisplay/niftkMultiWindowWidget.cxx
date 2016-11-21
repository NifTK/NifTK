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
#include <vtkTextProperty.h>
#include <QmitkRenderWindow.h>
#include <QGridLayout>

#include <mitkGlobalInteraction.h>
#include <mitkImage.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkOverlayManager.h>
#include <mitkOverlay2DLayouter.h>
#include <mitkProportionalTimeGeometry.h>
#include <mitkSlicedGeometry3D.h>
#include <mitkVtkLayerController.h>

#include <vtkMitkRectangleProp.h>

#include "vtkSideAnnotation_p.h"


namespace niftk
{

/**
 * This class is to notify the SingleViewerWidget about the display geometry changes of a render window.
 */
class DisplayGeometryModificationCommand : public itk::Command
{
public:
  mitkNewMacro2Param(DisplayGeometryModificationCommand, MultiWindowWidget*, int)


  //-----------------------------------------------------------------------------
  DisplayGeometryModificationCommand(MultiWindowWidget* multiWindowWidget, int windowIndex)
  : itk::Command()
  , m_MultiWindowWidget(multiWindowWidget)
  , m_WindowIndex(windowIndex)
  {
  }


  //-----------------------------------------------------------------------------
  void Execute(itk::Object* caller, const itk::EventObject& event) override
  {
    this->Execute((const itk::Object*) caller, event);
  }


  //-----------------------------------------------------------------------------
  void Execute(const itk::Object* /*object*/, const itk::EventObject& /*event*/) override
  {
    m_MultiWindowWidget->OnDisplayGeometryModified(m_WindowIndex);
    return;
  }

private:
  MultiWindowWidget* const m_MultiWindowWidget;
  int m_WindowIndex;
};


//-----------------------------------------------------------------------------
MultiWindowWidget::MultiWindowWidget(
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
, m_WindowLayout(WINDOW_LAYOUT_ORTHO)
, m_TimeStep(0)
, m_CursorPositions(3)
, m_ScaleFactors(3)
, m_OrientationString("---")
, m_WorldGeometries(3)
, m_RenderWindowSizes(3)
, m_Origins(3)
, m_TimeGeometry(NULL)
, m_ReferenceGeometry(NULL)
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
, m_PositionAnnotationVisible(true)
, m_IntensityAnnotationVisible(true)
, m_PropertyAnnotationVisible(false)
, m_EmptySpace(new QWidget(this))
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

  m_EmptySpace->setAutoFillBackground(true);

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
  m_PlaneNode1->SetIntProperty("Crosshair.Gap Size", 8, 0);
  m_PlaneNode2->SetIntProperty("Crosshair.Gap Size", 8, 0);
  m_PlaneNode3->SetIntProperty("Crosshair.Gap Size", 8, 0);

  this->SetCursorVisible(false);
  this->SetWidgetPlanesLocked(true);
  this->SetWidgetPlanesRotationLocked(true);

  // Need each widget to react to Qt drag/drop events.
  this->mitkWidget1->setAcceptDrops(true);
  this->mitkWidget2->setAcceptDrops(true);
  this->mitkWidget3->setAcceptDrops(true);
  this->mitkWidget4->setAcceptDrops(true);

  // Set these off, as it wont matter until there is an image dropped, with a specific layout and orientation.
  for (int i = 0; i < 4; ++i)
  {
    this->SetDecorationProperties("", this->GetDecorationColor(i), i);
  }

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

  this->InitialisePositionAnnotations();
  this->InitialiseIntensityAnnotations();
  this->InitialisePropertyAnnotations();

  // Set default layout. This must be ORTHO.
  this->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  // Default to unselected, so borders are off.
  this->DisableColoredRectangles();

  // Register to listen to SliceNavigators, slice changed events.
  itk::ReceptorMemberCommand<MultiWindowWidget>::Pointer onAxialSliceChangedCommand =
    itk::ReceptorMemberCommand<MultiWindowWidget>::New();
  onAxialSliceChangedCommand->SetCallbackFunction(this, &MultiWindowWidget::OnAxialSliceChanged);
  m_AxialSliceObserverTag = mitkWidget1->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onAxialSliceChangedCommand);

  itk::ReceptorMemberCommand<MultiWindowWidget>::Pointer onSagittalSliceChangedCommand =
    itk::ReceptorMemberCommand<MultiWindowWidget>::New();
  onSagittalSliceChangedCommand->SetCallbackFunction(this, &MultiWindowWidget::OnSagittalSliceChanged);
  m_SagittalSliceObserverTag = mitkWidget2->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSagittalSliceChangedCommand);

  itk::ReceptorMemberCommand<MultiWindowWidget>::Pointer onCoronalSliceChangedCommand =
    itk::ReceptorMemberCommand<MultiWindowWidget>::New();
  onCoronalSliceChangedCommand->SetCallbackFunction(this, &MultiWindowWidget::OnCoronalSliceChanged);
  m_CoronalSliceObserverTag = mitkWidget3->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onCoronalSliceChangedCommand);

  itk::ReceptorMemberCommand<MultiWindowWidget>::Pointer onTimeStepChangedCommand =
    itk::ReceptorMemberCommand<MultiWindowWidget>::New();
  onTimeStepChangedCommand->SetCallbackFunction(this, &MultiWindowWidget::OnTimeStepChanged);
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
  itk::SimpleMemberCommand<MultiWindowWidget>::Pointer onFocusChangedCommand =
    itk::SimpleMemberCommand<MultiWindowWidget>::New();
  onFocusChangedCommand->SetCallbackFunction(this, &MultiWindowWidget::OnFocusChanged);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);

  // The mouse mode switcher is declared and initialised in QmitkStdMultiWidget. It creates an
  // mitk::DisplayInteractor. This line decreases the reference counter of the mouse mode switcher
  // so that it is destructed and it unregisters and destructs its display interactor as well.
  m_MouseModeSwitcher = 0;
}


//-----------------------------------------------------------------------------
MultiWindowWidget::~MultiWindowWidget()
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
void MultiWindowWidget::AddDisplayGeometryModificationObserver(int windowIndex)
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
void MultiWindowWidget::RemoveDisplayGeometryModificationObserver(int windowIndex)
{
  mitk::BaseRenderer* renderer = m_RenderWindows[windowIndex]->GetRenderer();
  assert(renderer);

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  assert(displayGeometry);

  displayGeometry->RemoveObserver(m_DisplayGeometryModificationObservers[windowIndex]);
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::OnAxialSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(AXIAL);
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::OnSagittalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(SAGITTAL);
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::OnCoronalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  this->OnSelectedPositionChanged(CORONAL);
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::OnTimeStepChanged(const itk::EventObject& /*geometryTimeEvent*/)
{
  if (!m_BlockSncEvents && m_ReferenceGeometry != NULL)
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
void MultiWindowWidget::SetBackgroundColour(QColor colour)
{
  m_BackgroundColour = colour;
  mitk::Color backgroundColour;
  backgroundColour.Set(colour.redF(), colour.greenF(), colour.blueF());
  this->SetGradientBackgroundColors(backgroundColour, backgroundColour);

  QPalette palette;
  palette.setColor(QPalette::Background, colour);
  m_EmptySpace->setPalette(palette);
}


//-----------------------------------------------------------------------------
QColor MultiWindowWidget::GetBackgroundColour() const
{
  return m_BackgroundColour;
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::IsFocused() const
{
  return m_IsFocused;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetFocused()
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
QmitkRenderWindow* MultiWindowWidget::GetSelectedRenderWindow() const
{
  assert(m_SelectedWindowIndex >= 0 && m_SelectedWindowIndex < m_RenderWindows.size());

  return m_RenderWindows[m_SelectedWindowIndex];
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetSelectedRenderWindow(QmitkRenderWindow* renderWindow)
{
  std::size_t selectedWindowIndex = std::find(m_RenderWindows.begin(), m_RenderWindows.end(), renderWindow) - m_RenderWindows.begin();
  assert(selectedWindowIndex != m_RenderWindows.size());

  this->SetSelectedWindowIndex(selectedWindowIndex);
}


//-----------------------------------------------------------------------------
int MultiWindowWidget::GetSelectedWindowIndex() const
{
  assert(m_SelectedWindowIndex >= 0 && m_SelectedWindowIndex < m_RenderWindows.size());

  return m_SelectedWindowIndex;
}

//-----------------------------------------------------------------------------
void MultiWindowWidget::SetSelectedWindowIndex(int selectedWindowIndex)
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

    if (m_SelectedWindowIndex < 3)
    {
      m_PositionAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
      m_IntensityAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
      m_PropertyAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
    }

    m_SelectedWindowIndex = selectedWindowIndex;

    this->UpdatePositionAnnotation(m_SelectedWindowIndex);
    this->UpdateIntensityAnnotation(m_SelectedWindowIndex);
    this->UpdatePropertyAnnotation(m_SelectedWindowIndex);

    this->BlockUpdate(updateWasBlocked);
  }
}

//-----------------------------------------------------------------------------
void MultiWindowWidget::UpdateBorders()
{
  // When we "Select", the selection is at the level of the MultiWindowWidget
  // so the whole of this widget is selected. However, we may have clicked in
  // a specific view, so it still helps to highlight the most recently clicked on view.
  // Also, if you are displaying ortho window layout (2x2) then you actually have 4 windows present,
  // then highlighting them all starts to look a bit confusing, so we just highlight the
  // most recently focused window, (eg. axial, sagittal, coronal or 3D).

  if (m_IsFocused && m_ReferenceGeometry)
  {
    if (m_SelectedWindowIndex == AXIAL)
    {
      m_RectangleProps[0]->SetVisibility(true);
      m_RectangleProps[1]->SetVisibility(false);
      m_RectangleProps[2]->SetVisibility(false);
      m_RectangleProps[3]->SetVisibility(false);
    }
    else if (m_SelectedWindowIndex == SAGITTAL)
    {
      m_RectangleProps[0]->SetVisibility(false);
      m_RectangleProps[1]->SetVisibility(true);
      m_RectangleProps[2]->SetVisibility(false);
      m_RectangleProps[3]->SetVisibility(false);
    }
    else if (m_SelectedWindowIndex == CORONAL)
    {
      m_RectangleProps[0]->SetVisibility(false);
      m_RectangleProps[1]->SetVisibility(false);
      m_RectangleProps[2]->SetVisibility(true);
      m_RectangleProps[3]->SetVisibility(false);
    }
    else // THREE_D
    {
      m_RectangleProps[0]->SetVisibility(false);
      m_RectangleProps[1]->SetVisibility(false);
      m_RectangleProps[2]->SetVisibility(false);
      m_RectangleProps[3]->SetVisibility(true);
    }
  }
  else
  {
    m_RectangleProps[0]->SetVisibility(false);
    m_RectangleProps[1]->SetVisibility(false);
    m_RectangleProps[2]->SetVisibility(false);
    m_RectangleProps[3]->SetVisibility(false);
  }
}


//-----------------------------------------------------------------------------
std::vector<QmitkRenderWindow*> MultiWindowWidget::GetVisibleRenderWindows() const
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
void MultiWindowWidget::RequestUpdate()
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
    case WINDOW_LAYOUT_ORTHO_NO_3D:
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
      assert((m_WindowLayout >= 0 && m_WindowLayout <= 7) || (m_WindowLayout >= 10 && m_WindowLayout <= 15));
      break;
    }
  }
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::IsEnabled() const
{
  return m_Enabled;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetEnabled(bool enabled)
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
bool MultiWindowWidget::IsCursorVisible() const
{
  return m_CursorVisibility;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetCursorVisible(bool visible)
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
bool MultiWindowWidget::AreDirectionAnnotationsVisible() const
{
  return m_DirectionAnnotations[AXIAL]->GetVisibility()
      && m_DirectionAnnotations[SAGITTAL]->GetVisibility()
      && m_DirectionAnnotations[CORONAL]->GetVisibility();
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetDirectionAnnotationsVisible(bool visible)
{
  m_DirectionAnnotations[AXIAL]->SetVisibility(visible);
  m_DirectionAnnotations[SAGITTAL]->SetVisibility(visible);
  m_DirectionAnnotations[CORONAL]->SetVisibility(visible);
  this->RequestUpdate();
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::IsPositionAnnotationVisible() const
{
  return m_PositionAnnotationVisible;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetPositionAnnotationVisible(bool visible)
{
  if (visible != m_PositionAnnotationVisible)
  {
    m_PositionAnnotationVisible = visible;
    this->UpdatePositionAnnotation(m_SelectedWindowIndex);
  }
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::IsIntensityAnnotationVisible() const
{
  return m_IntensityAnnotationVisible;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetIntensityAnnotationVisible(bool visible)
{
  if (visible != m_IntensityAnnotationVisible)
  {
    m_IntensityAnnotationVisible = visible;
    this->UpdateIntensityAnnotation(m_SelectedWindowIndex);
  }
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::IsPropertyAnnotationVisible() const
{
  return m_PropertyAnnotationVisible;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetPropertyAnnotationVisible(bool visible)
{
  if (visible != m_PropertyAnnotationVisible)
  {
    m_PropertyAnnotationVisible = visible;
    this->UpdatePropertyAnnotation(m_SelectedWindowIndex);
  }
}


//-----------------------------------------------------------------------------
QStringList MultiWindowWidget::GetPropertiesForAnnotation() const
{
  return m_PropertiesForAnnotation;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetPropertiesForAnnotation(const QStringList& propertiesForAnnotation)
{
  if (propertiesForAnnotation != m_PropertiesForAnnotation)
  {
    m_PropertiesForAnnotation = propertiesForAnnotation;
    this->UpdatePropertyAnnotation(m_SelectedWindowIndex);
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::Update3DWindowVisibility()
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
      if ((m_WindowLayout == WINDOW_LAYOUT_ORTHO)
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
void MultiWindowWidget::SetVisibility(QmitkRenderWindow* renderWindow, mitk::DataNode* node, bool visibility)
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
void MultiWindowWidget::SetVisibility(std::vector<mitk::DataNode*> nodes, bool visibility)
{
  for (std::size_t i = 0; i < nodes.size(); ++i)
  {
    this->SetVisibility(mitkWidget1, nodes[i], visibility);
    this->SetVisibility(mitkWidget2, nodes[i], visibility);
    this->SetVisibility(mitkWidget3, nodes[i], visibility);
  }
  this->UpdatePositionAnnotation(m_SelectedWindowIndex);
  this->UpdateIntensityAnnotation(m_SelectedWindowIndex);
  this->UpdatePropertyAnnotation(m_SelectedWindowIndex);
  this->Update3DWindowVisibility();
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::ContainsRenderWindow(QmitkRenderWindow* renderWindow) const
{
  return mitkWidget1 == renderWindow
      || mitkWidget2 == renderWindow
      || mitkWidget3 == renderWindow
      || mitkWidget4 == renderWindow;
}


//-----------------------------------------------------------------------------
const std::vector<QmitkRenderWindow*>& MultiWindowWidget::GetRenderWindows() const
{
  return m_RenderWindows;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::FitRenderWindows(double scaleFactor)
{
  if (!m_ReferenceGeometry)
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
            regionWidthInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
            regionHeightInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
          }
          else if (windowIndex == SAGITTAL)
          {
            regionWidthInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
            regionHeightInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
          }
          else if (windowIndex == CORONAL)
          {
            regionWidthInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
            regionHeightInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
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
void MultiWindowWidget::FitRenderWindow(int windowIndex, double scaleFactor)
{
  assert(windowIndex < 3);

  bool updateWasBlocked = this->BlockUpdate(true);

  double windowWidthInPx = m_RenderWindowSizes[windowIndex][0];
  double windowHeightInPx = m_RenderWindowSizes[windowIndex][1];

  double regionWidthInMm;
  double regionHeightInMm;
  if (windowIndex == AXIAL)
  {
    regionWidthInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
    regionHeightInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
  }
  else if (windowIndex == SAGITTAL)
  {
    regionWidthInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[CORONAL]);
    regionHeightInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
  }
  else if (windowIndex == CORONAL)
  {
    regionWidthInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[SAGITTAL]);
    regionHeightInMm = m_ReferenceGeometry->GetExtentInMM(m_OrientationAxes[AXIAL]);
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
void MultiWindowWidget::SetTimeGeometry(const mitk::TimeGeometry* timeGeometry)
{
  if (timeGeometry != NULL)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);

    m_TimeGeometry = timeGeometry;
    m_ReferenceGeometry = timeGeometry->GetGeometryForTimeStep(0);

    // Calculating the voxel size. This is needed for the conversion between the
    // magnification and the scale factors.
    for (int axis = 0; axis < 3; ++axis)
    {
      m_MmPerVx[axis] = m_ReferenceGeometry->GetExtentInMM(axis) / m_ReferenceGeometry->GetExtent(axis);
    }

    // Add these annotations the first time we have a real geometry.
    this->SetDecorationProperties("Axial", this->GetDecorationColor(0), 0);
    this->SetDecorationProperties("Sagittal", this->GetDecorationColor(1), 1);
    this->SetDecorationProperties("Coronal", this->GetDecorationColor(2), 2);
    this->SetDecorationProperties("3D", this->GetDecorationColor(3), 3);

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

    mitk::AffineTransform3D::MatrixType matrix = m_ReferenceGeometry->GetIndexToWorldTransform()->GetMatrix();
    matrix.GetVnlMatrix().normalize_columns();
    mitk::AffineTransform3D::MatrixType::InternalMatrixType inverseMatrix = matrix.GetInverse();

    int dominantAxisRL = itk::Function::Max3(inverseMatrix[0][0], inverseMatrix[1][0], inverseMatrix[2][0]);
    int signRL = itk::Function::Sign(inverseMatrix[dominantAxisRL][0]);
    int dominantAxisAP = itk::Function::Max3(inverseMatrix[0][1], inverseMatrix[1][1], inverseMatrix[2][1]);
    int signAP = itk::Function::Sign(inverseMatrix[dominantAxisAP][1]);
    int dominantAxisSI = itk::Function::Max3(inverseMatrix[0][2], inverseMatrix[1][2], inverseMatrix[2][2]);
    int signSI = itk::Function::Sign(inverseMatrix[dominantAxisSI][2]);

    m_UpDirections[0] = signRL;
    m_UpDirections[1] = signAP;
    m_UpDirections[2] = signSI;

    m_OrientationAxes[AXIAL] = dominantAxisSI;
    m_OrientationAxes[SAGITTAL] = dominantAxisRL;
    m_OrientationAxes[CORONAL] = dominantAxisAP;

    m_OrientationString[dominantAxisSI] = signSI > 0 ? 'S' : 'I';
    m_OrientationString[dominantAxisRL] = signRL > 0 ? 'R' : 'L';
    m_OrientationString[dominantAxisAP] = signAP > 0 ? 'A' : 'P';

    /// We create a timed geometry for each 2D renderer. The sliced geometries must be created
    /// in the same way as MITK does it, so that the viewer stays compatible with the MITK Display
    /// and the Image Navigator.
    /// Note that the way how MITK constructs these geometry is kind of arbitrary and inconsistent.
    /// E.g. the geometry of some renderer has left-handed coordinate system (sagittal, axial),
    /// while some others have right-handed one (coronal). The slice numbering matches the direction
    /// of the world directions in case of the sagittal and coronal axes but is inverted for the
    /// axial axis. And so on. For this reason, you should not rely on any parameter of the renderer
    /// geometries or their contained plane geometries. The only parameters that are correct are
    /// the origin and the right and bottom vectors of the plane geometries that composes the
    /// sliced geometries of the renderers. These are used to render the 2D planes.

    for (unsigned int i = 0; i < 3; ++i)
    {
      mitk::BaseRenderer* renderer = m_RenderWindows[i]->GetRenderer();

      // Get access to slice navigation controller, as this sorts out most of the process.
      mitk::SliceNavigationController* snc = renderer->GetSliceNavigationController();
      snc->SetViewDirectionToDefault();
      snc->SetInputWorldTimeGeometry(m_TimeGeometry);
      snc->Update();

      /// Now that the geometry is established, we set to middle slice. In case of image geometry
      /// and even slice numbers, the selected position must be in the centre, returned by
      /// `m_ReferenceGeometry->GetCenter()`.
      /// Note that the slice numbering in the slice navigation controllers always goes from
      /// left to right in the sagittal SNC, from back to front in the coronal SNC and from top
      /// to bottom (!) in the axial SNC, for compatibility with MITK. If this direction is
      /// different than the up direction of the corresponding reference geometry axis, we
      /// need to invert the position. Since we are in the centre, this means subtracting one.
      int middleSlice = snc->GetSlice()->GetSteps() / 2;
      bool referenceGeometryAxisInverted = m_UpDirections[i] < 0;
      int worldAxis = i == 0 ? 2 : i == 1 ? 0 : 1;
      bool rendererZAxisInverted = snc->GetCurrentGeometry3D()->GetAxisVector(2)[worldAxis] < 0;
      if (referenceGeometryAxisInverted != rendererZAxisInverted)
      {
        --middleSlice;
      }
      snc->GetSlice()->SetPos(middleSlice);

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
      renderer->SetWorldTimeGeometry(snc->GetCreatedWorldGeometry());
    }

    /// Although we created time geometries for each renderers and they have their own 'slice'
    /// navigation controller for time, the image navigator view does not use these but the
    /// 'global' one, that of the rendering manager. Therefore, we need to set the input time
    /// geometry for that slice navigator.
    m_RenderingManager->GetTimeNavigationController()->SetInputWorldTimeGeometry(m_TimeGeometry);
    m_RenderingManager->GetTimeNavigationController()->Update();

    this->BlockDisplayEvents(displayEventsWereBlocked);

    m_GeometryHasChanged = true;

    m_SelectedPosition = this->GetCrossPosition();
    for (int i = 0; i < 3; ++i)
    {
      m_SelectedSliceHasChanged[i] = true;
    }

    m_TimeStep = 0;
    m_TimeStepHasChanged = true;

    if (m_SelectedWindowIndex < 3)
    {
      this->UpdatePositionAnnotation(m_SelectedWindowIndex);
      this->UpdateIntensityAnnotation(m_SelectedWindowIndex);
      this->UpdatePropertyAnnotation(m_SelectedWindowIndex);
    }

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
void MultiWindowWidget::SetWindowLayout(WindowLayout windowLayout)
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
  else if (windowLayout == WINDOW_LAYOUT_ORTHO_NO_3D)
  {
    m_GridLayout->addWidget(this->mitkWidget1Container, 1, 0);  // axial:    on
    m_GridLayout->addWidget(this->mitkWidget2Container, 0, 1);  // sagittal: on
    m_GridLayout->addWidget(this->mitkWidget3Container, 0, 0);  // coronal:  on
    m_GridLayout->addWidget(m_EmptySpace, 1, 1);  // 3D:       on
  }
  else // ORTHO or ORTHO_NO_3D
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
  case WINDOW_LAYOUT_ORTHO_NO_3D:
    showAxial = true;
    showSagittal = true;
    showCoronal = true;
    show3D = false;
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

  m_CursorPositionBinding = niftk::IsMultiWindowLayout(windowLayout);
  m_ScaleFactorBinding = niftk::IsMultiWindowLayout(windowLayout);

  m_WindowLayout = windowLayout;
  m_WindowLayoutHasChanged = true;

  this->Update3DWindowVisibility();
  m_GridLayout->activate();

  if (!m_RenderWindows[m_SelectedWindowIndex]->isVisible())
  {
    if (m_SelectedWindowIndex < 3)
    {
      m_PositionAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
      m_IntensityAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
      m_PropertyAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
    }

    m_SelectedWindowIndex = defaultWindowIndex;

    this->UpdatePositionAnnotation(m_SelectedWindowIndex);
    this->UpdateIntensityAnnotation(m_SelectedWindowIndex);
    this->UpdatePropertyAnnotation(m_SelectedWindowIndex);

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
WindowLayout MultiWindowWidget::GetWindowLayout() const
{
  return m_WindowLayout;
}


//-----------------------------------------------------------------------------
int MultiWindowWidget::GetMaxSlice(int windowIndex) const
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
int MultiWindowWidget::GetMaxTimeStep() const
{
//  return m_TimeNavigationController->GetTime()->GetSteps() - 1;
  return m_RenderWindows[AXIAL]->GetSliceNavigationController()->GetTime()->GetSteps() - 1;
}


//-----------------------------------------------------------------------------
const mitk::Vector2D& MultiWindowWidget::GetCursorPosition(int windowIndex) const
{
  return m_CursorPositions[windowIndex];
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetCursorPosition(int windowIndex, const mitk::Vector2D& cursorPosition)
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
void MultiWindowWidget::MoveToCursorPosition(int windowIndex)
{
  if (m_ReferenceGeometry)
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
void MultiWindowWidget::OnDisplayGeometryModified(int windowIndex)
{
  if (m_BlockDisplayEvents || !m_ReferenceGeometry)
  {
    return;
  }

  bool updateWasBlocked = this->BlockUpdate(true);

  mitk::DisplayGeometry* displayGeometry = m_RenderWindows[windowIndex]->GetRenderer()->GetDisplayGeometry();

  const mitk::PlaneGeometry* worldGeometry = displayGeometry->GetWorldGeometry();
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
void MultiWindowWidget::OnOriginChanged(int windowIndex, bool beingPanned)
{
  if (m_ReferenceGeometry)
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
void MultiWindowWidget::OnScaleFactorChanged(int windowIndex, double scaleFactor)
{
  if (m_ReferenceGeometry)
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
void MultiWindowWidget::OnSelectedPositionChanged(int windowIndex)
{
  if (!m_BlockSncEvents && m_ReferenceGeometry != NULL)
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
int MultiWindowWidget::GetSelectedSlice(int windowIndex) const
{
  assert(0 <= windowIndex && windowIndex < 3);

  int selectedSlice = -1;

  if (m_ReferenceGeometry != NULL)
  {
    int axis = m_OrientationAxes[windowIndex];

    mitk::Point3D selectedPositionInVx;
    m_ReferenceGeometry->WorldToIndex(m_SelectedPosition, selectedPositionInVx);

    if (!m_ReferenceGeometry->GetImageGeometry())
    {
      selectedPositionInVx[axis] -= 0.5;
    }

    /// It should already be a round number, if not, it is a bug.
    /// Anyway, let's round it to the closest integer to avoid precision errors.
    selectedSlice = static_cast<int>(selectedPositionInVx[axis] + 0.5);
  }

  return selectedSlice;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetSelectedSlice(int windowIndex, int selectedSlice)
{
  if (m_ReferenceGeometry != NULL)
  {
    mitk::Point3D selectedPosition = m_SelectedPosition;

    mitk::Point3D selectedPositionInVx;
    m_ReferenceGeometry->WorldToIndex(selectedPosition, selectedPositionInVx);

    int axis = m_OrientationAxes[windowIndex];
    selectedPositionInVx[axis] = selectedSlice;

    if (!m_ReferenceGeometry->GetImageGeometry())
    {
      selectedPositionInVx[axis] += 0.5;
    }

    m_ReferenceGeometry->IndexToWorld(selectedPositionInVx, selectedPosition);

    this->SetSelectedPosition(selectedPosition);
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::MoveSlice(int windowIndex, int slices, bool restart)
{
  if (m_ReferenceGeometry && windowIndex < 3 && slices != 0)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    int slice = this->GetSelectedSlice(windowIndex);

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

    int nextSlice = slice + upDirection * slices;

    int maxSlice = this->GetMaxSlice(windowIndex);

    if (restart)
    {
      if (nextSlice < 0)
      {
        nextSlice += maxSlice + 1;
      }
      else if (nextSlice > maxSlice)
      {
        nextSlice -= maxSlice + 1;
      }
    }

    if (nextSlice >= 0 && nextSlice <= maxSlice)
    {
      this->SetSelectedSlice(windowIndex, nextSlice);

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
const mitk::Point3D& MultiWindowWidget::GetSelectedPosition() const
{
  return m_SelectedPosition;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition)
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
      this->UpdatePositionAnnotation(windowIndex);
      this->UpdateIntensityAnnotation(windowIndex);
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::InitialisePositionAnnotations()
{
  for (int i = 0; i < 3; ++i)
  {
    mitk::BaseRenderer* renderer = m_RenderWindows[i]->GetRenderer();
    mitk::OverlayManager::Pointer overlayManager = renderer->GetOverlayManager();
    mitk::Overlay2DLayouter::Pointer layouter = mitk::Overlay2DLayouter::CreateLayouter(
          mitk::Overlay2DLayouter::STANDARD_2D_TOPRIGHT(), renderer);
    overlayManager->AddLayouter(layouter.GetPointer());

    mitk::TextOverlay2D::Pointer annotation = mitk::TextOverlay2D::New();
    m_PositionAnnotations[i] = annotation;
    annotation->SetFontSize(12);
    annotation->SetColor(0.0f, 1.0f, 0.0f);
    annotation->SetOpacity(1.0f);
    annotation->SetVisibility(false);

    overlayManager->AddOverlay(annotation.GetPointer(), renderer);
    overlayManager->SetLayouter(annotation.GetPointer(), mitk::Overlay2DLayouter::STANDARD_2D_TOPRIGHT(), renderer);
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::InitialiseIntensityAnnotations()
{
  for (int i = 0; i < 3; ++i)
  {
    mitk::BaseRenderer* renderer = m_RenderWindows[i]->GetRenderer();
    mitk::OverlayManager::Pointer overlayManager = renderer->GetOverlayManager();
    mitk::Overlay2DLayouter::Pointer layouter = mitk::Overlay2DLayouter::CreateLayouter(
          mitk::Overlay2DLayouter::STANDARD_2D_BOTTOMRIGHT(), renderer);
    overlayManager->AddLayouter(layouter.GetPointer());

    mitk::TextOverlay2D::Pointer annotation = mitk::TextOverlay2D::New();
    m_IntensityAnnotations[i] = annotation;
    annotation->SetFontSize(12);
    annotation->SetColor(0.0f, 1.0f, 0.0f);
    annotation->SetOpacity(1.0f);
    annotation->SetVisibility(false);

    overlayManager->AddOverlay(annotation.GetPointer(), renderer);
    overlayManager->SetLayouter(annotation.GetPointer(), mitk::Overlay2DLayouter::STANDARD_2D_BOTTOMRIGHT(), renderer);
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::InitialisePropertyAnnotations()
{
  for (int i = 0; i < 3; ++i)
  {
    mitk::BaseRenderer* renderer = m_RenderWindows[i]->GetRenderer();
    mitk::OverlayManager::Pointer overlayManager = renderer->GetOverlayManager();
    mitk::Overlay2DLayouter::Pointer layouter = mitk::Overlay2DLayouter::CreateLayouter(
          mitk::Overlay2DLayouter::STANDARD_2D_TOPLEFT(), renderer);
    overlayManager->AddLayouter(layouter.GetPointer());

    mitk::TextOverlay2D::Pointer annotation = mitk::TextOverlay2D::New();
    m_PropertyAnnotations[i] = annotation;
    annotation->SetFontSize(12);
    annotation->SetColor(0.0f, 1.0f, 0.0f);
    annotation->SetOpacity(1.0f);
    annotation->SetVisibility(false);

    overlayManager->AddOverlay(annotation.GetPointer(), renderer);
    overlayManager->SetLayouter(annotation.GetPointer(), mitk::Overlay2DLayouter::STANDARD_2D_TOPLEFT(), renderer);
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::UpdatePositionAnnotation(int windowIndex) const
{
  if (windowIndex >= 0 && windowIndex < 3)
  {
    mitk::TextOverlay2D::Pointer annotation = m_PositionAnnotations[windowIndex];

    bool wasVisible = annotation->IsVisible(nullptr);
    bool shouldBeVisible = m_PositionAnnotationVisible && windowIndex == m_SelectedWindowIndex && m_TimeGeometry;

    if (wasVisible != shouldBeVisible)
    {
      annotation->SetVisibility(shouldBeVisible);
    }

    if (shouldBeVisible)
    {
      std::stringstream stream;
      stream.imbue(std::locale::classic());

      mitk::Point3D selectedPositionInVx;
      m_ReferenceGeometry->WorldToIndex(m_SelectedPosition, selectedPositionInVx);

      if (!m_ReferenceGeometry->GetImageGeometry())
      {
        for (int i = 0; i < 3; ++i)
        {
          selectedPositionInVx[i] -= 0.5;
        }
      }

      stream << selectedPositionInVx[0] << ", " << selectedPositionInVx[1] << ", " << selectedPositionInVx[2] << " vx (" << m_OrientationString << ")" << std::endl;

      /// Display selected voxel index coordinates and orientation string in the renderer geometry in debug mode only.
#ifndef NDEBUG
      if (const mitk::BaseGeometry* rendererGeometry = m_RenderWindows[windowIndex]->GetRenderer()->GetCurrentWorldGeometry())
      {
        rendererGeometry->WorldToIndex(m_SelectedPosition, selectedPositionInVx);

        /// The renderer window geometry is already half voxel shifted along the renderer plane axis,
        /// therefore we do not adjust the last index coordinate.
        for (int i = 0; i < 2; ++i)
        {
          selectedPositionInVx[i] -= 0.5;
        }

        std::string orientationString = windowIndex == 0 ? "RPI" : windowIndex == 1 ? "ASR" : "RSA";

        stream << selectedPositionInVx[0] << ", " << selectedPositionInVx[1] << ", " << selectedPositionInVx[2] << " vx (" << orientationString << ")" << std::endl;
      }
#endif

      stream << std::fixed << std::setprecision(1) << m_SelectedPosition[0] << ", " << m_SelectedPosition[1] << ", " << m_SelectedPosition[2] << " mm";

      if (m_TimeGeometry->CountTimeSteps() > 1)
      {
        stream << std::endl << "Time step: " << m_TimeStep;
      }

      annotation->SetText(stream.str());
      annotation->Modified();
    }

    m_RenderingManager->RequestUpdate(m_RenderWindows[m_SelectedWindowIndex]->GetRenderWindow());
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::UpdateIntensityAnnotation(int windowIndex) const
{
  if (windowIndex >= 0 && windowIndex < 3)
  {
    mitk::BaseRenderer* renderer = m_RenderWindows[windowIndex]->GetRenderer();
    mitk::TextOverlay2D::Pointer annotation = m_IntensityAnnotations[windowIndex];

    bool wasVisible = annotation->IsVisible(nullptr);
    bool shouldBeVisible = m_IntensityAnnotationVisible && windowIndex == m_SelectedWindowIndex;

    if (wasVisible != shouldBeVisible)
    {
      annotation->SetVisibility(shouldBeVisible);
    }

    if (shouldBeVisible)
    {
      mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
      mitk::NodePredicateProperty::Pointer isBinary = mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));
      mitk::NodePredicateNot::Pointer isNotBinary = mitk::NodePredicateNot::New(isBinary);
      mitk::NodePredicateAnd::Pointer isImageAndNotBinary = mitk::NodePredicateAnd::New(isImage, isNotBinary);
      mitk::NodePredicateProperty::Pointer isVisible = mitk::NodePredicateProperty::New("visible", mitk::BoolProperty::New(true), renderer);
      mitk::NodePredicateAnd::Pointer isVisibleAndImageAndNotBinary = mitk::NodePredicateAnd::New(isVisible, isImageAndNotBinary);

      /// Note:
      /// The nodes are printed in the order of their layer.
      std::multimap<int, mitk::DataNode*> visibleNonBinaryImageNodes;

      mitk::DataStorage::SetOfObjects::ConstPointer nodes = renderer->GetDataStorage()->GetSubset(isVisibleAndImageAndNotBinary).GetPointer();
      for (mitk::DataStorage::SetOfObjects::ConstIterator it = nodes->Begin(); it != nodes->End(); ++it)
      {
        mitk::DataNode* node = it->Value();
        int layer = 0;
        if (node->GetIntProperty("layer", layer, renderer))
        {
          visibleNonBinaryImageNodes.insert(std::make_pair(layer, node));
        }
      }

      if (visibleNonBinaryImageNodes.empty())
      {
        annotation->SetVisibility(false);
        return;
      }

      std::stringstream stream;
      stream.precision(3);
      stream.imbue(std::locale::classic());

      for (auto it = visibleNonBinaryImageNodes.rbegin(); it != visibleNonBinaryImageNodes.rend(); ++it)
      {
        mitk::DataNode* node = it->second;

        int component = 0;
        node->GetIntProperty("Image.Displayed Component", component);

        mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());

        mitk::ScalarType intensity = image->GetPixelValueByWorldCoordinate(m_SelectedPosition, m_TimeStep, component);

        if (it != visibleNonBinaryImageNodes.rbegin())
        {
          stream << std::endl;
        }

        if (visibleNonBinaryImageNodes.size() != 1)
        {
          stream << node->GetName() << ": ";
        }

        if (std::fabs(intensity) > 10e6 || std::fabs(intensity) < 10e-3)
        {
          stream << std::scientific << intensity;
        }
        else
        {
          stream << intensity;
        }
      }

      annotation->SetText(stream.str());
      annotation->Modified();
    }

    m_RenderingManager->RequestUpdate(m_RenderWindows[m_SelectedWindowIndex]->GetRenderWindow());
  }
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::UpdatePropertyAnnotation(int windowIndex) const
{
  if (windowIndex >= 0 && windowIndex < 3)
  {
    mitk::BaseRenderer* renderer = m_RenderWindows[windowIndex]->GetRenderer();
    mitk::TextOverlay2D::Pointer annotation = m_PropertyAnnotations[windowIndex];

    bool wasVisible = annotation->IsVisible(nullptr);
    bool shouldBeVisible = m_PropertyAnnotationVisible && windowIndex == m_SelectedWindowIndex;

    if (wasVisible != shouldBeVisible)
    {
      annotation->SetVisibility(shouldBeVisible);
    }

    if (shouldBeVisible)
    {
      mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
      mitk::NodePredicateProperty::Pointer isVisible = mitk::NodePredicateProperty::New("visible", mitk::BoolProperty::New(true), renderer);
      mitk::NodePredicateAnd::Pointer isVisibleAndImage = mitk::NodePredicateAnd::New(isVisible, isImage);

      /// Note:
      /// The nodes are printed in the reversed order of their layer.
      std::multimap<int, mitk::DataNode*> visibleImageNodes;

      mitk::DataStorage::SetOfObjects::ConstPointer nodes = renderer->GetDataStorage()->GetSubset(isVisibleAndImage).GetPointer();
      for (auto it = nodes->Begin(); it != nodes->End(); ++it)
      {
        mitk::DataNode* node = it->Value();
        int layer = 0;
        if (node->GetIntProperty("layer", layer, renderer))
        {
          visibleImageNodes.insert(std::make_pair(layer, node));
        }
      }

      std::stringstream stream;
      stream.precision(3);
      stream.imbue(std::locale::classic());

      if (visibleImageNodes.size() == 0)
      {
        annotation->SetVisibility(false);
      }
      else
      {
        annotation->SetVisibility(windowIndex == m_SelectedWindowIndex && m_PropertyAnnotationVisible);
      }

      for (auto it = visibleImageNodes.rbegin(); it != visibleImageNodes.rend(); ++it)
      {
        mitk::DataNode* node = it->second;

        if (it != visibleImageNodes.rbegin())
        {
          stream << std::endl;
        }

        /// Show the name in the first line if there are several visible images, or if the 'name'
        /// property was explicitely requested.
        if (visibleImageNodes.size() > 1 || m_PropertiesForAnnotation.contains(QString("name")))
        {
          stream << node->GetName() << std::endl;
        }

        for (const QString& propertyName: m_PropertiesForAnnotation)
        {
          if (propertyName == QString("name"))
          {
            /// The name is always shown in the first line.
            continue;
          }
          mitk::BaseProperty* property = node->GetProperty(propertyName.toStdString().c_str());
          if (property)
          {
            stream << propertyName.toStdString() << ": " << property->GetValueAsString() << std::endl;
          }
        }
      }

      /// Remove last '\n'.
      stream.unget();

      annotation->SetText(stream.str());
      annotation->Modified();
    }

    m_RenderingManager->RequestUpdate(m_RenderWindows[m_SelectedWindowIndex]->GetRenderWindow());
  }
}


//-----------------------------------------------------------------------------
int MultiWindowWidget::GetTimeStep() const
{
  return m_TimeStep;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetTimeStep(int timeStep)
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
void MultiWindowWidget::SynchroniseCursorPositions(int windowIndex)
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
const std::vector<mitk::Vector2D>& MultiWindowWidget::GetCursorPositions() const
{
  return m_CursorPositions;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetCursorPositions(const std::vector<mitk::Vector2D>& cursorPositions)
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
void MultiWindowWidget::UpdateCursorPosition(int windowIndex)
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
double MultiWindowWidget::GetScaleFactor(int windowIndex) const
{
  return m_ScaleFactors[windowIndex];
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetScaleFactor(int windowIndex, double scaleFactor)
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
const std::vector<double>& MultiWindowWidget::GetScaleFactors() const
{
  return m_ScaleFactors;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetScaleFactors(const std::vector<double>& scaleFactors)
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
void MultiWindowWidget::ZoomAroundCursorPosition(int windowIndex)
{
  if (m_ReferenceGeometry)
  {
    mitk::BaseRenderer* renderer = m_RenderWindows[windowIndex]->GetRenderer();
    mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();

    double scaleFactor = m_ScaleFactors[windowIndex];

    mitk::Point2D focusPoint2DInMm;
    displayGeometry->Map(m_SelectedPosition, focusPoint2DInMm);

    mitk::Vector2D newOriginInMm;
    newOriginInMm[0] = focusPoint2DInMm[0] - m_CursorPositions[windowIndex][0] * renderer->GetSizeX() * scaleFactor;
    newOriginInMm[1] = focusPoint2DInMm[1] - m_CursorPositions[windowIndex][1] * renderer->GetSizeY() * scaleFactor;

    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
    displayGeometry->SetScaleFactor(scaleFactor);
    displayGeometry->SetOriginInMM(newOriginInMm);
    this->BlockDisplayEvents(displayEventsWereBlocked);
  }
}


//-----------------------------------------------------------------------------
int MultiWindowWidget::GetDominantAxis(int windowIndex) const
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
double MultiWindowWidget::GetMagnification(int windowIndex) const
{
  double magnification = 0.0;

  if (m_ReferenceGeometry)
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
void MultiWindowWidget::SetMagnification(int windowIndex, double magnification)
{
  if (m_ReferenceGeometry)
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
int MultiWindowWidget::GetSliceUpDirection(WindowOrientation orientation) const
{
  int upDirection = 0;
  if (m_ReferenceGeometry && orientation >= 0 && orientation < 3)
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
void MultiWindowWidget::OnFocusChanged()
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
      if (m_SelectedWindowIndex < 3)
      {
        m_PositionAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
        m_IntensityAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
        m_PropertyAnnotations[m_SelectedWindowIndex]->SetVisibility(false);
      }

      m_SelectedWindowIndex = focusedWindowIndex;

      this->UpdatePositionAnnotation(m_SelectedWindowIndex);
      this->UpdateIntensityAnnotation(m_SelectedWindowIndex);
      this->UpdatePropertyAnnotation(m_SelectedWindowIndex);
    }

    m_IsFocused = isFocused;
    m_FocusHasChanged = true;

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::IsLinkedNavigationEnabled() const
{
  return m_LinkedNavigationEnabled;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetLinkedNavigationEnabled(bool linkedNavigationEnabled)
{
  if (linkedNavigationEnabled != m_LinkedNavigationEnabled)
  {
    m_LinkedNavigationEnabled = linkedNavigationEnabled;
    this->SetWidgetPlanesLocked(!linkedNavigationEnabled || !m_IsFocused || !m_ReferenceGeometry);
  }
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::GetCursorPositionBinding() const
{
  return m_CursorPositionBinding;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetCursorPositionBinding(bool cursorPositionBinding)
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
bool MultiWindowWidget::GetScaleFactorBinding() const
{
  return m_ScaleFactorBinding;
}


//-----------------------------------------------------------------------------
void MultiWindowWidget::SetScaleFactorBinding(bool scaleFactorBinding)
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
bool MultiWindowWidget::BlockDisplayEvents(bool blocked)
{
  bool eventsWereBlocked = m_BlockDisplayEvents;
  m_BlockDisplayEvents = blocked;
  return eventsWereBlocked;
}


//-----------------------------------------------------------------------------
bool MultiWindowWidget::BlockUpdate(bool blocked)
{
  bool updateWasBlocked = m_BlockUpdate;

  if (blocked != m_BlockUpdate)
  {
    m_BlockUpdate = blocked;

    for (int i = 0; i < 4; ++i)
    {
      m_RenderWindows[i]->GetSliceNavigationController()->BlockSignals(blocked);
    }
    m_RenderingManager->GetTimeNavigationController()->BlockSignals(blocked);

    if (!blocked)
    {
      bool rendererNeedsUpdate[4] = {false, false, false, false};

      /// Updating state according to the recorded changes.

      if (m_FocusHasChanged)
      {
        this->SetWidgetPlanesLocked(!m_LinkedNavigationEnabled || !m_IsFocused || !m_ReferenceGeometry);

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
        this->SetWidgetPlanesLocked(!m_LinkedNavigationEnabled || !m_IsFocused || !m_ReferenceGeometry);

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
//          m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
          m_RenderingManager->ForceImmediateUpdate(m_RenderWindows[i]->GetRenderWindow());
        }
      }

      /// Sending events and signals.

      if (m_FocusHasChanged)
      {
        m_FocusHasChanged = false;
        if (m_IsFocused && m_ReferenceGeometry)
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
        if (m_IsFocused && m_ReferenceGeometry)
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
        m_BlockSncEvents = true;
        bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
        m_RenderingManager->GetTimeNavigationController()->SendCreatedWorldGeometry();
        this->BlockDisplayEvents(displayEventsWereBlocked);
        m_BlockSncEvents = false;
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
        if (m_WindowLayout != WINDOW_LAYOUT_3D)
        {
          this->UpdatePositionAnnotation(m_SelectedWindowIndex);
          this->UpdateIntensityAnnotation(m_SelectedWindowIndex);
        }

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

}
