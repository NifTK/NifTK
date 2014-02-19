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
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <QmitkRenderWindow.h>
#include <QGridLayout>
#include <mitkMIDASOrientationUtils.h>

#include <usGetModuleContext.h>
#include <usModuleRegistry.h>
#include <mitkGlobalInteraction.h>
#include <mitkVtkLayerController.h>

#include "vtkSideAnnotation_p.h"

/**
 * This class is to notify the SingleViewerWidget about the display geometry changes of a render window.
 */
class DisplayGeometryModificationCommand : public itk::Command
{
public:
  mitkNewMacro2Param(DisplayGeometryModificationCommand, niftkMultiWindowWidget*, MIDASOrientation);


  //-----------------------------------------------------------------------------
  DisplayGeometryModificationCommand(niftkMultiWindowWidget* multiWindowWidget, MIDASOrientation orientation)
  : itk::Command()
  , m_MultiWindowWidget(multiWindowWidget)
  , m_Orientation(orientation)
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
    m_MultiWindowWidget->OnDisplayGeometryModified(m_Orientation);
    return;
  }

private:
  niftkMultiWindowWidget* const m_MultiWindowWidget;
  MIDASOrientation m_Orientation;
};


//-----------------------------------------------------------------------------
niftkMultiWindowWidget::niftkMultiWindowWidget(
    QWidget* parent,
    Qt::WindowFlags flags,
    mitk::RenderingManager* renderingManager,
    mitk::DataStorage* dataStorage
    )
: QmitkStdMultiWidget(parent, flags, renderingManager)
, m_RenderWindows(4)
, m_GridLayout(NULL)
, m_AxialSliceTag(0ul)
, m_SagittalSliceTag(0ul)
, m_CoronalSliceTag(0ul)
, m_IsSelected(false)
, m_IsEnabled(false)
, m_SelectedRenderWindow(0)
, m_CursorVisibility(true)
, m_CursorGlobalVisibility(false)
, m_Show3DWindowIn2x2WindowLayout(false)
, m_Orientation(MIDAS_ORIENTATION_UNKNOWN)
, m_WindowLayout(WINDOW_LAYOUT_ORTHO)
, m_CursorPositions(3)
, m_ScaleFactors(3)
, m_Magnifications(3)
, m_Origins(3)
, m_RenderWindowSizes(3)
, m_Geometry(NULL)
, m_TimeGeometry(NULL)
, m_BlockDisplayEvents(false)
, m_BlockUpdate(false)
, m_SelectedRenderWindowHasChanged(false)
, m_SncGeometryHasChanged(3)
, m_SncTimeHasChanged(3)
, m_SncSliceHasChanged(3)
, m_SelectedPositionHasChanged(false)
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
  this->SetCursorGloballyVisible(false);
  this->SetCursorVisible(false);
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
  this->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  // Default to unselected, so borders are off.
  this->SetSelected(false);

  // Register to listen to SliceNavigators, slice changed events.
  itk::ReceptorMemberCommand<niftkMultiWindowWidget>::Pointer onAxialSliceChangedCommand =
    itk::ReceptorMemberCommand<niftkMultiWindowWidget>::New();
  onAxialSliceChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnAxialSliceChanged);
  m_AxialSliceTag = mitkWidget1->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onAxialSliceChangedCommand);

  itk::ReceptorMemberCommand<niftkMultiWindowWidget>::Pointer onSagittalSliceChangedCommand =
    itk::ReceptorMemberCommand<niftkMultiWindowWidget>::New();
  onSagittalSliceChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnSagittalSliceChanged);
  m_SagittalSliceTag = mitkWidget2->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSagittalSliceChangedCommand);

  itk::ReceptorMemberCommand<niftkMultiWindowWidget>::Pointer onCoronalSliceChangedCommand =
    itk::ReceptorMemberCommand<niftkMultiWindowWidget>::New();
  onCoronalSliceChangedCommand->SetCallbackFunction(this, &niftkMultiWindowWidget::OnCoronalSliceChanged);
  m_CoronalSliceTag = mitkWidget3->GetSliceNavigationController()->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onCoronalSliceChangedCommand);

  // The world position is unknown until the geometry is set. These values are invalid,
  // but still better then having undefined values.
  m_SelectedPosition.Fill(0.0);

  // The cursor is at the middle of the display at the beginning.
  m_CursorPositions[MIDAS_ORIENTATION_AXIAL].Fill(0.5);
  m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL].Fill(0.5);
  m_CursorPositions[MIDAS_ORIENTATION_CORONAL].Fill(0.5);

  m_ScaleFactors[MIDAS_ORIENTATION_AXIAL] = 1.0;
  m_ScaleFactors[MIDAS_ORIENTATION_SAGITTAL] = 1.0;
  m_ScaleFactors[MIDAS_ORIENTATION_CORONAL] = 1.0;

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
    this->AddDisplayGeometryModificationObserver(MIDASOrientation(i));
  }

  // The mouse mode switcher is declared and initialised in QmitkStdMultiWidget. It creates an
  // mitk::DisplayInteractor. This line decreases the reference counter of the mouse mode switcher
  // so that it is destructed and it unregisters and destructs its display interactor as well.
  m_MouseModeSwitcher = 0;
}


//-----------------------------------------------------------------------------
niftkMultiWindowWidget::~niftkMultiWindowWidget()
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
  for (int i = 0; i < 3; ++i)
  {
    this->RemoveDisplayGeometryModificationObserver(MIDASOrientation(i));
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
void niftkMultiWindowWidget::AddDisplayGeometryModificationObserver(MIDASOrientation orientation)
{
  mitk::BaseRenderer* renderer = m_RenderWindows[orientation]->GetRenderer();
  assert(renderer);

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  assert(displayGeometry);

  DisplayGeometryModificationCommand::Pointer command = DisplayGeometryModificationCommand::New(this, orientation);
  m_RenderWindowSizes[orientation] = displayGeometry->GetSizeInDisplayUnits();
  m_Origins[orientation] = displayGeometry->GetOriginInDisplayUnits();
  m_ScaleFactors[orientation] = displayGeometry->GetScaleFactorMMPerDisplayUnit();
  m_Magnifications[orientation] = this->GetMagnification(orientation);
  unsigned long observerTag = displayGeometry->AddObserver(itk::ModifiedEvent(), command);
  m_DisplayGeometryModificationObservers[orientation] = observerTag;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::RemoveDisplayGeometryModificationObserver(MIDASOrientation orientation)
{
  mitk::BaseRenderer* renderer = m_RenderWindows[orientation]->GetRenderer();
  assert(renderer);

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  assert(displayGeometry);

  displayGeometry->RemoveObserver(m_DisplayGeometryModificationObservers[orientation]);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnAxialSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  if (!m_BlockProcessingSncSignals)
  {
    this->OnSelectedPositionChanged(MIDAS_ORIENTATION_AXIAL);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnSagittalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  if (!m_BlockProcessingSncSignals)
  {
    this->OnSelectedPositionChanged(MIDAS_ORIENTATION_SAGITTAL);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnCoronalSliceChanged(const itk::EventObject& /*geometrySliceEvent*/)
{
  if (!m_BlockProcessingSncSignals)
  {
    this->OnSelectedPositionChanged(MIDAS_ORIENTATION_CORONAL);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetBackgroundColor(QColor colour)
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
  this->RequestUpdate();
}


//-----------------------------------------------------------------------------
QColor niftkMultiWindowWidget::GetBackgroundColor() const
{
  return m_BackgroundColor;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetSelected(bool selected)
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
bool niftkMultiWindowWidget::IsSelected() const
{
  return m_IsSelected;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkMultiWindowWidget::GetSelectedRenderWindow() const
{
  return m_SelectedRenderWindow;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetSelectedRenderWindow(QmitkRenderWindow* renderWindow)
{
  // When we "Select", the selection is at the level of the niftkMultiWindowWidget
  // so the whole of this widget is selected. However, we may have clicked in
  // a specific view, so it still helps to highlight the most recently clicked on view.
  // Also, if you are displaying ortho window layout (2x2) then you actually have 4 windows present,
  // then highlighting them all starts to look a bit confusing, so we just highlight the
  // most recently focused window, (eg. axial, sagittal, coronal or 3D).

  if (renderWindow != m_SelectedRenderWindow)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    if (renderWindow == this->GetRenderWindow1())
    {
      m_Orientation = MIDAS_ORIENTATION_AXIAL;
      m_IsSelected = true;
      m_SelectedRenderWindow = renderWindow;
      m_SelectedRenderWindowHasChanged = true;

      m_RectangleRendering1->Enable(1.0, 0.0, 0.0);
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (renderWindow == this->GetRenderWindow2())
    {
      m_Orientation = MIDAS_ORIENTATION_SAGITTAL;
      m_IsSelected = true;
      m_SelectedRenderWindow = renderWindow;
      m_SelectedRenderWindowHasChanged = true;

      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Enable(0.0, 1.0, 0.0);
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Disable();
    }
    else if (renderWindow == this->GetRenderWindow3())
    {
      m_Orientation = MIDAS_ORIENTATION_CORONAL;
      m_IsSelected = true;
      m_SelectedRenderWindow = renderWindow;
      m_SelectedRenderWindowHasChanged = true;

      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Enable(0.0, 0.0, 1.0);
      m_RectangleRendering4->Disable();
    }
    else if (renderWindow == this->GetRenderWindow4())
    {
      m_Orientation = MIDAS_ORIENTATION_UNKNOWN;
      m_IsSelected = true;
      m_SelectedRenderWindow = renderWindow;
      m_SelectedRenderWindowHasChanged = true;

      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering4->Enable(1.0, 1.0, 0.0);
    }
    else
    {
      m_Orientation = MIDAS_ORIENTATION_UNKNOWN;
      m_IsSelected = false;
      if (m_SelectedRenderWindow != 0)
      {
        m_SelectedRenderWindow = 0;
        m_SelectedRenderWindowHasChanged = true;
      }

      m_RectangleRendering1->Disable();
      m_RectangleRendering2->Disable();
      m_RectangleRendering3->Disable();
      m_RectangleRendering3->Disable();
    }

    this->BlockUpdate(updateWasBlocked);
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
void niftkMultiWindowWidget::SetEnabled(bool b)
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
bool niftkMultiWindowWidget::IsEnabled() const
{
  return m_IsEnabled;
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
bool niftkMultiWindowWidget::IsCursorVisible() const
{
  return m_CursorVisibility;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetCursorGloballyVisible(bool visible)
{
  // Here, "globally" means the plane nodes created within this widget will be available in ALL other render windows.
  m_CursorGlobalVisibility = visible;
  m_PlaneNode1->SetVisibility(visible);
  m_PlaneNode1->Modified();
  m_PlaneNode2->SetVisibility(visible);
  m_PlaneNode2->Modified();
  m_PlaneNode3->SetVisibility(visible);
  m_PlaneNode3->Modified();
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::IsCursorGloballyVisible() const
{
  return m_CursorGlobalVisibility;
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::AreDirectionAnnotationsVisible() const
{
  return m_DirectionAnnotations[0]->GetVisibility()
      && m_DirectionAnnotations[1]->GetVisibility()
      && m_DirectionAnnotations[2]->GetVisibility();
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetDirectionAnnotationsVisible(bool visible)
{
  for (int i = 0; i < 3; ++i)
  {
    m_DirectionAnnotations[i]->SetVisibility(visible);
  }
  this->RequestUpdate();
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetShow3DWindowIn2x2WindowLayout(bool visible)
{
  m_Show3DWindowIn2x2WindowLayout = visible;
  this->Update3DWindowVisibility();
}


//-----------------------------------------------------------------------------
bool niftkMultiWindowWidget::GetShow3DWindowIn2x2WindowLayout() const
{
  return m_Show3DWindowIn2x2WindowLayout;
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
void niftkMultiWindowWidget::SetVisibility(QmitkRenderWindow* renderWindow, mitk::DataNode* node, bool visible)
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
void niftkMultiWindowWidget::SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible)
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
MIDASOrientation niftkMultiWindowWidget::GetOrientation() const
{
  return m_Orientation;
//  MIDASOrientation result = MIDAS_ORIENTATION_UNKNOWN;
//  if (m_WindowLayout == WINDOW_LAYOUT_AXIAL)
//  {
//    result = MIDAS_ORIENTATION_AXIAL;
//  }
//  else if (m_WindowLayout == WINDOW_LAYOUT_SAGITTAL)
//  {
//    result = MIDAS_ORIENTATION_SAGITTAL;
//  }
//  else if (m_WindowLayout == WINDOW_LAYOUT_CORONAL)
//  {
//    result = MIDAS_ORIENTATION_CORONAL;
//  }
//  else if (m_WindowLayout == WINDOW_LAYOUT_ORTHO
//           || m_WindowLayout == WINDOW_LAYOUT_3H
//           || m_WindowLayout == WINDOW_LAYOUT_3V
//           || m_WindowLayout == WINDOW_LAYOUT_COR_SAG_H
//           || m_WindowLayout == WINDOW_LAYOUT_COR_SAG_V
//           || m_WindowLayout == WINDOW_LAYOUT_COR_AX_H
//           || m_WindowLayout == WINDOW_LAYOUT_COR_AX_V
//           || m_WindowLayout == WINDOW_LAYOUT_SAG_AX_H
//           || m_WindowLayout == WINDOW_LAYOUT_SAG_AX_V
//           )
//  {
//    if (m_RectangleRendering1->IsEnabled())
//    {
//      result = MIDAS_ORIENTATION_AXIAL;
//    }
//    else if (m_RectangleRendering2->IsEnabled())
//    {
//      result = MIDAS_ORIENTATION_SAGITTAL;
//    }
//    else if (m_RectangleRendering3->IsEnabled())
//    {
//      result = MIDAS_ORIENTATION_CORONAL;
//    }
//  }
//  return result;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::FitToDisplay()
{
  bool updateWasBlocked = this->BlockUpdate(true);

  double largestScaleFactor = -1.0;
  MIDASOrientation orientationWithLargestScaleFactor = MIDAS_ORIENTATION_UNKNOWN;

  for (int i = 0; i < 3; ++i)
  {
    MIDASOrientation orientation = MIDASOrientation(i);
    if (m_RenderWindows[orientation]->isVisible())
    {
      mitk::DisplayGeometry* displayGeometry = m_RenderWindows[orientation]->GetRenderer()->GetDisplayGeometry();

      if (!m_ScaleFactorBinding)
      {
        /// If the scale factors are not bound, we simply fit each 'world' into their window.
        bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
        displayGeometry->Fit();
        this->BlockDisplayEvents(displayEventsWereBlocked);
        m_ScaleFactors[i] = displayGeometry->GetScaleFactorMMPerDisplayUnit();
        m_ScaleFactorHasChanged[i] = true;
        m_Magnifications[i] = this->GetMagnification(MIDASOrientation(i));
        this->UpdateCursorPosition(orientation);
      }
      else
      {
        /// If the scale factors are bound, first we find the window that requires the largest scaling...
        double widthMmPerPx = displayGeometry->GetWorldGeometry()->GetParametricExtentInMM(0) / displayGeometry->GetDisplayWidth();
        double heightMmPerPx = displayGeometry->GetWorldGeometry()->GetParametricExtentInMM(1) / displayGeometry->GetDisplayHeight();
        if (widthMmPerPx > largestScaleFactor)
        {
          largestScaleFactor = widthMmPerPx;
          orientationWithLargestScaleFactor = orientation;
        }
        if (heightMmPerPx > largestScaleFactor)
        {
          largestScaleFactor = heightMmPerPx;
          orientationWithLargestScaleFactor = orientation;
        }
      }
    }
  }

  if (m_ScaleFactorBinding && orientationWithLargestScaleFactor != MIDAS_ORIENTATION_UNKNOWN)
  {
    /// ... then call mitk::DisplayGeometry::Fit() for that window ...
    mitk::DisplayGeometry* displayGeometry = m_RenderWindows[orientationWithLargestScaleFactor]->GetRenderer()->GetDisplayGeometry();
    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
    displayGeometry->Fit();
    this->BlockDisplayEvents(displayEventsWereBlocked);
    double largestScaleFactor = displayGeometry->GetScaleFactorMMPerDisplayUnit();
    m_ScaleFactors[orientationWithLargestScaleFactor] = largestScaleFactor;
    m_ScaleFactorHasChanged[orientationWithLargestScaleFactor] = true;
    m_Magnifications[orientationWithLargestScaleFactor] = this->GetMagnification(orientationWithLargestScaleFactor);
    this->UpdateCursorPosition(orientationWithLargestScaleFactor);

    /// ... and finally, apply the same scale factor for the other render windows.
    for (int i = 0; i < 3; ++i)
    {
      MIDASOrientation otherOrientation = MIDASOrientation(i);
      if (otherOrientation != orientationWithLargestScaleFactor && m_RenderWindows[otherOrientation]->isVisible())
      {
        mitk::Vector2D cursorPosition;
        cursorPosition.Fill(0.5);
        this->SetCursorPosition(otherOrientation, cursorPosition);
        m_ScaleFactors[otherOrientation] = largestScaleFactor;
        m_ScaleFactorHasChanged[otherOrientation] = true;
        m_Magnifications[otherOrientation] = this->GetMagnification(otherOrientation);
      }
    }
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetGeometry(mitk::TimeGeometry* geometry)
{
  if (geometry != NULL)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);

    m_Geometry = geometry->GetGeometryForTimeStep(0);
    m_TimeGeometry = geometry;

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

    /// The place of the direction annotations on the render window:
    ///
    /// +----0----+
    /// |         |
    /// 3         1
    /// |         |
    /// +----2----+
    m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(0, "A");
    m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(2, "P");
    m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(3, "R");
    m_DirectionAnnotations[MIDAS_ORIENTATION_AXIAL]->SetText(1, "L");

    m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(0, "S");
    m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(2, "I");
    m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(3, "A");
    m_DirectionAnnotations[MIDAS_ORIENTATION_SAGITTAL]->SetText(1, "P");

    m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(0, "S");
    m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(2, "I");
    m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(3, "R");
    m_DirectionAnnotations[MIDAS_ORIENTATION_CORONAL]->SetText(1, "L");

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

    permutedBoundingBox[0] = static_cast<int>(m_Geometry->GetExtent(dominantAxisRL));
    permutedBoundingBox[1] = static_cast<int>(m_Geometry->GetExtent(dominantAxisAP));
    permutedBoundingBox[2] = static_cast<int>(m_Geometry->GetExtent(dominantAxisSI));

    permutedSpacing[0] = m_Geometry->GetSpacing()[permutedAxes[0]];
    permutedSpacing[1] = m_Geometry->GetSpacing()[permutedAxes[1]];
    permutedSpacing[2] = m_Geometry->GetSpacing()[permutedAxes[2]];

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

//    MITK_DEBUG << "Matt, extent=" << geometry->GetExtent(0) << ", " << geometry->GetExtent(1) << ", " << geometry->GetExtent(2) << std::endl;
//    MITK_DEBUG << "Matt, domRL=" << dominantAxisRL << ", signRL=" << signRL << ", domAP=" << dominantAxisAP << ", signAP=" << signAP << ", dominantAxisSI=" << dominantAxisSI << ", signSI=" << signSI << std::endl;
//    MITK_DEBUG << "Matt, permutedBoundingBox=" << permutedBoundingBox[0] << ", " << permutedBoundingBox[1] << ", " << permutedBoundingBox[2] << std::endl;
//    MITK_DEBUG << "Matt, permutedAxes=" << permutedAxes[0] << ", " << permutedAxes[1] << ", " << permutedAxes[2] << std::endl;
//    MITK_DEBUG << "Matt, permutedSpacing=" << permutedSpacing[0] << ", " << permutedSpacing[1] << ", " << permutedSpacing[2] << std::endl;
//    MITK_DEBUG << "Matt, flippedAxes=" << flippedAxes[0] << ", " << flippedAxes[1] << ", " << flippedAxes[2] << std::endl;

//    MITK_DEBUG << "Matt, input normalised matrix=" << std::endl;

//    for (unsigned int i=0; i < 3; i++)
//    {
//      MITK_DEBUG << affineTransformMatrix[i][0] << " " << affineTransformMatrix[i][1] << " " << affineTransformMatrix[i][2];
//    }

//    MITK_DEBUG << "Matt, inverse normalised matrix=" << std::endl;

//    for (unsigned int i=0; i < 3; i++)
//    {
//      MITK_DEBUG << inverseTransformMatrix[i][0] << " " << inverseTransformMatrix[i][1] << " " << inverseTransformMatrix[i][2];
//    }

//    MITK_DEBUG << "Matt, permuted matrix=" << std::endl;

//    for (unsigned int i=0; i < 3; i++)
//    {
//      MITK_DEBUG << permutedMatrix[i][0] << " " << permutedMatrix[i][1] << " " << permutedMatrix[i][2];
//    }

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
        if (m_Geometry->GetImageGeometry())
        {
          voxelOffset = 0.5;
        }

        originInVx[0] = 0;
        originInVx[1] = 0;
        originInVx[2] = 0;

        if (flippedAxes[0] < 0)
        {
//          MITK_DEBUG << "Matt, flippedAxis[0] < 0, so flipping axis " << permutedAxes[0] << std::endl;
          originInVx[permutedAxes[0]] = m_Geometry->GetExtent(permutedAxes[0]) - 1;
        }
        if (flippedAxes[1] < 0)
        {
//          MITK_DEBUG << "Matt, flippedAxis[1] < 0, so flipping axis " << permutedAxes[1] << std::endl;
          originInVx[permutedAxes[1]] = m_Geometry->GetExtent(permutedAxes[1]) - 1;
        }
        if (flippedAxes[2] < 0)
        {
//          MITK_DEBUG << "Matt, flippedAxis[2] < 0, so flipping axis " << permutedAxes[2] << std::endl;
          originInVx[permutedAxes[2]] = m_Geometry->GetExtent(permutedAxes[2]) - 1;
        }

        m_Geometry->IndexToWorld(originInVx, originInMm);

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

//        MITK_DEBUG << "Matt, image=" << geometry->GetImageGeometry() << std::endl;
//        MITK_DEBUG << "Matt, width=" << width << std::endl;
//        MITK_DEBUG << "Matt, height=" << height << std::endl;
//        MITK_DEBUG << "Matt, originOfSlice=" << originOfSlice << std::endl;
//        MITK_DEBUG << "Matt, rightDV=" << rightDV << std::endl;
//        MITK_DEBUG << "Matt, bottomDV=" << bottomDV << std::endl;
//        MITK_DEBUG << "Matt, normal=" << normal << std::endl;
//        MITK_DEBUG << "Matt, viewSpacing=" << viewSpacing << std::endl;
//        MITK_DEBUG << "Matt, slices=" << slices << std::endl;
//        MITK_DEBUG << "Matt, isFlipped=" << isFlipped << std::endl;

        // TODO Commented out when migrating to the redesigned MITK geometry framework.
        // This will definitely not work. Should be fixed.

        mitk::TimeStepType numberOfTimeSteps = geometry->CountTimeSteps();

        mitk::ProportionalTimeGeometry::Pointer createdTimeGeometry = mitk::ProportionalTimeGeometry::New();
        createdTimeGeometry->Initialize();
        createdTimeGeometry->Expand(numberOfTimeSteps);
//        createdTimeSlicedGeometry->SetImageGeometry(false);
//        createdTimeSlicedGeometry->SetEvenlyTimed(true);

//        if (inputTimeSlicedGeometry.IsNotNull())
//        {
//          createdTimeSlicedGeometry->SetEvenlyTimed(inputTimeSlicedGeometry->GetEvenlyTimed());
//          createdTimeSlicedGeometry->SetTimeBounds(inputTimeSlicedGeometry->GetTimeBounds());
//          createdTimeSlicedGeometry->SetBounds(inputTimeSlicedGeometry->GetBounds());
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

          slicedGeometry->SetTimeBounds(geometry->GetGeometryForTimeStep(timeStep)->GetTimeBounds());
          createdTimeGeometry->SetTimeStepGeometry(slicedGeometry, timeStep);
        }
        createdTimeGeometry->Update();

//        MITK_DEBUG << "Matt - final geometry=" << createdTimeSlicedGeometry << std::endl;
//        MITK_DEBUG << "Matt - final geometry origin=" << createdTimeSlicedGeometry->GetOrigin() << std::endl;
//        MITK_DEBUG << "Matt - final geometry center=" << createdTimeSlicedGeometry->GetCenter() << std::endl;
//        for (int j = 0; j < 8; j++)
//        {
//          MITK_DEBUG << "Matt - final geometry j=" << j << ", p=" << createdTimeSlicedGeometry->GetCornerPoint(j) << std::endl;
//        }

        sliceNavigationController->SetInputWorldTimeGeometry(createdTimeGeometry);
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
      }
    }

    this->BlockDisplayEvents(displayEventsWereBlocked);

    for (int i = 0; i < 3; ++i)
    {
      m_SncGeometryHasChanged[i] = true;
      m_SncTimeHasChanged[i] = true;
      m_SncSliceHasChanged[i] = true;
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
void niftkMultiWindowWidget::SetWindowLayout(WindowLayout windowLayout)
{
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
  else // if (windowLayout == WINDOW_LAYOUT_ORTHO)
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

  switch (windowLayout)
  {
  case WINDOW_LAYOUT_AXIAL:
    showAxial = true;
    break;
  case WINDOW_LAYOUT_SAGITTAL:
    showSagittal = true;
    break;
  case WINDOW_LAYOUT_CORONAL:
    showCoronal = true;
    break;
  case WINDOW_LAYOUT_ORTHO:
    showAxial = true;
    showSagittal = true;
    showCoronal = true;
    show3D = true;
    m_CursorAxialPositionsAreBound = true;
    m_CursorSagittalPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_3H:
    showAxial = true;
    showSagittal = true;
    showCoronal = true;
    m_CursorAxialPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_3V:
    showAxial = true;
    showSagittal = true;
    showCoronal = true;
    m_CursorSagittalPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_3D:
    show3D = true;
    break;
  case WINDOW_LAYOUT_COR_SAG_H:
    showSagittal = true;
    showCoronal = true;
    m_CursorAxialPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_COR_SAG_V:
    showSagittal = true;
    showCoronal = true;
    break;
  case WINDOW_LAYOUT_COR_AX_H:
    showAxial = true;
    showCoronal = true;
    break;
  case WINDOW_LAYOUT_COR_AX_V:
    showAxial = true;
    showCoronal = true;
    m_CursorSagittalPositionsAreBound = true;
    break;
  case WINDOW_LAYOUT_SAG_AX_H:
    showAxial = true;
    showSagittal = true;
    break;
  case WINDOW_LAYOUT_SAG_AX_V:
    showAxial = true;
    showSagittal = true;
    break;
  default:
    // die, this should never happen
    assert((m_WindowLayout >= 0 && m_WindowLayout <= 6) || (m_WindowLayout >= 9 && m_WindowLayout <= 14));
    break;
  }

  if (!showAxial)
  {
    this->mitkWidget1->GetRenderWindow()->SetSize(0, 0);
  }
  if (!showSagittal)
  {
    this->mitkWidget2->GetRenderWindow()->SetSize(0, 0);
  }
  if (!showCoronal)
  {
    this->mitkWidget3->GetRenderWindow()->SetSize(0, 0);
  }
  if (!show3D)
  {
    this->mitkWidget4->GetRenderWindow()->SetSize(0, 0);
  }
  this->mitkWidget1Container->setVisible(showAxial);
  this->mitkWidget2Container->setVisible(showSagittal);
  this->mitkWidget3Container->setVisible(showCoronal);
  this->mitkWidget4Container->setVisible(show3D);

  m_CursorPositionBinding = ::IsMultiWindowLayout(windowLayout);
  m_ScaleFactorBinding = ::IsMultiWindowLayout(windowLayout);

  m_WindowLayout = windowLayout;

  this->Update3DWindowVisibility();
  m_GridLayout->activate();

  for (size_t i = 0; i < 3; ++i)
  {
    mitk::DisplayGeometry* displayGeometry = m_RenderWindows[i]->GetRenderer()->GetDisplayGeometry();
    m_RenderWindowSizes[i] = displayGeometry->GetSizeInDisplayUnits();
  }

  this->BlockDisplayEvents(displayEventsWereBlocked);
}


//-----------------------------------------------------------------------------
WindowLayout niftkMultiWindowWidget::GetWindowLayout() const
{
  return m_WindowLayout;
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetMaxSlice(MIDASOrientation orientation) const
{
  assert(orientation != MIDAS_ORIENTATION_UNKNOWN);

  int maxSlice = 0;

  mitk::SliceNavigationController* snc = m_RenderWindows[orientation]->GetSliceNavigationController();

  if (snc->GetSlice() != NULL && snc->GetSlice()->GetSteps() > 0)
  {
    maxSlice = snc->GetSlice()->GetSteps() - 1;
  }

  return maxSlice;
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetMaxTimeStep() const
{
  int maxTimeStep = 0;

  mitk::SliceNavigationController* snc = m_RenderWindows[MIDAS_ORIENTATION_AXIAL]->GetSliceNavigationController();
  assert(snc);

  if (snc->GetTime() != NULL && snc->GetTime()->GetSteps() >= 1)
  {
    maxTimeStep = snc->GetTime()->GetSteps() -1;
  }

  return maxTimeStep;
}


//-----------------------------------------------------------------------------
const mitk::Vector2D& niftkMultiWindowWidget::GetCursorPosition(MIDASOrientation orientation) const
{
  return m_CursorPositions[orientation];
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetCursorPosition(MIDASOrientation orientation, const mitk::Vector2D& cursorPosition)
{
  assert(orientation >= 0 && orientation < 3);

  bool updateWasBlocked = this->BlockUpdate(true);

  if (cursorPosition != m_CursorPositions[orientation])
  {
    m_CursorPositions[orientation] = cursorPosition;
    m_CursorPositionHasChanged[orientation] = true;
  }

  if (m_CursorPositionBinding)
  {
    if (orientation == MIDAS_ORIENTATION_AXIAL)
    {
      if (m_CursorSagittalPositionsAreBound)
      {
        m_CursorPositions[MIDAS_ORIENTATION_CORONAL][0] = m_CursorPositions[MIDAS_ORIENTATION_AXIAL][0];
        m_CursorPositionHasChanged[MIDAS_ORIENTATION_CORONAL] = true;
      }

      if (m_CursorCoronalPositionsAreBound)
      {
        m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 1.0 - m_CursorPositions[MIDAS_ORIENTATION_AXIAL][1];
        m_CursorPositionHasChanged[MIDAS_ORIENTATION_SAGITTAL] = true;
      }
    }
    else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      if (m_CursorCoronalPositionsAreBound)
      {
        m_CursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 1.0 - m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][0];
        m_CursorPositionHasChanged[MIDAS_ORIENTATION_AXIAL] = true;
      }

      if (m_CursorAxialPositionsAreBound)
      {
        m_CursorPositions[MIDAS_ORIENTATION_CORONAL][1] = m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][1];
        m_CursorPositionHasChanged[MIDAS_ORIENTATION_CORONAL] = true;
      }
    }
    else if (orientation == MIDAS_ORIENTATION_CORONAL)
    {
      if (m_CursorSagittalPositionsAreBound)
      {
        m_CursorPositions[MIDAS_ORIENTATION_AXIAL][0] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][0];
        m_CursorPositionHasChanged[MIDAS_ORIENTATION_AXIAL] = true;
      }

      if (m_CursorAxialPositionsAreBound)
      {
        m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][1];
        m_CursorPositionHasChanged[MIDAS_ORIENTATION_SAGITTAL] = true;
      }
    }
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::MoveToCursorPosition(MIDASOrientation orientation)
{
  if (m_Geometry)
  {
    QmitkRenderWindow* renderWindow = m_RenderWindows[orientation];
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
      positionInPx[0] = m_CursorPositions[orientation][0] * displaySize[0];
      positionInPx[1] = m_CursorPositions[orientation][1] * displaySize[1];

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
void niftkMultiWindowWidget::OnDisplayGeometryModified(MIDASOrientation orientation)
{
  if (m_BlockDisplayEvents)
  {
    return;
  }

  bool updateWasBlocked = this->BlockUpdate(true);

  mitk::DisplayGeometry* displayGeometry = m_RenderWindows[orientation]->GetRenderer()->GetDisplayGeometry();
  mitk::Vector2D renderWindowSize = displayGeometry->GetSizeInDisplayUnits();
  if (renderWindowSize != m_RenderWindowSizes[orientation])
  {
    double horizontalSizeChange = m_RenderWindowSizes[orientation][0] / renderWindowSize[0];
    double verticalSizeChange = m_RenderWindowSizes[orientation][1] / renderWindowSize[1];
    double horizontalScaleFactor = m_ScaleFactors[orientation] * horizontalSizeChange;
    double verticalScaleFactor = m_ScaleFactors[orientation] * verticalSizeChange;

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

    m_Origins[orientation] = displayGeometry->GetOriginInDisplayUnits();
    m_CursorPositionHasChanged[orientation] = true;
    m_ScaleFactors[orientation] = scaleFactor;
    m_ScaleFactorHasChanged[orientation] = true;
    m_Magnifications[orientation] = this->GetMagnification(orientation);
    m_RenderWindowSizes[orientation] = renderWindowSize;

    this->BlockUpdate(updateWasBlocked);
    return;
  }

  // Note that the scaling changes the scale factor *and* the origin,
  // while the moving changes the origin only.

  bool beingPanned = true;

  double scaleFactor = displayGeometry->GetScaleFactorMMPerDisplayUnit();
  if (scaleFactor != m_ScaleFactors[orientation])
  {
    beingPanned = false;

    mitk::Vector2D origin = displayGeometry->GetOriginInDisplayUnits();
    mitk::Vector2D focusPoint = (m_Origins[orientation] * m_ScaleFactors[orientation] - origin * scaleFactor) / (scaleFactor - m_ScaleFactors[orientation]);
    mitk::Vector2D cursorPosition = focusPoint;
    cursorPosition[0] /= m_RenderWindowSizes[orientation][0];
    cursorPosition[1] /= m_RenderWindowSizes[orientation][1];

    if (cursorPosition != m_CursorPositions[orientation])
    {
      if (m_Geometry)
      {
        mitk::Point2D focusPointInPx;
        focusPointInPx[0] = focusPoint[0];
        focusPointInPx[1] = focusPoint[1];
        mitk::Point2D focusPointInMm;
        displayGeometry->DisplayToWorld(focusPointInPx, focusPointInMm);
        mitk::Point3D focusPoint3DInMm;
        displayGeometry->Map(focusPointInMm, focusPoint3DInMm);
        this->SetSelectedPosition(focusPoint3DInMm);
      }
    }
    this->OnScaleFactorChanged(orientation, scaleFactor);
  }

  mitk::Vector2D origin = displayGeometry->GetOriginInDisplayUnits();
  if (origin != m_Origins[orientation])
  {
    if (beingPanned)
    {
      this->OnOriginChanged(m_Orientation, beingPanned);
    }
    m_Origins[orientation] = origin;
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnOriginChanged(MIDASOrientation orientation, bool beingPanned)
{
  if (m_Geometry)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    this->UpdateCursorPosition(orientation);

    if (beingPanned && this->GetCursorPositionBinding())
    {
      // sagittal[1] <-> coronal[1]      (if m_CursorAxialPositionsAreBound is set)
      // axial[0] <-> coronal[0]         (if m_CursorSagittalPositionsAreBound is set)
      // axial[1] <-> 1.0 - sagittal[0]  (if m_CursorCoronalPositionsAreBound is set)

      if (orientation == MIDAS_ORIENTATION_AXIAL)
      {
        if (m_CursorSagittalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_CORONAL][0] = m_CursorPositions[MIDAS_ORIENTATION_AXIAL][0];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_CORONAL] = true;
        }

        if (m_CursorCoronalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 1.0 - m_CursorPositions[MIDAS_ORIENTATION_AXIAL][1];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_SAGITTAL] = true;
        }
      }
      else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        if (m_CursorCoronalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 1.0 - m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][0];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_AXIAL] = true;
        }

        if (m_CursorAxialPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_CORONAL][1] = m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][1];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_CORONAL] = true;
        }
      }
      else if (orientation == MIDAS_ORIENTATION_CORONAL)
      {
        if (m_CursorSagittalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_AXIAL][0] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][0];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_AXIAL] = true;
        }

        if (m_CursorAxialPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][1];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_SAGITTAL] = true;
        }
      }
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnZoomFocusChanged(MIDASOrientation orientation, const mitk::Vector2D& focusPoint)
{
  if (m_Geometry)
  {
    mitk::DisplayGeometry* displayGeometry = m_RenderWindows[orientation]->GetRenderer()->GetDisplayGeometry();
    mitk::Point2D focusPointInPx;
    focusPointInPx[0] = focusPoint[0];
    focusPointInPx[1] = focusPoint[1];
    mitk::Point2D focusPointInMm;
    displayGeometry->DisplayToWorld(focusPointInPx, focusPointInMm);
    mitk::Point3D focusPoint3DInMm;
    displayGeometry->Map(focusPointInMm, focusPoint3DInMm);
    this->SetSelectedPosition(focusPoint3DInMm);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnScaleFactorChanged(MIDASOrientation orientation, double scaleFactor)
{
  if (m_Geometry)
  {
    if (scaleFactor != m_ScaleFactors[orientation])
    {
      bool updateWasBlocked = this->BlockUpdate(true);
      m_ScaleFactors[orientation] = scaleFactor;
      m_ScaleFactorHasChanged[orientation] = true;
      m_Magnifications[orientation] = this->GetMagnification(orientation);

      if (this->GetScaleFactorBinding())
      {
        // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
        for (int i = 0; i < 3; ++i)
        {
          if (i != orientation && m_RenderWindows[i]->isVisible())
          {
            this->SetScaleFactor(MIDASOrientation(i), scaleFactor);
          }
        }
      }
      this->BlockUpdate(updateWasBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::OnSelectedPositionChanged(MIDASOrientation orientation)
{
  if (m_Geometry != NULL && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    m_SelectedPosition = this->GetCrossPosition();
    m_SelectedPositionHasChanged = true;

    /// Note that this is the orientation of the selected render window, in which the
    /// the mouse click or scroll changed the selected position.
    /// In contrast, the 'orientation' argument of this function contains the orientation along
    /// which the selected position has changed. If the event was a scroll event then
    /// the two orientations are equal. If it was a mouse click event then they will
    /// be different.
    MIDASOrientation selectedWindowOrientation = this->GetOrientation();

    if (selectedWindowOrientation == orientation)
    {
      /// If the position has changed by scrolling through the slices in one window
      /// then the image stays in place in each window, but the cursor positions change
      /// in the other two windows, therefore we have to update them.
      if (selectedWindowOrientation == MIDAS_ORIENTATION_AXIAL)
      {
        this->UpdateCursorPosition(MIDAS_ORIENTATION_CORONAL);

        if (m_CursorPositionBinding && m_CursorAxialPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][1];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_SAGITTAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
        }
      }
      else if (selectedWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        this->UpdateCursorPosition(MIDAS_ORIENTATION_CORONAL);

        if (m_CursorPositionBinding && m_CursorSagittalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_AXIAL][0] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][0];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_AXIAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_AXIAL);
        }
      }
      else if (selectedWindowOrientation == MIDAS_ORIENTATION_CORONAL)
      {
        this->UpdateCursorPosition(MIDAS_ORIENTATION_AXIAL);
        this->UpdateCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
      }
    }
    else if (selectedWindowOrientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      /// If the selected position has changed by clicking into the render window.

      this->UpdateCursorPosition(selectedWindowOrientation);

      /// sagittal[1] <-> coronal[1]      (if m_CursorAxialPositionsAreBound is set)
      /// axial[0] <-> coronal[0]         (if m_CursorSagittalPositionsAreBound is set)
      /// axial[1] <-> 1.0 - sagittal[0]  (if m_CursorCoronalPositionsAreBound is set)

      if (selectedWindowOrientation == MIDAS_ORIENTATION_AXIAL && orientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        if (m_CursorPositionBinding && m_CursorSagittalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_CORONAL][0] = m_CursorPositions[MIDAS_ORIENTATION_AXIAL][0];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_CORONAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_CORONAL);
        }
      }
      else if (selectedWindowOrientation == MIDAS_ORIENTATION_AXIAL && orientation == MIDAS_ORIENTATION_CORONAL)
      {
        if (m_CursorPositionBinding && m_CursorCoronalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 1.0 - m_CursorPositions[MIDAS_ORIENTATION_AXIAL][1];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_SAGITTAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
        }
      }
      else if (selectedWindowOrientation == MIDAS_ORIENTATION_SAGITTAL && orientation == MIDAS_ORIENTATION_AXIAL)
      {
        if (m_CursorPositionBinding && m_CursorAxialPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_CORONAL][1] = m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][1];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_CORONAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_CORONAL);
        }
      }
      else if (selectedWindowOrientation == MIDAS_ORIENTATION_SAGITTAL && orientation == MIDAS_ORIENTATION_CORONAL)
      {
        if (m_CursorPositionBinding && m_CursorCoronalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 1.0 - m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][0];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_AXIAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_AXIAL);
        }
      }
      else if (selectedWindowOrientation == MIDAS_ORIENTATION_CORONAL && orientation == MIDAS_ORIENTATION_AXIAL)
      {
        if (m_CursorPositionBinding && m_CursorAxialPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][1];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_SAGITTAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
        }
      }
      else if (selectedWindowOrientation == MIDAS_ORIENTATION_CORONAL && orientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        if (m_CursorPositionBinding && m_CursorSagittalPositionsAreBound)
        {
          m_CursorPositions[MIDAS_ORIENTATION_AXIAL][0] = m_CursorPositions[MIDAS_ORIENTATION_CORONAL][0];
          m_CursorPositionHasChanged[MIDAS_ORIENTATION_AXIAL] = true;
        }
        else
        {
          this->UpdateCursorPosition(MIDAS_ORIENTATION_AXIAL);
        }
      }
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetSelectedSlice(MIDASOrientation orientation) const
{
  int selectedSlice = 0;

  if (m_Geometry != NULL)
  {
    mitk::Index3D selectedPositionInVx;
    m_Geometry->WorldToIndex(m_SelectedPosition, selectedPositionInVx);

    int axis = m_OrientationToAxisMap[orientation];
    selectedSlice = selectedPositionInVx[axis];
  }

  return selectedSlice;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetSelectedSlice(MIDASOrientation orientation, int selectedSlice)
{
  const mitk::Geometry3D* geometry = m_Geometry;
  if (geometry != NULL)
  {
    mitk::Index3D selectedPositionInVx;
    mitk::Point3D selectedPosition = m_SelectedPosition;

    geometry->WorldToIndex(selectedPosition, selectedPositionInVx);

    int axis = m_OrientationToAxisMap[orientation];
    selectedPositionInVx[axis] = selectedSlice;

    mitk::Point3D tmp;
    tmp[0] = selectedPositionInVx[0];
    tmp[1] = selectedPositionInVx[1];
    tmp[2] = selectedPositionInVx[2];

    geometry->IndexToWorld(tmp, selectedPosition);

    // Does not work, as it relies on the StateMachine event broadcasting mechanism,
    // and if the widget is not listening, then it goes unnoticed.
    //this->MoveCrossToPosition(selectedPosition);

    // This however, directly forces the SNC to the right place.
    this->SetSelectedPosition(selectedPosition);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::MoveAnteriorOrPosterior(MIDASOrientation orientation, int slices)
{
  if (orientation != MIDAS_ORIENTATION_UNKNOWN && slices != 0)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    int selectedSlice = this->GetSelectedSlice(orientation);
    int upDirection = this->GetSliceUpDirection(orientation);

    int nextSelectedSlice = selectedSlice + slices * upDirection;

    int maxSlice = this->GetMaxSlice(orientation);

    if (nextSelectedSlice >= 0 && nextSelectedSlice <= static_cast<int>(maxSlice))
    {
      this->SetSelectedSlice(orientation, nextSelectedSlice);

      /// Note. As a request and for MIDAS compatibility, all the slice have to be forcibly rendered
      /// when scrolling through them by keeping the 'a' or 'z' key pressed.
      /// Otherwise, issues on the scan or in the segmentation may be not seen.

      /// TODO:
      /// In spite of the comment above, it is not right to do any render window update here.
      /// The forced immediate update should be done in the BlockUpdate() function.
//      m_RenderingManager->ForceImmediateUpdate(m_MultiWidget->GetRenderWindow(orientation)->GetRenderWindow());
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetTimeStep(int timeStep)
{
  mitk::SliceNavigationController* snc = m_RenderWindows[MIDAS_ORIENTATION_AXIAL]->GetSliceNavigationController();
  snc->GetTime()->SetPos(timeStep);

  snc = m_RenderWindows[MIDAS_ORIENTATION_SAGITTAL]->GetSliceNavigationController();
  snc->GetTime()->SetPos(timeStep);

  snc = m_RenderWindows[MIDAS_ORIENTATION_CORONAL]->GetSliceNavigationController();
  snc->GetTime()->SetPos(timeStep);
}


//-----------------------------------------------------------------------------
int niftkMultiWindowWidget::GetTimeStep() const
{
  mitk::SliceNavigationController* snc = m_RenderWindows[MIDAS_ORIENTATION_AXIAL]->GetSliceNavigationController();

  return snc->GetTime()->GetPos();
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
    m_SelectedPositionHasChanged = true;

    bool displayEventsWereBlocked = this->BlockDisplayEvents(true);

    mitk::SliceNavigationController* axialSnc = m_RenderWindows[MIDAS_ORIENTATION_AXIAL]->GetSliceNavigationController();
    mitk::SliceNavigationController* sagittalSnc = m_RenderWindows[MIDAS_ORIENTATION_SAGITTAL]->GetSliceNavigationController();
    mitk::SliceNavigationController* coronalSnc = m_RenderWindows[MIDAS_ORIENTATION_CORONAL]->GetSliceNavigationController();

    if (axialSnc->GetCreatedWorldGeometry())
    {
      unsigned pos = axialSnc->GetSlice()->GetPos();
      axialSnc->SelectSliceByPoint(selectedPosition);
      m_SncSliceHasChanged[0] = m_SncSliceHasChanged[0] || pos != axialSnc->GetSlice()->GetPos();
    }
    if (sagittalSnc->GetCreatedWorldGeometry())
    {
      unsigned pos = sagittalSnc->GetSlice()->GetPos();
      sagittalSnc->SelectSliceByPoint(selectedPosition);
      m_SncSliceHasChanged[1] = m_SncSliceHasChanged[1] || pos != sagittalSnc->GetSlice()->GetPos();
    }
    if (coronalSnc->GetCreatedWorldGeometry())
    {
      unsigned pos = coronalSnc->GetSlice()->GetPos();
      coronalSnc->SelectSliceByPoint(selectedPosition);
      m_SncSliceHasChanged[2] = m_SncSliceHasChanged[2] || pos != coronalSnc->GetSlice()->GetPos();
    }

    this->BlockDisplayEvents(displayEventsWereBlocked);

    if (m_RenderWindows[MIDAS_ORIENTATION_AXIAL]->isVisible())
    {
      this->UpdateCursorPosition(MIDAS_ORIENTATION_AXIAL);
    }
    if (m_RenderWindows[MIDAS_ORIENTATION_SAGITTAL]->isVisible())
    {
      this->UpdateCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
    }
    if (m_RenderWindows[MIDAS_ORIENTATION_CORONAL]->isVisible())
    {
      this->UpdateCursorPosition(MIDAS_ORIENTATION_CORONAL);
    }

    this->BlockUpdate(updateWasBlocked);
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
    if (cursorPositions[i] != m_CursorPositions[i])
    {
      m_CursorPositions[i] = cursorPositions[i];
      m_CursorPositionHasChanged[i] = true;
    }
  }
  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::UpdateCursorPosition(MIDASOrientation orientation)
{
  bool updateWasBlocked = this->BlockUpdate(true);

  mitk::BaseRenderer* renderer = m_RenderWindows[orientation]->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  mitk::Vector2D displaySize = displayGeometry->GetSizeInDisplayUnits();

  mitk::Point2D point2DInMm;
  displayGeometry->Map(m_SelectedPosition, point2DInMm);

  mitk::Point2D point2DInPx;
  displayGeometry->WorldToDisplay(point2DInMm, point2DInPx);

  m_CursorPositions[orientation][0] = point2DInPx[0] / displaySize[0];
  m_CursorPositions[orientation][1] = point2DInPx[1] / displaySize[1];
  m_CursorPositionHasChanged[orientation] = true;

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
double niftkMultiWindowWidget::GetScaleFactor(MIDASOrientation orientation) const
{
  return m_ScaleFactors[orientation];
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetScaleFactor(MIDASOrientation orientation, double scaleFactor)
{
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    bool updateWasBlocked = this->BlockUpdate(true);

    if (scaleFactor != m_ScaleFactors[orientation])
    {
      m_ScaleFactors[orientation] = scaleFactor;
      m_ScaleFactorHasChanged[orientation] = true;
      m_Magnifications[orientation] = this->GetMagnification(orientation);
    }

    if (this->GetScaleFactorBinding())
    {
      // Loop over axial, coronal, sagittal windows, the first 3 of 4 QmitkRenderWindow.
      for (int i = 0; i < 3; ++i)
      {
        MIDASOrientation otherOrientation = MIDASOrientation(i);
        if (otherOrientation != orientation && scaleFactor != m_ScaleFactors[otherOrientation])
        {
          m_ScaleFactors[otherOrientation] = scaleFactor;
          m_ScaleFactorHasChanged[otherOrientation] = true;
          m_Magnifications[otherOrientation] = this->GetMagnification(otherOrientation);
        }
      }
    }

    this->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
const std::vector<double>& niftkMultiWindowWidget::GetScaleFactors() const
{
  return m_ScaleFactors;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetScaleFactors(const std::vector<double>& scaleFactors)
{
  bool updateWasBlocked = this->BlockUpdate(true);

  m_ScaleFactors = scaleFactors;
  for (int i = 0; i < 3; ++i)
  {
    MIDASOrientation orientation = MIDASOrientation(i);
    m_ScaleFactorHasChanged[orientation] = true;
    m_Magnifications[orientation] = this->GetMagnification(orientation);
  }

  this->BlockUpdate(updateWasBlocked);
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::ZoomAroundCursorPosition(MIDASOrientation orientation)
{
  if (m_Geometry)
  {
    mitk::DisplayGeometry* displayGeometry = m_RenderWindows[orientation]->GetRenderer()->GetDisplayGeometry();

    mitk::Vector2D displaySize = displayGeometry->GetSizeInDisplayUnits();

    mitk::Point2D focusPoint2DInPx;
    focusPoint2DInPx[0] = m_CursorPositions[orientation][0] * displaySize[0];
    focusPoint2DInPx[1] = m_CursorPositions[orientation][1] * displaySize[1];

    double scaleFactor = m_ScaleFactors[orientation];
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
int niftkMultiWindowWidget::GetDominantAxis(MIDASOrientation orientation) const
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
double niftkMultiWindowWidget::GetMagnification(MIDASOrientation orientation) const
{
  int dominantAxis = this->GetDominantAxis(orientation);

  // Note that we take the scale factor from the first axis always, consequently.
  // Other strategies could be followed as well, e.g. taking the axes with the smallest
  // voxel size.
  double scaleFactorPxPerVx = m_MmPerVx[dominantAxis] / m_ScaleFactors[orientation];

  // Finally, we calculate the magnification from the scale factor.
  double magnification = scaleFactorPxPerVx - 1.0;
  if (magnification < 0.0)
  {
    magnification /= scaleFactorPxPerVx;
  }

  return magnification;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetMagnification(MIDASOrientation orientation, double magnification)
{
  int dominantAxis = this->GetDominantAxis(orientation);

  mitk::Vector3D scaleFactors = this->ComputeScaleFactors(magnification);
  double scaleFactor = scaleFactors[dominantAxis];

  this->SetScaleFactor(orientation, scaleFactor);
}


//-----------------------------------------------------------------------------
mitk::Vector3D niftkMultiWindowWidget::ComputeScaleFactors(double magnification)
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
QmitkRenderWindow* niftkMultiWindowWidget::GetRenderWindow(MIDASOrientation orientation) const
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
MIDASOrientation niftkMultiWindowWidget::GetOrientation(const QmitkRenderWindow* renderWindow) const
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
int niftkMultiWindowWidget::GetSliceUpDirection(int orientation) const
{
  int result = 0;
  if (m_Geometry != NULL)
  {
    result = mitk::GetUpDirection(m_Geometry, itk::Orientation(orientation));
  }
  return result;
}


//-----------------------------------------------------------------------------
void niftkMultiWindowWidget::SetDisplayInteractionsEnabled(bool enabled)
{
  if (enabled == this->AreDisplayInteractionsEnabled())
  {
    // Already enabled/disabled.
    return;
  }

  if (enabled)
  {
    // Here we create our own display interactor...
    m_DisplayInteractor = mitk::DnDDisplayInteractor::New(this);

    us::Module* niftkDnDDisplayModule = us::ModuleRegistry::GetModule("niftkDnDDisplay");
    m_DisplayInteractor->LoadStateMachine("DnDDisplayInteraction.xml", niftkDnDDisplayModule);
    m_DisplayInteractor->SetEventConfig("DnDDisplayConfig.xml", niftkDnDDisplayModule);

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
bool niftkMultiWindowWidget::AreDisplayInteractionsEnabled() const
{
  return m_DisplayInteractor.IsNotNull();
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
      MIDASOrientation orientation = this->GetOrientation();
      if (orientation != MIDAS_ORIENTATION_UNKNOWN)
      {
        this->OnOriginChanged(orientation, true);
        /// We raise the event in another window as well so that the cursors are in sync
        /// along the third axis as well.
        MIDASOrientation someOtherOrientation =
            orientation == MIDAS_ORIENTATION_CORONAL ? MIDAS_ORIENTATION_SAGITTAL : MIDAS_ORIENTATION_CORONAL;
        this->OnOriginChanged(someOtherOrientation, true);
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
      MIDASOrientation orientation = this->GetOrientation();
      if (orientation != MIDAS_ORIENTATION_UNKNOWN)
      {
        for (int i = 0; i < 3; ++i)
        {
          if (i != orientation)
          {
            this->SetScaleFactor(MIDASOrientation(i), m_ScaleFactors[orientation]);
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
      bool rendererUpdated[4] = {false, false, false, false};
      if (m_SelectedRenderWindowHasChanged)
      {
        m_SelectedRenderWindowHasChanged = false;
        if (m_RenderWindows[m_Orientation]->isVisible() && !rendererUpdated[m_Orientation])
        {
          m_RenderingManager->RequestUpdate(m_SelectedRenderWindow->GetRenderWindow());
          rendererUpdated[m_Orientation] = true;
        }
        m_SelectedRenderWindow->setFocus();
        mitk::GlobalInteraction::GetInstance()->SetFocus(m_SelectedRenderWindow->GetRenderer());
        emit SelectedRenderWindowChanged(m_Orientation);
      }
      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_SncGeometryHasChanged[i])
        {
          m_SncGeometryHasChanged[i] = false;
          if (m_RenderWindows[i]->isVisible() && !rendererUpdated[i])
          {
            m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
            rendererUpdated[i] = true;
          }
          m_BlockProcessingSncSignals = true;
          bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
          m_RenderWindows[i]->GetSliceNavigationController()->SendCreatedWorldGeometry();
          this->BlockDisplayEvents(displayEventsWereBlocked);
          m_BlockProcessingSncSignals = false;
        }
        if (m_SncTimeHasChanged[i])
        {
          m_SncTimeHasChanged[i] = false;
          if (m_RenderWindows[i]->isVisible() && !rendererUpdated[i])
          {
            m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
            rendererUpdated[i] = true;
          }
          m_BlockProcessingSncSignals = true;
          bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
          m_RenderWindows[i]->GetSliceNavigationController()->SendTime();
          this->BlockDisplayEvents(displayEventsWereBlocked);
          m_BlockProcessingSncSignals = false;
        }
        if (m_SncSliceHasChanged[i])
        {
          m_SncSliceHasChanged[i] = false;
          if (m_RenderWindows[i]->isVisible() && !rendererUpdated[i])
          {
            m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
            rendererUpdated[i] = true;
          }
          m_BlockProcessingSncSignals = true;
          bool displayEventsWereBlocked = this->BlockDisplayEvents(true);
          m_RenderWindows[i]->GetSliceNavigationController()->SendSlice();
          this->BlockDisplayEvents(displayEventsWereBlocked);
          m_BlockProcessingSncSignals = false;
        }
      }
      if (m_SelectedPositionHasChanged)
      {
        m_SelectedPositionHasChanged = false;
        for (unsigned i = 0; i < 4; ++i)
        {
          if (m_RenderWindows[i]->isVisible() && !rendererUpdated[i])
          {
            m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
            rendererUpdated[i] = true;
          }
        }
        emit SelectedPositionChanged(m_SelectedPosition);
      }
      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_CursorPositionHasChanged[i])
        {
          m_CursorPositionHasChanged[i] = false;
          this->MoveToCursorPosition(MIDASOrientation(i));
          if (m_RenderWindows[i]->isVisible() && !rendererUpdated[i])
          {
            m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
            rendererUpdated[i] = true;
          }
          if (m_RenderWindows[i]->isVisible())
          {
            emit CursorPositionChanged(MIDASOrientation(i), m_CursorPositions[i]);
          }
        }
      }
      for (unsigned i = 0; i < 3; ++i)
      {
        if (m_ScaleFactorHasChanged[i])
        {
          m_ScaleFactorHasChanged[i] = false;
          this->ZoomAroundCursorPosition(MIDASOrientation(i));
          if (m_RenderWindows[i]->isVisible() && !rendererUpdated[i])
          {
            m_RenderingManager->RequestUpdate(m_RenderWindows[i]->GetRenderWindow());
            rendererUpdated[i] = true;
          }
          if (m_RenderWindows[i]->isVisible())
          {
            emit ScaleFactorChanged(MIDASOrientation(i), m_ScaleFactors[i]);
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
