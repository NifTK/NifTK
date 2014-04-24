/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <QStackedLayout>
#include <QDebug>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <QmitkRenderWindow.h>
#include <itkMatrix.h>
#include <itkSpatialOrientationAdapter.h>

#include <itkConversionUtils.h>
#include <mitkPointUtils.h>
#include "niftkSingleViewerWidget.h"
#include "niftkMultiWindowWidget_p.h"


//-----------------------------------------------------------------------------
niftkSingleViewerWidget::niftkSingleViewerWidget(QWidget *parent, mitk::RenderingManager* renderingManager)
: QWidget(parent)
, m_DataStorage(NULL)
, m_GridLayout(NULL)
, m_MultiWidget(NULL)
, m_IsBoundGeometryActive(false)
, m_Geometry(NULL)
, m_BoundGeometry(NULL)
, m_MinimumMagnification(-5.0)
, m_MaximumMagnification(20.0)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_LinkedNavigation(false)
, m_GeometryInitialised(false)
, m_RememberSettingsPerWindowLayout(false)
, m_SingleWindowLayout(WINDOW_LAYOUT_CORONAL)
, m_MultiWindowLayout(WINDOW_LAYOUT_ORTHO)
, m_DnDDisplayStateMachine(0)
{
  if (renderingManager == NULL)
  {
    m_RenderingManager = mitk::RenderingManager::GetInstance();
  }
  else
  {
    m_RenderingManager = renderingManager;
  }

  this->setAcceptDrops(true);

  for (int windowLayoutIndex = 0; windowLayoutIndex < WINDOW_LAYOUT_NUMBER * 2; windowLayoutIndex++)
  {
    m_WindowLayoutInitialised[windowLayoutIndex] = false;
  }

  // Create the main niftkMultiWindowWidget
  m_MultiWidget = new niftkMultiWindowWidget(this, NULL, m_RenderingManager);
  m_MultiWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->EnableLinkedNavigation(false);

  m_GridLayout = new QGridLayout(this);
  m_GridLayout->setObjectName(QString::fromUtf8("niftkSingleViewerWidget::m_GridLayout"));
  m_GridLayout->setContentsMargins(1, 1, 1, 1);
  m_GridLayout->setVerticalSpacing(0);
  m_GridLayout->setHorizontalSpacing(0);
  m_GridLayout->addWidget(m_MultiWidget);

  // Connect to niftkMultiWindowWidget, so we can listen for signals.
  this->connect(this->GetAxialWindow(), SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(this->GetSagittalWindow(), SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(this->GetCoronalWindow(), SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(this->Get3DWindow(), SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(m_MultiWidget, SIGNAL(FocusChanged(int)), SLOT(OnFocusChanged(int)));
  this->connect(m_MultiWidget, SIGNAL(SelectedPositionChanged(const mitk::Point3D&)), SLOT(OnSelectedPositionChanged(const mitk::Point3D&)));
  this->connect(m_MultiWidget, SIGNAL(CursorPositionChanged(int, const mitk::Vector2D&)), SLOT(OnCursorPositionChanged(int, const mitk::Vector2D&)));
  this->connect(m_MultiWidget, SIGNAL(ScaleFactorChanged(int, double)), SLOT(OnScaleFactorChanged(int, double)));
  this->connect(m_MultiWidget, SIGNAL(CursorPositionBindingChanged()), SLOT(OnCursorPositionBindingChanged()));
  this->connect(m_MultiWidget, SIGNAL(ScaleFactorBindingChanged()), SLOT(OnScaleFactorBindingChanged()));

  // Create/Connect the state machine
  mitk::DnDDisplayStateMachine::LoadBehaviourString();
  m_DnDDisplayStateMachine = mitk::DnDDisplayStateMachine::New("DnDDisplayStateMachine", this);
  std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  for (std::size_t j = 0; j < renderWindows.size(); ++j)
  {
    m_DnDDisplayStateMachine->AddRenderer(renderWindows[j]->GetRenderer());
  }
  mitk::GlobalInteraction::GetInstance()->AddListener(m_DnDDisplayStateMachine);
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidget::~niftkSingleViewerWidget()
{
  mitk::GlobalInteraction::GetInstance()->RemoveListener(m_DnDDisplayStateMachine);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnNodesDropped(QmitkRenderWindow *renderWindow, std::vector<mitk::DataNode*> nodes)
{
  emit NodesDropped(this, renderWindow, nodes);
  m_MultiWidget->SetFocused();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnFocusChanged(int windowIndex)
{
  emit FocusChanged(windowIndex);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnSelectedPositionChanged(const mitk::Point3D& selectedPosition)
{
  /// A double click can result in 0 or 1 SelectedPositionChanged event, depending on if you
  /// double click exactly where the cursor is or not.
  /// Therefore, we need to keep the last two selected positions, including the current one.
  if (m_LastSelectedPositions.size() == 2)
  {
    m_LastSelectedPositions.pop_front();
    m_LastSelectedPositionTimes.pop_front();
  }
  m_LastSelectedPositions.push_back(selectedPosition);
  m_LastSelectedPositionTimes.push_back(QTime::currentTime());

  emit SelectedPositionChanged(this, selectedPosition);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnCursorPositionChanged(int windowIndex, const mitk::Vector2D& cursorPosition)
{
  /// A double click can result in up to three CursorPositionChanged events, depending on
  /// how many coordinates of the selected position have changed, if any.
  /// A SelectedPositionChanged event can cause two or three CursorPositionChanged events.
  /// Therefore, we need to keep the last four cursor positions, including the current one.
  if (m_LastCursorPositions.size() == 4)
  {
    m_LastCursorPositions.pop_front();
    m_LastCursorPositionTimes.pop_front();
  }
  m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPositions());
  m_LastCursorPositionTimes.push_back(QTime::currentTime());

  emit CursorPositionChanged(this, MIDASOrientation(windowIndex), cursorPosition);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnScaleFactorChanged(int windowIndex, double scaleFactor)
{
  emit ScaleFactorChanged(this, MIDASOrientation(windowIndex), scaleFactor);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnCursorPositionBindingChanged()
{
  emit CursorPositionBindingChanged(this, this->GetCursorPositionBinding());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnScaleFactorBindingChanged()
{
  emit ScaleFactorBindingChanged(this, this->GetScaleFactorBinding());
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::IsFocused() const
{
  return m_MultiWidget->IsFocused();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetFocused()
{
  m_MultiWidget->SetFocused();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetSelectedRenderWindow() const
{
  return m_MultiWidget->GetSelectedRenderWindow();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetSelectedRenderWindow(QmitkRenderWindow* renderWindow)
{
  m_MultiWidget->SetSelectedRenderWindow(renderWindow);
}


//-----------------------------------------------------------------------------
std::vector<QmitkRenderWindow*> niftkSingleViewerWidget::GetVisibleRenderWindows() const
{
  return m_MultiWidget->GetVisibleRenderWindows();
}


//-----------------------------------------------------------------------------
const std::vector<QmitkRenderWindow*>& niftkSingleViewerWidget::GetRenderWindows() const
{
  return m_MultiWidget->GetRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetAxialWindow() const
{
  return m_MultiWidget->GetRenderWindow1();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetSagittalWindow() const
{
  return m_MultiWidget->GetRenderWindow2();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetCoronalWindow() const
{
  return m_MultiWidget->GetRenderWindow3();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::Get3DWindow() const
{
  return m_MultiWidget->GetRenderWindow4();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetEnabled(bool enabled)
{
  m_MultiWidget->SetEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::IsEnabled() const
{
  return m_MultiWidget->IsEnabled();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorVisible(bool visible)
{
  m_MultiWidget->SetCursorVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::IsCursorVisible() const
{
  return m_MultiWidget->IsCursorVisible();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorGloballyVisible(bool visible)
{
  m_MultiWidget->SetCursorGloballyVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::IsCursorGloballyVisible() const
{
  return m_MultiWidget->IsCursorGloballyVisible();
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::AreDirectionAnnotationsVisible() const
{
  return m_MultiWidget->AreDirectionAnnotationsVisible();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetDirectionAnnotationsVisible(bool visible)
{
  m_MultiWidget->SetDirectionAnnotationsVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::GetShow3DWindowIn2x2WindowLayout() const
{
  return m_MultiWidget->GetShow3DWindowIn2x2WindowLayout();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetShow3DWindowIn2x2WindowLayout(bool enabled)
{
  m_MultiWidget->SetShow3DWindowIn2x2WindowLayout(enabled);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetBackgroundColour(QColor colour)
{
  m_MultiWidget->SetBackgroundColour(colour);
}


//-----------------------------------------------------------------------------
QColor niftkSingleViewerWidget::GetBackgroundColour() const
{
  return m_MultiWidget->GetBackgroundColour();
}


//-----------------------------------------------------------------------------
int niftkSingleViewerWidget::GetMaxSlice(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMaxSlice(orientation);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerWidget::GetMaxTimeStep() const
{
  return m_MultiWidget->GetMaxTimeStep();
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::ContainsRenderWindow(QmitkRenderWindow *renderWindow) const
{
  return m_MultiWidget->ContainsRenderWindow(renderWindow);
}


//-----------------------------------------------------------------------------
MIDASOrientation niftkSingleViewerWidget::GetOrientation() const
{
  /// Note:
  /// This line exploits that the order of orientations are the same and
  /// THREE_D equals to MIDAS_ORIENTATION_UNKNOWN.
  return MIDASOrientation(m_MultiWidget->GetSelectedWindowIndex());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::FitToDisplay(double scaleFactor)
{
  m_MultiWidget->FitRenderWindows(scaleFactor);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible)
{
  m_MultiWidget->SetRendererSpecificVisibility(nodes, visible);
}


//-----------------------------------------------------------------------------
double niftkSingleViewerWidget::GetMinMagnification() const
{
  return m_MinimumMagnification;
}


//-----------------------------------------------------------------------------
double niftkSingleViewerWidget::GetMaxMagnification() const
{
  return m_MaximumMagnification;
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer niftkSingleViewerWidget::GetDataStorage() const
{
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetRememberSettingsPerWindowLayout(bool remember)
{
  m_RememberSettingsPerWindowLayout = remember;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::GetRememberSettingsPerWindowLayout() const
{
  return m_RememberSettingsPerWindowLayout;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  m_MultiWidget->SetDataStorage(m_DataStorage);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::EnableLinkedNavigation(bool linkedNavigation)
{
  m_MultiWidget->SetWidgetPlanesLocked(!linkedNavigation);
  m_LinkedNavigation = linkedNavigation;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::IsLinkedNavigationEnabled() const
{
  return m_LinkedNavigation;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetDisplayInteractionsEnabled(bool enabled)
{
  m_MultiWidget->SetDisplayInteractionsEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::AreDisplayInteractionsEnabled() const
{
  return m_MultiWidget->AreDisplayInteractionsEnabled();
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::GetCursorPositionBinding() const
{
  return m_MultiWidget->GetCursorPositionBinding();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorPositionBinding(bool cursorPositionBinding)
{
  m_MultiWidget->SetCursorPositionBinding(cursorPositionBinding);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::GetScaleFactorBinding() const
{
  return m_MultiWidget->GetScaleFactorBinding();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetScaleFactorBinding(bool scaleFactorBinding)
{
  m_MultiWidget->SetScaleFactorBinding(scaleFactorBinding);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::RequestUpdate()
{
  m_MultiWidget->RequestUpdate();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::ResetLastPositions()
{
  m_LastSelectedPositions.clear();
  m_LastSelectedPositionTimes.clear();
  m_LastCursorPositions.clear();
  m_LastCursorPositionTimes.clear();

  m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
  m_LastSelectedPositionTimes.push_back(QTime::currentTime());
  m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPositions());
  m_LastCursorPositionTimes.push_back(QTime::currentTime());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetGeometry(mitk::TimeGeometry::Pointer timeGeometry)
{
  assert(timeGeometry);
  m_Geometry = timeGeometry;
  m_GeometryInitialised = false;

  if (!m_IsBoundGeometryActive)
  {
    bool updateWasBlocked = m_MultiWidget->BlockUpdate(true);

    m_MultiWidget->SetTimeGeometry(timeGeometry);

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      this->ResetLastPositions();
      m_WindowLayoutInitialised[Index(m_WindowLayout)] = true;
    }

    for (int otherWindowLayout = 0; otherWindowLayout < WINDOW_LAYOUT_NUMBER; otherWindowLayout++)
    {
      if (otherWindowLayout != m_WindowLayout)
      {
        m_WindowLayoutInitialised[Index(otherWindowLayout)] = false;
      }
    }

    m_MultiWidget->BlockUpdate(updateWasBlocked);
  }

  emit GeometryChanged(this, timeGeometry);
}


//-----------------------------------------------------------------------------
mitk::TimeGeometry::Pointer niftkSingleViewerWidget::GetGeometry()
{
  assert(m_Geometry);
  return m_Geometry;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetBoundGeometry(mitk::TimeGeometry::Pointer timeGeometry)
{
  assert(timeGeometry);
  m_BoundGeometry = timeGeometry;
  m_GeometryInitialised = false;

  if (m_IsBoundGeometryActive)
  {
    bool updateWasBlocked = m_MultiWidget->BlockUpdate(true);

    m_MultiWidget->SetTimeGeometry(timeGeometry);

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      this->ResetLastPositions();
      m_WindowLayoutInitialised[Index(m_WindowLayout)] = true;
    }

    for (int otherWindowLayout = 0; otherWindowLayout < WINDOW_LAYOUT_NUMBER; otherWindowLayout++)
    {
      if (otherWindowLayout != m_WindowLayout)
      {
        m_WindowLayoutInitialised[Index(otherWindowLayout)] = false;
      }
    }

    m_MultiWidget->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::IsBoundGeometryActive()
{
  return m_IsBoundGeometryActive;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetBoundGeometryActive(bool isBoundGeometryActive)
{
  if (isBoundGeometryActive == m_IsBoundGeometryActive)
  {
    // No change, nothing to do.
    return;
  }

  mitk::TimeGeometry* timeGeometry = isBoundGeometryActive ? m_BoundGeometry : m_Geometry;
  m_MultiWidget->SetTimeGeometry(timeGeometry);

  m_IsBoundGeometryActive = isBoundGeometryActive;
  //  m_WindowLayout = WINDOW_LAYOUT_UNKNOWN;
}


//-----------------------------------------------------------------------------
int niftkSingleViewerWidget::GetSelectedSlice(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetSelectedSlice(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetSelectedSlice(MIDASOrientation orientation, int selectedSlice)
{
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_MultiWidget->SetSelectedSlice(orientation, selectedSlice);
  }
}


//-----------------------------------------------------------------------------
int niftkSingleViewerWidget::GetTimeStep() const
{
  return m_MultiWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetTimeStep(int timeStep)
{
  m_MultiWidget->SetTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
WindowLayout niftkSingleViewerWidget::GetWindowLayout() const
{
  return m_WindowLayout;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout)
{
  if (windowLayout != WINDOW_LAYOUT_UNKNOWN && windowLayout != m_WindowLayout)
  {
    mitk::TimeGeometry* geometry = m_IsBoundGeometryActive ? m_BoundGeometry : m_Geometry;

    // If for whatever reason, we have no geometry... bail out.
    if (!geometry)
    {
      return;
    }

    bool updateWasBlocked = m_MultiWidget->BlockUpdate(true);

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      // If we have a currently valid window layout/orientation, then store the current position, so we can switch back to it if necessary.
      m_CursorPositions[Index(m_WindowLayout)] = m_MultiWidget->GetCursorPositions();
      m_ScaleFactors[Index(m_WindowLayout)] = m_MultiWidget->GetScaleFactors();
      m_CursorPositionBinding[Index(m_WindowLayout)] = m_MultiWidget->GetCursorPositionBinding();
      m_ScaleFactorBinding[Index(m_WindowLayout)] = m_MultiWidget->GetScaleFactorBinding();
    }

    // This will initialise the whole QmitkStdMultiWidget according to the supplied geometry (normally an image).
    m_MultiWidget->SetWindowLayout(windowLayout);

    // Now store the current window layout/orientation.
    m_WindowLayout = windowLayout;
    if (! ::IsSingleWindowLayout(windowLayout))
    {
      m_MultiWindowLayout = windowLayout;
    }

    if (!m_GeometryInitialised)
    {
      m_GeometryInitialised = true;
      m_MultiWidget->SetTimeStep(0);
      m_MultiWidget->SetSelectedPosition(geometry->GetCenterInWorld());
    }

    // Now, in MIDAS, which only shows 2D window layouts, if we revert to a previous window layout,
    // we should go back to the same slice index, time step, cursor position on display, scale factor.
    if (m_RememberSettingsPerWindowLayout && m_WindowLayoutInitialised[Index(windowLayout)])
    {
      m_MultiWidget->SetCursorPositions(m_CursorPositions[Index(windowLayout)]);
      m_MultiWidget->SetScaleFactors(m_ScaleFactors[Index(windowLayout)]);
      m_MultiWidget->SetCursorPositionBinding(m_CursorPositionBinding[Index(windowLayout)]);
      m_MultiWidget->SetScaleFactorBinding(m_ScaleFactorBinding[Index(windowLayout)]);
    }
    else
    {
      /// If the positions are not remembered for each window layout, we reset them.
      /// This moves the displayed region to the middle of the render window and the
      /// sets the scale factor so that the image fits the render window.
      m_MultiWidget->FitRenderWindows();
      m_MultiWidget->SetCursorPositionBinding(::IsMultiWindowLayout(windowLayout));
      m_MultiWidget->SetScaleFactorBinding(::IsMultiWindowLayout(windowLayout));

      m_WindowLayoutInitialised[Index(windowLayout)] = true;
    }

    this->ResetLastPositions();

    m_MultiWidget->BlockUpdate(updateWasBlocked);
  }
}


//-----------------------------------------------------------------------------
const mitk::Point3D& niftkSingleViewerWidget::GetSelectedPosition() const
{
  return m_MultiWidget->GetSelectedPosition();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition)
{
  if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetSelectedPosition(selectedPosition);
  }
}


//-----------------------------------------------------------------------------
mitk::Vector2D niftkSingleViewerWidget::GetCursorPosition(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetCursorPosition(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorPosition(MIDASOrientation orientation, const mitk::Vector2D& cursorPosition)
{
  if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetCursorPosition(orientation, cursorPosition);
  }
}


//-----------------------------------------------------------------------------
const std::vector<mitk::Vector2D>& niftkSingleViewerWidget::GetCursorPositions() const
{
  return m_MultiWidget->GetCursorPositions();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorPositions(const std::vector<mitk::Vector2D>& cursorPositions)
{
  if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetCursorPositions(cursorPositions);
  }
}


//-----------------------------------------------------------------------------
double niftkSingleViewerWidget::GetScaleFactor(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetScaleFactor(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetScaleFactor(MIDASOrientation orientation, double scaleFactor)
{
  if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetScaleFactor(orientation, scaleFactor);
  }
}


//-----------------------------------------------------------------------------
const std::vector<double>& niftkSingleViewerWidget::GetScaleFactors() const
{
  return m_MultiWidget->GetScaleFactors();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetScaleFactors(const std::vector<double>& scaleFactors)
{
  if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetScaleFactors(scaleFactors);
  }
}


//-----------------------------------------------------------------------------
double niftkSingleViewerWidget::GetMagnification(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMagnification(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetMagnification(MIDASOrientation orientation, double magnification)
{
  if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetMagnification(orientation, magnification);
  }
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::paintEvent(QPaintEvent *event)
{
  QWidget::paintEvent(event);
  std::vector<QmitkRenderWindow*> renderWindows = this->GetVisibleRenderWindows();
  for (std::size_t i = 0; i < renderWindows.size(); i++)
  {
    renderWindows[i]->GetVtkRenderWindow()->Render();
  }
}


//-----------------------------------------------------------------------------
std::vector<mitk::DataNode*> niftkSingleViewerWidget::GetWidgetPlanes()
{
  std::vector<mitk::DataNode*> result;
  result.push_back(m_MultiWidget->GetWidgetPlane1());
  result.push_back(m_MultiWidget->GetWidgetPlane2());
  result.push_back(m_MultiWidget->GetWidgetPlane3());
  return result;
}


//-----------------------------------------------------------------------------
int niftkSingleViewerWidget::GetSliceUpDirection(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetSliceUpDirection(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetDefaultSingleWindowLayout(WindowLayout windowLayout)
{
  m_SingleWindowLayout = windowLayout;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetDefaultMultiWindowLayout(WindowLayout windowLayout)
{
  m_MultiWindowLayout = windowLayout;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::MoveAnterior()
{
  m_MultiWidget->MoveAnteriorOrPosterior(this->GetOrientation(), +1);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::MovePosterior()
{
  m_MultiWidget->MoveAnteriorOrPosterior(this->GetOrientation(), -1);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchToAxial()
{
  this->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  emit WindowLayoutChanged(this, WINDOW_LAYOUT_AXIAL);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchToSagittal()
{
  this->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  emit WindowLayoutChanged(this, WINDOW_LAYOUT_SAGITTAL);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchToCoronal()
{
  this->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  emit WindowLayoutChanged(this, WINDOW_LAYOUT_CORONAL);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchTo3D()
{
  this->SetWindowLayout(WINDOW_LAYOUT_3D);
  emit WindowLayoutChanged(this, WINDOW_LAYOUT_3D);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::ToggleMultiWindowLayout()
{
  WindowLayout nextWindowLayout;

  if (::IsSingleWindowLayout(m_WindowLayout))
  {
    nextWindowLayout = m_MultiWindowLayout;
  }
  else
  {
    switch (this->GetOrientation())
    {
    case MIDAS_ORIENTATION_AXIAL:
      nextWindowLayout = WINDOW_LAYOUT_AXIAL;
      break;
    case MIDAS_ORIENTATION_SAGITTAL:
      nextWindowLayout = WINDOW_LAYOUT_SAGITTAL;
      break;
    case MIDAS_ORIENTATION_CORONAL:
      nextWindowLayout = WINDOW_LAYOUT_CORONAL;
      break;
    case MIDAS_ORIENTATION_UNKNOWN:
      nextWindowLayout = WINDOW_LAYOUT_3D;
      break;
    default:
      nextWindowLayout = WINDOW_LAYOUT_CORONAL;
    }
  }

  /// We have to discard the selected position changes during the double clicking.
  QTime currentTime = QTime::currentTime();
  int doubleClickInterval = QApplication::doubleClickInterval();
  QTime doubleClickTime = currentTime;

  while (m_LastSelectedPositions.size() > 1 && m_LastSelectedPositionTimes.back().msecsTo(currentTime) < doubleClickInterval)
  {
    doubleClickTime = m_LastSelectedPositionTimes.back();
    m_LastSelectedPositions.pop_back();
    m_LastSelectedPositionTimes.pop_back();
  }

  /// We also discard the cursor position changes since the double click.
  while (m_LastCursorPositions.size() > 1 && m_LastCursorPositionTimes.back() >= doubleClickTime)
  {
    m_LastCursorPositions.pop_back();
    m_LastCursorPositionTimes.pop_back();
  }

  bool updateWasBlocked = m_MultiWidget->BlockUpdate(true);

  /// Restore the selected position and cursor positions from before the double clicking.
  m_MultiWidget->SetSelectedPosition(m_LastSelectedPositions.back());
  m_MultiWidget->SetCursorPositions(m_LastCursorPositions.back());

  this->SetWindowLayout(nextWindowLayout);

  m_MultiWidget->BlockUpdate(updateWasBlocked);

  emit WindowLayoutChanged(this, nextWindowLayout);

  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::ToggleCursorVisibility()
{
  bool visible = !this->IsCursorVisible();

  this->SetCursorVisible(visible);

  this->RequestUpdate();

  emit CursorVisibilityChanged(this, visible);

  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::BlockUpdate(bool blocked)
{
  return m_MultiWidget->BlockUpdate(blocked);
}
