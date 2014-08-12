/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleViewerWidget.h"

#include <QStackedLayout>
#include <QDebug>
#include <QmitkRenderWindow.h>
#include <itkMatrix.h>
#include <itkSpatialOrientationAdapter.h>

#include <itkConversionUtils.h>

#include <usGetModuleContext.h>
#include <usModuleRegistry.h>
#include <mitkGlobalInteraction.h>
#include <mitkPointUtils.h>

#include "niftkMultiWindowWidget_p.h"


//-----------------------------------------------------------------------------
niftkSingleViewerWidget::niftkSingleViewerWidget(QWidget *parent, mitk::RenderingManager* renderingManager, const QString& name)
: QWidget(parent)
, m_GridLayout(NULL)
, m_MultiWidget(NULL)
, m_IsBoundTimeGeometryActive(false)
, m_TimeGeometry(NULL)
, m_BoundTimeGeometry(NULL)
, m_MinimumMagnification(-5.0)
, m_MaximumMagnification(20.0)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
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
  m_MultiWidget = new niftkMultiWindowWidget(this, NULL, m_RenderingManager, mitk::BaseRenderer::RenderingMode::Standard, name);
  m_MultiWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

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
  this->connect(m_MultiWidget, SIGNAL(WindowLayoutChanged(WindowLayout)), SLOT(OnWindowLayoutChanged(WindowLayout)));
  this->connect(m_MultiWidget, SIGNAL(SelectedPositionChanged(const mitk::Point3D&)), SLOT(OnSelectedPositionChanged(const mitk::Point3D&)));
  this->connect(m_MultiWidget, SIGNAL(TimeStepChanged(int)), SIGNAL(TimeStepChanged(int)));
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
  // Release the display interactor.
  this->SetDisplayInteractionsEnabled(false);

  mitk::GlobalInteraction::GetInstance()->RemoveListener(m_DnDDisplayStateMachine);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnNodesDropped(QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes)
{
  Q_UNUSED(renderWindow);
  emit NodesDropped(nodes);
  m_MultiWidget->SetFocused();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnWindowLayoutChanged(WindowLayout windowLayout)
{
  emit WindowLayoutChanged(windowLayout);
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

  emit SelectedPositionChanged(selectedPosition);
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

  emit CursorPositionChanged(WindowOrientation(windowIndex), cursorPosition);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnScaleFactorChanged(int windowIndex, double scaleFactor)
{
  emit ScaleFactorChanged(WindowOrientation(windowIndex), scaleFactor);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnCursorPositionBindingChanged()
{
  emit CursorPositionBindingChanged(this->GetCursorPositionBinding());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnScaleFactorBindingChanged()
{
  emit ScaleFactorBindingChanged(this->GetScaleFactorBinding());
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
  return m_MultiWidget->GetRenderWindows()[0];
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetSagittalWindow() const
{
  return m_MultiWidget->GetRenderWindows()[1];
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetCoronalWindow() const
{
  return m_MultiWidget->GetRenderWindows()[2];
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::Get3DWindow() const
{
  return m_MultiWidget->GetRenderWindows()[3];
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
bool niftkSingleViewerWidget::IsCursorVisible() const
{
  return m_MultiWidget->IsCursorVisible();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorVisible(bool visible)
{
  m_MultiWidget->SetCursorVisible(visible);
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
int niftkSingleViewerWidget::GetMaxSlice(WindowOrientation orientation) const
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
WindowOrientation niftkSingleViewerWidget::GetOrientation() const
{
  /// Note:
  /// This line exploits that the order of orientations are the same and
  /// THREE_D equals to WINDOW_ORIENTATION_UNKNOWN.
  return WindowOrientation(m_MultiWidget->GetSelectedWindowIndex());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::FitToDisplay(double scaleFactor)
{
  m_MultiWidget->FitRenderWindows(scaleFactor);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetVisibility(std::vector<mitk::DataNode*> nodes, bool visible)
{
  m_MultiWidget->SetVisibility(nodes, visible);
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
bool niftkSingleViewerWidget::IsLinkedNavigationEnabled() const
{
  return m_MultiWidget->IsLinkedNavigationEnabled();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetLinkedNavigationEnabled(bool linkedNavigationEnabled)
{
  m_MultiWidget->SetLinkedNavigationEnabled(linkedNavigationEnabled);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetDisplayInteractionsEnabled(bool enabled)
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
bool niftkSingleViewerWidget::AreDisplayInteractionsEnabled() const
{
  return m_DisplayInteractor.IsNotNull();
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
const mitk::TimeGeometry* niftkSingleViewerWidget::GetTimeGeometry() const
{
  assert(m_TimeGeometry);
  return m_TimeGeometry;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetTimeGeometry(const mitk::TimeGeometry* timeGeometry)
{
  assert(timeGeometry);
  m_TimeGeometry = timeGeometry;
  m_GeometryInitialised = false;

  if (!m_IsBoundTimeGeometryActive)
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

  emit TimeGeometryChanged(timeGeometry);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetBoundTimeGeometry(const mitk::TimeGeometry* timeGeometry)
{
  assert(timeGeometry);
  m_BoundTimeGeometry = timeGeometry;
  m_GeometryInitialised = false;

  if (m_IsBoundTimeGeometryActive)
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
bool niftkSingleViewerWidget::IsBoundTimeGeometryActive()
{
  return m_IsBoundTimeGeometryActive;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetBoundTimeGeometryActive(bool isBoundTimeGeometryActive)
{
  if (isBoundTimeGeometryActive == m_IsBoundTimeGeometryActive)
  {
    // No change, nothing to do.
    return;
  }

  const mitk::TimeGeometry* timeGeometry = isBoundTimeGeometryActive ? m_BoundTimeGeometry : m_TimeGeometry;
  m_MultiWidget->SetTimeGeometry(timeGeometry);

  m_IsBoundTimeGeometryActive = isBoundTimeGeometryActive;
  //  m_WindowLayout = WINDOW_LAYOUT_UNKNOWN;
}


//-----------------------------------------------------------------------------
int niftkSingleViewerWidget::GetSelectedSlice(WindowOrientation orientation) const
{
  return m_MultiWidget->GetSelectedSlice(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetSelectedSlice(WindowOrientation orientation, int selectedSlice)
{
  if (orientation != WINDOW_ORIENTATION_UNKNOWN)
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


// --------------------------------------------------------------------------
mitk::Vector2D niftkSingleViewerWidget::GetCentrePosition(int windowIndex)
{
  mitk::BaseRenderer* renderer = this->GetRenderWindows()[windowIndex]->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  const mitk::Geometry2D* worldGeometry2D = renderer->GetCurrentWorldGeometry2D();

  mitk::Point3D centreInMm = worldGeometry2D->GetCenter();
  mitk::Point2D centreInMm2D;
  displayGeometry->Map(centreInMm, centreInMm2D);
  mitk::Point2D centreInPx2D;
  displayGeometry->WorldToDisplay(centreInMm2D, centreInPx2D);

  mitk::Vector2D centrePosition;
  centrePosition[0] = centreInPx2D[0] / this->GetRenderWindows()[windowIndex]->width();
  centrePosition[1] = centreInPx2D[1] / this->GetRenderWindows()[windowIndex]->height();

  return centrePosition;
}


// --------------------------------------------------------------------------
std::vector<mitk::Vector2D> niftkSingleViewerWidget::GetCentrePositions()
{
  const std::vector<QmitkRenderWindow*>& renderWindows = this->GetRenderWindows();

  std::vector<mitk::Vector2D> centrePositions(3);
  for (std::size_t windowIndex = 0; windowIndex < 3; ++windowIndex)
  {
    if (renderWindows[windowIndex]->isVisible())
    {
      centrePositions[windowIndex] = this->GetCentrePosition(windowIndex);
    }
    else
    {
      /// This is not necessary but makes the printed results easier to read.
      centrePositions[windowIndex][0] = std::numeric_limits<double>::quiet_NaN();
      centrePositions[windowIndex][1] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  return centrePositions;
}


//-----------------------------------------------------------------------------
mitk::Vector2D niftkSingleViewerWidget::GetCursorPositionFromCentre(int windowIndex, const mitk::Vector2D& centrePosition)
{
  mitk::BaseRenderer* renderer = this->GetRenderWindows()[windowIndex]->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  const mitk::Geometry2D* worldGeometry2D = renderer->GetCurrentWorldGeometry2D();

  const mitk::Point3D& selectedPosition = m_MultiWidget->GetSelectedPosition();
  double scaleFactor = m_MultiWidget->GetScaleFactor(windowIndex);

  /// World centre relative to the display origin, in pixels.
  mitk::Vector2D centreOnDisplayInPx;
  centreOnDisplayInPx[0] = centrePosition[0] * renderer->GetSizeX();
  centreOnDisplayInPx[1] = centrePosition[1] * renderer->GetSizeY();

  /// World centre in mm.
  mitk::Point3D centreInMm = worldGeometry2D->GetCenter();
  mitk::Point2D centreInMm2D;
  displayGeometry->Map(centreInMm, centreInMm2D);

  /// World origin in mm.
  mitk::Point3D originInMm = worldGeometry2D->GetOrigin();
  mitk::Point2D originInMm2D;
  displayGeometry->Map(originInMm, originInMm2D);

  /// World centre relative to the world origin, in pixels.
  mitk::Vector2D centreInWorldInPx;
  centreInWorldInPx[0] = centreInMm2D[0] - originInMm2D[0];
  centreInWorldInPx[1] = centreInMm2D[1] - originInMm2D[1];
  centreInWorldInPx /= scaleFactor;

  /// World origin relative to the display origin, in pixels.
  mitk::Vector2D worldOriginInPx;
  worldOriginInPx = centreOnDisplayInPx - centreInWorldInPx;

  /// Cursor position relative to the world origin.
  mitk::Point2D cursorPosition2DInMm;
  displayGeometry->Map(selectedPosition, cursorPosition2DInMm);
  mitk::Vector2D cursorPositionInWorldInPx;
  cursorPositionInWorldInPx[0] = cursorPosition2DInMm[0] / scaleFactor;
  cursorPositionInWorldInPx[1] = cursorPosition2DInMm[1] / scaleFactor;

  mitk::Vector2D cursorPosition;
  cursorPosition = worldOriginInPx + cursorPositionInWorldInPx;
  cursorPosition[0] /= renderer->GetSizeX();
  cursorPosition[1] /= renderer->GetSizeY();

  return cursorPosition;
}


// --------------------------------------------------------------------------
std::vector<mitk::Vector2D> niftkSingleViewerWidget::GetCursorPositionsFromCentres(const std::vector<mitk::Vector2D>& centrePositions)
{
  const std::vector<QmitkRenderWindow*>& renderWindows = this->GetRenderWindows();

  std::vector<mitk::Vector2D> cursorPositions(3);
  for (std::size_t windowIndex = 0; windowIndex < 3; ++windowIndex)
  {
    if (renderWindows[windowIndex]->isVisible())
    {
      cursorPositions[windowIndex] = this->GetCursorPositionFromCentre(windowIndex, centrePositions[windowIndex]);
    }
    else
    {
      /// This is not necessary but makes the printed results easier to read.
      cursorPositions[windowIndex][0] = std::numeric_limits<double>::quiet_NaN();
      cursorPositions[windowIndex][1] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  return cursorPositions;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout)
{
  if (windowLayout != WINDOW_LAYOUT_UNKNOWN && windowLayout != m_WindowLayout)
  {
    const mitk::TimeGeometry* timeGeometry = m_IsBoundTimeGeometryActive ? m_BoundTimeGeometry : m_TimeGeometry;

    // If for whatever reason, we have no geometry... bail out.
    if (!timeGeometry)
    {
      return;
    }

    bool updateWasBlocked = m_MultiWidget->BlockUpdate(true);

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      // If we have a currently valid window layout/orientation, then store the current position, so we can switch back to it if necessary.
      m_SelectedPositions[Index(m_WindowLayout)] = m_MultiWidget->GetSelectedPosition();
      m_CentrePositions[Index(m_WindowLayout)] = this->GetCentrePositions();
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
    }

    // Now, in MIDAS, which only shows 2D window layouts, if we revert to a previous window layout,
    // we should go back to the same cursor position on display and scale factor.
    if (m_RememberSettingsPerWindowLayout && m_WindowLayoutInitialised[Index(windowLayout)])
    {
      /// Note: We set the scale factors first because the cursor position calculation relies on it.
      m_MultiWidget->SetScaleFactors(m_ScaleFactors[Index(windowLayout)]);
      std::vector<mitk::Vector2D> cursorPositions = this->GetCursorPositionsFromCentres(m_CentrePositions[Index(windowLayout)]);
      m_MultiWidget->SetCursorPositions(cursorPositions);
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
mitk::Vector2D niftkSingleViewerWidget::GetCursorPosition(WindowOrientation orientation) const
{
  return m_MultiWidget->GetCursorPosition(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorPosition(WindowOrientation orientation, const mitk::Vector2D& cursorPosition)
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
double niftkSingleViewerWidget::GetScaleFactor(WindowOrientation orientation) const
{
  return m_MultiWidget->GetScaleFactor(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetScaleFactor(WindowOrientation orientation, double scaleFactor)
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
double niftkSingleViewerWidget::GetMagnification(WindowOrientation orientation) const
{
  return m_MultiWidget->GetMagnification(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetMagnification(WindowOrientation orientation, double magnification)
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
int niftkSingleViewerWidget::GetSliceUpDirection(WindowOrientation orientation) const
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
bool niftkSingleViewerWidget::SelectPreviousTimeStep()
{
  int timeStep = m_MultiWidget->GetTimeStep() - 1;

  if (timeStep >= 0)
  {
    m_MultiWidget->SetTimeStep(timeStep);
  }

  return timeStep == m_MultiWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SelectNextTimeStep()
{
  int timeStep = m_MultiWidget->GetTimeStep() + 1;

  if (timeStep <= m_MultiWidget->GetMaxTimeStep())
  {
    m_MultiWidget->SetTimeStep(timeStep);
  }

  return timeStep == m_MultiWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchToAxial()
{
  this->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchToSagittal()
{
  this->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchToCoronal()
{
  this->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::SwitchTo3D()
{
  this->SetWindowLayout(WINDOW_LAYOUT_3D);
  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::ToggleMultiWindowLayout()
{
  if (m_GeometryInitialised)
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
      case WINDOW_ORIENTATION_AXIAL:
        nextWindowLayout = WINDOW_LAYOUT_AXIAL;
        break;
      case WINDOW_ORIENTATION_SAGITTAL:
        nextWindowLayout = WINDOW_LAYOUT_SAGITTAL;
        break;
      case WINDOW_ORIENTATION_CORONAL:
        nextWindowLayout = WINDOW_LAYOUT_CORONAL;
        break;
      case WINDOW_ORIENTATION_UNKNOWN:
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
  }

  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::ToggleCursorVisibility()
{
  bool visible = !this->IsCursorVisible();

  this->SetCursorVisible(visible);

  this->RequestUpdate();

  emit CursorVisibilityChanged(visible);

  return true;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::BlockUpdate(bool blocked)
{
  return m_MultiWidget->BlockUpdate(blocked);
}
