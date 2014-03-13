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
  this->connect(m_MultiWidget, SIGNAL(SelectedRenderWindowChanged(int)), SLOT(OnSelectedRenderWindowChanged(int)));
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
  m_MultiWidget->SetSelected(true);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnSelectedRenderWindowChanged(int windowIndex)
{
  emit SelectedRenderWindowChanged(MIDASOrientation(windowIndex));
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
void niftkSingleViewerWidget::SetSelected(bool selected)
{
  m_MultiWidget->SetSelected(selected);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::IsSelected() const
{
  return m_MultiWidget->IsSelected();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetSelectedRenderWindow() const
{
  QmitkRenderWindow* selectedRenderWindow = m_MultiWidget->GetSelectedRenderWindow();
  if (!selectedRenderWindow)
  {
    std::vector<QmitkRenderWindow*> visibleRenderWindows = m_MultiWidget->GetVisibleRenderWindows();
    if (!visibleRenderWindows.empty())
    {
      selectedRenderWindow = visibleRenderWindows[0];
    }
    else
    {
      selectedRenderWindow = m_MultiWidget->GetRenderWindow1();
    }
  }
  return selectedRenderWindow;
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
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

  QmitkRenderWindow* renderWindow = this->GetSelectedRenderWindow();
  if (!renderWindow)
  {
    std::vector<QmitkRenderWindow*> visibleRenderWindows = m_MultiWidget->GetVisibleRenderWindows();
    if (!visibleRenderWindows.empty())
    {
      renderWindow = visibleRenderWindows[0];
    }
  }

  if (renderWindow == this->GetAxialWindow())
  {
    orientation = MIDAS_ORIENTATION_AXIAL;
  }
  else if (renderWindow == this->GetSagittalWindow())
  {
    orientation = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (renderWindow == this->GetCoronalWindow())
  {
    orientation = MIDAS_ORIENTATION_CORONAL;
  }

  return orientation;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::FitToDisplay()
{
  m_MultiWidget->FitRenderWindows();
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
void niftkSingleViewerWidget::EnableLinkedNavigation(bool enabled)
{
  m_MultiWidget->SetWidgetPlanesLocked(!enabled);
  m_LinkedNavigation = enabled;
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
void niftkSingleViewerWidget::SetGeometry(mitk::TimeGeometry::Pointer timeGeometry)
{
  assert(timeGeometry);
  m_Geometry = timeGeometry;

  if (!m_IsBoundGeometryActive)
  {
    bool updateWasBlocked = m_MultiWidget->BlockUpdate(true);

    m_MultiWidget->SetTimeGeometry(timeGeometry);

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      m_LastSelectedPositions.clear();
      m_LastSelectedPositionTimes.clear();
      m_LastCursorPositions.clear();
      m_LastCursorPositionTimes.clear();

      m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
      m_LastSelectedPositionTimes.push_back(QTime::currentTime());
      m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPositions());
      m_LastCursorPositionTimes.push_back(QTime::currentTime());

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

  if (m_IsBoundGeometryActive)
  {
    bool updateWasBlocked = m_MultiWidget->BlockUpdate(true);

    m_MultiWidget->SetTimeGeometry(timeGeometry);

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      m_LastSelectedPositions.clear();
      m_LastSelectedPositionTimes.clear();
      m_LastCursorPositions.clear();
      m_LastCursorPositionTimes.clear();

      m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
      m_LastSelectedPositionTimes.push_back(QTime::currentTime());
      m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPositions());
      m_LastCursorPositionTimes.push_back(QTime::currentTime());

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
void niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout, bool dontSetSelectedPosition, bool dontSetCursorPositions, bool dontSetScaleFactors)
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

    bool wasSelected = this->IsSelected();
    QmitkRenderWindow* selectedRenderWindow = m_MultiWidget->GetSelectedRenderWindow();

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      // If we have a currently valid window layout/orientation, then store the current position, so we can switch back to it if necessary.
      m_SelectedPositions[Index(m_WindowLayout)] = m_LastSelectedPositions.back();
      m_TimeSteps[Index(0)] = m_MultiWidget->GetTimeStep();
      m_CursorPositions[Index(m_WindowLayout)] = m_LastCursorPositions.back();
      m_ScaleFactors[Index(m_WindowLayout)] = m_MultiWidget->GetScaleFactors();
      m_SelectedRenderWindow[Index(m_WindowLayout)] = m_MultiWidget->GetSelectedRenderWindow();
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

    // Now, in MIDAS, which only shows 2D window layouts, if we revert to a previous window layout,
    // we should go back to the same slice index, time step, cursor position on display, scale factor.
    bool hasBeenInitialised = m_WindowLayoutInitialised[Index(windowLayout)];
    if (m_RememberSettingsPerWindowLayout && hasBeenInitialised)
    {
      if (!dontSetSelectedPosition)
      {
        m_MultiWidget->SetTimeStep(m_TimeSteps[Index(0)]);
        m_MultiWidget->SetSelectedPosition(m_SelectedPositions[Index(windowLayout)]);
      }

      if (!dontSetCursorPositions)
      {
        m_MultiWidget->SetCursorPositions(m_CursorPositions[Index(windowLayout)]);
      }

      if (!dontSetScaleFactors)
      {
        m_MultiWidget->SetScaleFactors(m_ScaleFactors[Index(windowLayout)]);
      }

      if (wasSelected)
      {
        m_MultiWidget->SetSelectedRenderWindow(m_SelectedRenderWindow[Index(windowLayout)]);
      }

      if (!dontSetCursorPositions)
      {
        m_MultiWidget->SetCursorPositionBinding(m_CursorPositionBinding[Index(windowLayout)]);
      }
      if (!dontSetScaleFactors)
      {
        m_MultiWidget->SetScaleFactorBinding(m_ScaleFactorBinding[Index(windowLayout)]);
      }

      m_LastSelectedPositions.clear();
      m_LastSelectedPositionTimes.clear();
      m_LastCursorPositions.clear();
      m_LastCursorPositionTimes.clear();

      m_LastSelectedPositions.push_back(m_SelectedPositions[Index(windowLayout)]);
      m_LastSelectedPositionTimes.push_back(QTime::currentTime());
      m_LastCursorPositions.push_back(m_CursorPositions[Index(windowLayout)]);
      m_LastCursorPositionTimes.push_back(QTime::currentTime());
    }
    else
    {
      /// If the positions are not remembered for each window layout,
      /// we reset them.
//      if (!hasBeenInitialised)
      {
        if (!dontSetSelectedPosition)
        {
          m_MultiWidget->SetTimeStep(0);
          m_MultiWidget->SetSelectedPosition(geometry->GetCenterInWorld());
        }
        if (!dontSetCursorPositions || !dontSetScaleFactors)
        {
          m_MultiWidget->FitRenderWindows();
        }

        m_LastSelectedPositions.clear();
        m_LastSelectedPositionTimes.clear();
        m_LastCursorPositions.clear();
        m_LastCursorPositionTimes.clear();

        m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
        m_LastSelectedPositionTimes.push_back(QTime::currentTime());
        m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPositions());
        m_LastCursorPositionTimes.push_back(QTime::currentTime());

        m_MultiWidget->SetCursorPositionBinding(::IsMultiWindowLayout(windowLayout));
        m_MultiWidget->SetScaleFactorBinding(::IsMultiWindowLayout(windowLayout));

        m_WindowLayoutInitialised[Index(windowLayout)] = true;
      }

      if (wasSelected)
      {
        /// If this viewer was selected before the window layout change, we select a window in the new layout.
        /// If the previously selected window is still visible, we do not do anything.
        /// Otherwise, we select the first visible window.
        std::vector<QmitkRenderWindow*> visibleRenderWindows = m_MultiWidget->GetVisibleRenderWindows();
        if (!visibleRenderWindows.empty()
            && std::find(visibleRenderWindows.begin(), visibleRenderWindows.end(), selectedRenderWindow) == visibleRenderWindows.end())
        {
          m_MultiWidget->SetSelectedRenderWindow(visibleRenderWindows[0]);
        }
      }
    }

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
  return this->MoveAnteriorPosterior(+1);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::MovePosterior()
{
  return this->MoveAnteriorPosterior(-1);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::MoveAnteriorPosterior(int slices)
{
  m_MultiWidget->MoveAnteriorOrPosterior(this->GetOrientation(), slices);
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

  while (m_LastSelectedPositions.size() > 1 && m_LastSelectedPositionTimes.back().msecsTo(currentTime) <= doubleClickInterval)
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

  this->SetWindowLayout(nextWindowLayout);
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
void niftkSingleViewerWidget::SetFocus()
{
  QmitkRenderWindow* renderWindow = this->GetSelectedRenderWindow();
  if (renderWindow)
  {
    renderWindow->setFocus();
  }
  else
  {
    this->setFocus();
  }
}
