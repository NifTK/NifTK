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
niftkSingleViewerWidget::niftkSingleViewerWidget(QWidget* parent)
: QWidget(parent)
, m_DataStorage(NULL)
, m_RenderingManager(NULL)
, m_GridLayout(NULL)
, m_MultiWidget(NULL)
, m_IsBoundGeometryActive(false)
, m_Geometry(NULL)
, m_BoundGeometry(NULL)
, m_MinimumMagnification(-5.0)
, m_MaximumMagnification(20.0)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_Orientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberSettingsPerWindowLayout(false)
, m_SingleWindowLayout(WINDOW_LAYOUT_CORONAL)
, m_MultiWindowLayout(WINDOW_LAYOUT_ORTHO)
, m_DnDDisplayStateMachine(0)
{
  mitk::RenderingManager::Pointer renderingManager = mitk::RenderingManager::GetInstance();

  QString name("niftkSingleViewerWidget");
  this->Initialize(name, renderingManager, NULL);
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidget::niftkSingleViewerWidget(
    QString windowName,
    double minimumMagnification,
    double maximumMagnification,
    QWidget *parent,
    mitk::RenderingManager* renderingManager,
    mitk::DataStorage* dataStorage)
: QWidget(parent)
, m_DataStorage(NULL)
, m_RenderingManager(NULL)
, m_GridLayout(NULL)
, m_MultiWidget(NULL)
, m_IsBoundGeometryActive(false)
, m_Geometry(NULL)
, m_BoundGeometry(NULL)
, m_MinimumMagnification(minimumMagnification)
, m_MaximumMagnification(maximumMagnification)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_Orientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberSettingsPerWindowLayout(false)
, m_SingleWindowLayout(WINDOW_LAYOUT_CORONAL)
, m_MultiWindowLayout(WINDOW_LAYOUT_ORTHO)
, m_DnDDisplayStateMachine(0)
{
  this->Initialize(windowName, renderingManager, dataStorage);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::Initialize(QString windowName,
                mitk::RenderingManager* renderingManager,
                mitk::DataStorage* dataStorage
               )
{
  if (renderingManager == NULL)
  {
    m_RenderingManager = mitk::RenderingManager::GetInstance();
  }
  else
  {
    m_RenderingManager = renderingManager;
  }

  m_DataStorage = dataStorage;

  this->setAcceptDrops(true);

  for (int windowLayoutIndex = 0; windowLayoutIndex < WINDOW_LAYOUT_NUMBER * 2; windowLayoutIndex++)
  {
    m_WindowLayoutInitialised[windowLayoutIndex] = false;
  }

  // Create the main niftkMultiWindowWidget
  m_MultiWidget = new niftkMultiWindowWidget(this, NULL, m_RenderingManager, m_DataStorage);
  m_MultiWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->SetNavigationControllerEventListening(false);

  m_GridLayout = new QGridLayout(this);
  m_GridLayout->setObjectName(QString::fromUtf8("niftkSingleViewerWidget::m_GridLayout"));
  m_GridLayout->setContentsMargins(1, 1, 1, 1);
  m_GridLayout->setVerticalSpacing(0);
  m_GridLayout->setHorizontalSpacing(0);
  m_GridLayout->addWidget(m_MultiWidget);

  // Connect to niftkMultiWindowWidget, so we can listen for signals.
  this->connect(m_MultiWidget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(m_MultiWidget, SIGNAL(SelectedPositionChanged(const mitk::Point3D&)), SLOT(OnSelectedPositionChanged(const mitk::Point3D&)));
  this->connect(m_MultiWidget, SIGNAL(CursorPositionChanged(MIDASOrientation, const mitk::Vector2D&)), SLOT(OnCursorPositionChanged(MIDASOrientation, const mitk::Vector2D&)));
  this->connect(m_MultiWidget, SIGNAL(ScaleFactorChanged(MIDASOrientation, double)), SLOT(OnScaleFactorChanged(MIDASOrientation, double)));

  // Create/Connect the state machine
  m_DnDDisplayStateMachine = mitk::DnDDisplayStateMachine::New("DnDDisplayStateMachine", this);
  std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  for (unsigned j = 0; j < renderWindows.size(); ++j)
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
void niftkSingleViewerWidget::OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  emit NodesDropped(window, nodes);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnSelectedPositionChanged(const mitk::Point3D& selectedPosition)
{
  if (m_LastSelectedPositions.size() == 3)
  {
    m_LastSelectedPositions.pop_front();
  }
  m_LastSelectedPositions.push_back(selectedPosition);

  if (m_LastCursorPositions.size() == 3)
  {
    m_LastCursorPositions.pop_front();
  }
  m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPosition(m_MultiWidget->GetOrientation()));

  emit SelectedPositionChanged(this, selectedPosition);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnCursorPositionChanged(MIDASOrientation orientation, const mitk::Vector2D& cursorPosition)
{
  if (m_LastCursorPositions.size() == 3)
  {
    m_LastCursorPositions.pop_front();
  }
  m_LastCursorPositions.push_back(cursorPosition);

  emit CursorPositionChanged(this, orientation, cursorPosition);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::OnScaleFactorChanged(MIDASOrientation orientation, double scaleFactor)
{
  emit ScaleFactorChanged(this, orientation, scaleFactor);
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
std::vector<QmitkRenderWindow*> niftkSingleViewerWidget::GetRenderWindows() const
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
void niftkSingleViewerWidget::SetBackgroundColor(QColor color)
{
  m_MultiWidget->SetBackgroundColor(color);
}


//-----------------------------------------------------------------------------
QColor niftkSingleViewerWidget::GetBackgroundColor() const
{
  return m_MultiWidget->GetBackgroundColor();
}


//-----------------------------------------------------------------------------
unsigned int niftkSingleViewerWidget::GetMaxSliceIndex(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMaxSliceIndex(orientation);
}


//-----------------------------------------------------------------------------
unsigned int niftkSingleViewerWidget::GetMaxTimeStep() const
{
  return m_MultiWidget->GetMaxTimeStep();
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::ContainsRenderWindow(QmitkRenderWindow *renderWindow) const
{
  return m_MultiWidget->ContainsRenderWindow(renderWindow);
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkSingleViewerWidget::GetRenderWindow(vtkRenderWindow *aVtkRenderWindow) const
{
  return m_MultiWidget->GetRenderWindow(aVtkRenderWindow);
}


//-----------------------------------------------------------------------------
MIDASOrientation niftkSingleViewerWidget::GetOrientation()
{
  return m_MultiWidget->GetOrientation();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::FitToDisplay()
{
  m_MultiWidget->FitToDisplay();
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
void niftkSingleViewerWidget::SetNavigationControllerEventListening(bool enabled)
{
  if (enabled)
  {
    m_MultiWidget->SetWidgetPlanesLocked(false);
  }
  else
  {
    m_MultiWidget->SetWidgetPlanesLocked(true);
  }
  m_NavigationControllerEventListening = enabled;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::GetNavigationControllerEventListening() const
{
  return m_NavigationControllerEventListening;
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
bool niftkSingleViewerWidget::AreCursorPositionsBound() const
{
  return m_MultiWidget->AreCursorPositionsBound();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetCursorPositionsBound(bool bound)
{
  m_MultiWidget->SetCursorPositionsBound(bound);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::AreScaleFactorsBound() const
{
  return m_MultiWidget->AreScaleFactorsBound();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetScaleFactorsBound(bool bound)
{
  m_MultiWidget->SetScaleFactorsBound(bound);
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
    m_MultiWidget->SetGeometry(timeGeometry);
    m_MultiWidget->FitToDisplay();

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      m_LastSelectedPositions.clear();
      m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
      m_LastCursorPositions.clear();
      m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPosition(m_MultiWidget->GetOrientation()));

      m_WindowLayoutInitialised[Index(m_WindowLayout)] = true;
    }

    for (int otherWindowLayout = 0; otherWindowLayout < WINDOW_LAYOUT_NUMBER; otherWindowLayout++)
    {
      if (otherWindowLayout != m_WindowLayout)
      {
        m_WindowLayoutInitialised[Index(otherWindowLayout)] = false;
      }
    }
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
void niftkSingleViewerWidget::SetBoundGeometry(mitk::TimeGeometry::Pointer geometry)
{
  assert(geometry);
  m_BoundGeometry = geometry;

  if (m_IsBoundGeometryActive)
  {
    m_MultiWidget->SetGeometry(geometry);
    m_MultiWidget->FitToDisplay();

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      m_LastSelectedPositions.clear();
      m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
      m_LastCursorPositions.clear();
      m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPosition(m_MultiWidget->GetOrientation()));

      m_WindowLayoutInitialised[Index(m_WindowLayout)] = true;
    }

    for (int otherWindowLayout = 0; otherWindowLayout < WINDOW_LAYOUT_NUMBER; otherWindowLayout++)
    {
      if (otherWindowLayout != m_WindowLayout)
      {
        m_WindowLayoutInitialised[Index(otherWindowLayout)] = false;
      }
    }
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

  mitk::TimeGeometry* geometry = isBoundGeometryActive ? m_BoundGeometry : m_Geometry;
  m_MultiWidget->SetGeometry(geometry);

  m_IsBoundGeometryActive = isBoundGeometryActive;
  //  m_WindowLayout = WINDOW_LAYOUT_UNKNOWN;
}


//-----------------------------------------------------------------------------
unsigned int niftkSingleViewerWidget::GetSliceIndex(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetSliceIndex(orientation);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetSliceIndex(MIDASOrientation orientation, unsigned int sliceIndex)
{
  if (m_Orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_MultiWidget->SetSliceIndex(orientation, sliceIndex);
  }
}


//-----------------------------------------------------------------------------
unsigned int niftkSingleViewerWidget::GetTimeStep() const
{
  return m_MultiWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidget::SetTimeStep(unsigned int timeStep)
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

    if (m_WindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      // If we have a currently valid window layout/orientation, then store the current position, so we can switch back to it if necessary.
      m_SelectedPositions[Index(m_WindowLayout)] = m_MultiWidget->GetSelectedPosition();
      m_TimeSteps[Index(0)] = m_MultiWidget->GetTimeStep();
      m_CursorPositions[Index(m_WindowLayout)] = m_MultiWidget->GetCursorPositions();
      m_ScaleFactors[Index(m_WindowLayout)] = m_MultiWidget->GetScaleFactors();
      m_SelectedRenderWindow[Index(m_WindowLayout)] = m_MultiWidget->GetSelectedRenderWindow();

      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) saving positions...";
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) window layout: " << m_WindowLayout;
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) selected point: " << m_SelectedPositions[Index(m_WindowLayout)];
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) cursor positions: " << m_CursorPositions[Index(m_WindowLayout)][0];
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) scale factors: " << m_ScaleFactors[Index(m_WindowLayout)][0];
    }

    // This will initialise the whole QmitkStdMultiWidget according to the supplied geometry (normally an image).
    m_MultiWidget->SetGeometry(geometry);
    m_MultiWidget->SetWindowLayout(windowLayout);
    // Call Qt update to try and make sure we are painted at the right size.
    m_MultiWidget->update();

    // Now store the current window layout/orientation.
    MIDASOrientation orientation = this->GetOrientation();
    m_Orientation = orientation;
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
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) restoring positions...";
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) window layout: " << m_WindowLayout;
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) selected point: " << m_SelectedPositions[Index(m_WindowLayout)];
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) cursor position: " << m_CursorPositions[Index(m_WindowLayout)][0];
      MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) scale factor: " << m_ScaleFactors[Index(m_WindowLayout)][0];

      m_MultiWidget->SetSelectedPosition(m_SelectedPositions[Index(windowLayout)]);
      m_MultiWidget->SetTimeStep(m_TimeSteps[Index(0)]);

      m_MultiWidget->SetCursorPositions(m_CursorPositions[Index(windowLayout)]);
      m_MultiWidget->SetScaleFactors(m_ScaleFactors[Index(windowLayout)]);

      m_MultiWidget->SetSelectedRenderWindow(m_SelectedRenderWindow[Index(windowLayout)]);

      m_LastSelectedPositions.clear();
      m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
      m_LastCursorPositions.clear();
      m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPosition(m_MultiWidget->GetOrientation()));

      emit SelectedPositionChanged(this, m_SelectedPositions[Index(windowLayout)]);
      emit CursorPositionChanged(this, orientation, m_CursorPositions[Index(windowLayout)][orientation]);
      emit ScaleFactorChanged(this, orientation, m_ScaleFactors[Index(windowLayout)][orientation]);
    }
    else
    {
      if (orientation == MIDAS_ORIENTATION_UNKNOWN)
      {
        orientation = MIDAS_ORIENTATION_AXIAL; // somewhat arbitrary.
      }

      if (!hasBeenInitialised)
      {
        m_MultiWidget->SetSelectedPosition(geometry->GetCenterInWorld());
        m_MultiWidget->SetTimeStep(0);
        m_MultiWidget->FitToDisplay();

        m_LastSelectedPositions.clear();
        m_LastSelectedPositions.push_back(m_MultiWidget->GetSelectedPosition());
        m_LastCursorPositions.clear();
        m_LastCursorPositions.push_back(m_MultiWidget->GetCursorPosition(m_MultiWidget->GetOrientation()));

        /// If this is a multi window layout and the scale factors are bound,
        /// we have to set the same scale factors for each window.
        if (m_MultiWindowLayout == m_WindowLayout && this->AreScaleFactorsBound())
        {
          m_MultiWidget->SetScaleFactor(m_MultiWidget->GetScaleFactor());
        }

        MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) selected point: " << m_MultiWidget->GetSelectedPosition();
        MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) cursor position: " << m_MultiWidget->GetCursorPosition(m_MultiWidget->GetOrientation());
        MITK_INFO << "niftkSingleViewerWidget::SetWindowLayout(WindowLayout windowLayout) scale factor: " << m_MultiWidget->GetScaleFactor();
        m_WindowLayoutInitialised[Index(windowLayout)] = true;
      }
    }
  }
}


//-----------------------------------------------------------------------------
mitk::Point3D niftkSingleViewerWidget::GetSelectedPosition() const
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
void niftkSingleViewerWidget::paintEvent(QPaintEvent *event)
{
  QWidget::paintEvent(event);
  std::vector<QmitkRenderWindow*> renderWindows = this->GetVisibleRenderWindows();
  for (unsigned i = 0; i < renderWindows.size(); i++)
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
  return this->MoveAnteriorPosterior(1);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::MovePosterior()
{
  return this->MoveAnteriorPosterior(-1);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidget::MoveAnteriorPosterior(int slices)
{
  bool actuallyDidSomething = false;

  MIDASOrientation orientation = this->GetOrientation();
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    unsigned int sliceIndex = this->GetSliceIndex(orientation);
    int upDirection = this->GetSliceUpDirection(orientation);

    int nextSliceIndex = sliceIndex + slices * upDirection;

    unsigned int maxSliceIndex = this->GetMaxSliceIndex(orientation);

    if (nextSliceIndex >= 0 && nextSliceIndex <= static_cast<int>(maxSliceIndex))
    {
      this->SetSliceIndex(orientation, nextSliceIndex);

      /// Note. As a request and for MIDAS compatibility, all the slice have to be forcibly rendered
      /// when scrolling through them by keeping the 'a' or 'z' key pressed.
      /// Otherwise, issues on the scan or in the segmentation may be not seen.
      m_RenderingManager->ForceImmediateUpdate(m_MultiWidget->GetRenderWindow(orientation)->GetRenderWindow());

      actuallyDidSomething = true;
      emit SelectedPositionChanged(this, this->GetSelectedPosition());
    }
  }

  return actuallyDidSomething;
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

  /// We have to switch back to the actual positions before the double clicking,
  /// because the double click should not change neither the selected position
  /// nor the cursor position.
  this->SetSelectedPosition(m_LastSelectedPositions.front());
  this->SetCursorPosition(this->GetOrientation(), m_LastCursorPositions.front());

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
