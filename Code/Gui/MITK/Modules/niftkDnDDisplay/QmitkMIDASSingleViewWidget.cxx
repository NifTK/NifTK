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
#include <mitkMIDASOrientationUtils.h>
#include "QmitkMIDASSingleViewWidget.h"
#include "QmitkMIDASStdMultiWidget.h"


//-----------------------------------------------------------------------------
QmitkMIDASSingleViewWidget::QmitkMIDASSingleViewWidget(QWidget* parent)
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
, m_Layout(MIDAS_LAYOUT_UNKNOWN)
, m_Orientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberSettingsPerLayout(false)
, m_SingleWindowLayout(MIDAS_LAYOUT_CORONAL)
, m_MultiWindowLayout(MIDAS_LAYOUT_ORTHO)
, m_ViewKeyPressStateMachine(0)
{
  mitk::RenderingManager::Pointer renderingManager = mitk::RenderingManager::GetInstance();

  QString name("QmitkMIDASSingleViewWidget");
  this->Initialize(name, renderingManager, NULL);
}


//-----------------------------------------------------------------------------
QmitkMIDASSingleViewWidget::QmitkMIDASSingleViewWidget(
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
, m_Layout(MIDAS_LAYOUT_UNKNOWN)
, m_Orientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberSettingsPerLayout(false)
, m_SingleWindowLayout(MIDAS_LAYOUT_CORONAL)
, m_MultiWindowLayout(MIDAS_LAYOUT_ORTHO)
, m_ViewKeyPressStateMachine(0)
{
  this->Initialize(windowName, renderingManager, dataStorage);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::Initialize(QString windowName,
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

  for (int i = 0; i < MIDAS_ORIENTATION_NUMBER * 2; i++)
  {
    m_SliceIndexes[i] = 0;
    m_TimeSteps[i] = 0;
  }
  for (int i = 0; i < MIDAS_LAYOUT_NUMBER * 2; i++)
  {
    m_ScaleFactors[i] = 1.0;
    m_LayoutInitialised[i] = false;
  }

  // Create the main QmitkMIDASStdMultiWidget
  m_MultiWidget = new QmitkMIDASStdMultiWidget(this, NULL, m_RenderingManager, m_DataStorage);
  m_MultiWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->SetNavigationControllerEventListening(false);

  m_GridLayout = new QGridLayout(this);
  m_GridLayout->setObjectName(QString::fromUtf8("QmitkMIDASSingleViewWidget::m_GridLayout"));
  m_GridLayout->setContentsMargins(1, 1, 1, 1);
  m_GridLayout->setVerticalSpacing(0);
  m_GridLayout->setHorizontalSpacing(0);
  m_GridLayout->addWidget(m_MultiWidget);

  // Connect to QmitkMIDASStdMultiWidget, so we can listen for signals.
  QObject::connect(m_MultiWidget, SIGNAL(NodesDropped(QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  QObject::connect(m_MultiWidget, SIGNAL(SelectedPositionChanged(QmitkRenderWindow*, int)), this, SLOT(OnSelectedPositionChanged(QmitkRenderWindow*, int)));
  QObject::connect(m_MultiWidget, SIGNAL(CursorPositionChanged(const mitk::Vector3D&)), this, SLOT(OnCursorPositionChanged(const mitk::Vector3D&)));
  QObject::connect(m_MultiWidget, SIGNAL(ScaleFactorChanged(double)), this, SLOT(OnScaleFactorChanged(double)));

  // Create/Connect the state machine
  m_ViewKeyPressStateMachine = mitk::MIDASViewKeyPressStateMachine::New("MIDASViewKeyPressStateMachine", this);
  std::vector<QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  for (unsigned j = 0; j < renderWindows.size(); ++j)
  {
    m_ViewKeyPressStateMachine->AddRenderer(renderWindows[j]->GetRenderer());
  }
  mitk::GlobalInteraction::GetInstance()->AddListener(m_ViewKeyPressStateMachine);
}


//-----------------------------------------------------------------------------
QmitkMIDASSingleViewWidget::~QmitkMIDASSingleViewWidget()
{
  mitk::GlobalInteraction::GetInstance()->RemoveListener(m_ViewKeyPressStateMachine);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnNodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  // Try not to emit the QmitkMIDASStdMultiWidget pointer.
  emit NodesDropped(window, nodes);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnSelectedPositionChanged(QmitkRenderWindow *window, int sliceIndex)
{
  mitk::Point3D selectedPosition = this->GetSelectedPosition();
  if (selectedPosition != m_SelectedPosition)
  {
    m_SecondLastSelectedPosition = m_LastSelectedPosition;
    m_LastSelectedPosition = m_SelectedPosition;
    m_SelectedPosition = selectedPosition;
  }
  emit SelectedPositionChanged(this, window, sliceIndex);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnCursorPositionChanged(const mitk::Vector3D& cursorPosition)
{
  if (cursorPosition != m_CursorPosition)
  {
    m_LastCursorPosition = m_CursorPosition;
    m_CursorPosition = cursorPosition;
  }
  emit CursorPositionChanged(this, cursorPosition);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnScaleFactorChanged(double scaleFactor)
{
  emit ScaleFactorChanged(this, scaleFactor);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetSelected(bool selected)
{
  m_MultiWidget->SetSelected(selected);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::IsSelected() const
{
  return m_MultiWidget->IsSelected();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetSelectedRenderWindow() const
{
  return m_MultiWidget->GetSelectedRenderWindow();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetSelectedRenderWindow(QmitkRenderWindow* renderWindow)
{
  m_MultiWidget->SetSelectedRenderWindow(renderWindow);
}


//-----------------------------------------------------------------------------
std::vector<QmitkRenderWindow*> QmitkMIDASSingleViewWidget::GetVisibleRenderWindows() const
{
  return m_MultiWidget->GetVisibleRenderWindows();
}


//-----------------------------------------------------------------------------
std::vector<QmitkRenderWindow*> QmitkMIDASSingleViewWidget::GetRenderWindows() const
{
  return m_MultiWidget->GetRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetAxialWindow() const
{
  return m_MultiWidget->GetRenderWindow1();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetSagittalWindow() const
{
  return m_MultiWidget->GetRenderWindow2();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetCoronalWindow() const
{
  return m_MultiWidget->GetRenderWindow3();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASSingleViewWidget::Get3DWindow() const
{
  return m_MultiWidget->GetRenderWindow4();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetEnabled(bool enabled)
{
  m_MultiWidget->SetEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::IsEnabled() const
{
  return m_MultiWidget->IsEnabled();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetDisplay2DCursorsLocally(bool visible)
{
  m_MultiWidget->SetDisplay2DCursorsLocally(visible);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::GetDisplay2DCursorsLocally() const
{
  return m_MultiWidget->GetDisplay2DCursorsLocally();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetDisplay2DCursorsGlobally(bool visible)
{
  m_MultiWidget->SetDisplay2DCursorsGlobally(visible);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::GetDisplay2DCursorsGlobally() const
{
  return m_MultiWidget->GetDisplay2DCursorsGlobally();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::AreDirectionAnnotationsVisible() const
{
  return m_MultiWidget->AreDirectionAnnotationsVisible();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetDirectionAnnotationsVisible(bool visible)
{
  m_MultiWidget->SetDirectionAnnotationsVisible(visible);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::GetShow3DWindowInOrthoView() const
{
  return m_MultiWidget->GetShow3DWindowInOrthoView();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetShow3DWindowInOrthoView(bool enabled)
{
  m_MultiWidget->SetShow3DWindowInOrthoView(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetBackgroundColor(QColor color)
{
  m_MultiWidget->SetBackgroundColor(color);
}


//-----------------------------------------------------------------------------
QColor QmitkMIDASSingleViewWidget::GetBackgroundColor() const
{
  return m_MultiWidget->GetBackgroundColor();
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetMaxSliceIndex(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMaxSliceIndex(orientation);
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetMaxTimeStep() const
{
  return m_MultiWidget->GetMaxTimeStep();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::ContainsRenderWindow(QmitkRenderWindow *renderWindow) const
{
  return m_MultiWidget->ContainsRenderWindow(renderWindow);
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetRenderWindow(vtkRenderWindow *aVtkRenderWindow) const
{
  return m_MultiWidget->GetRenderWindow(aVtkRenderWindow);
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASSingleViewWidget::GetOrientation()
{
  return m_MultiWidget->GetOrientation();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::FitToDisplay()
{
  m_MultiWidget->FitToDisplay();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible)
{
  m_MultiWidget->SetRendererSpecificVisibility(nodes, visible);
}


//-----------------------------------------------------------------------------
double QmitkMIDASSingleViewWidget::GetMinMagnification() const
{
  return m_MinimumMagnification;
}


//-----------------------------------------------------------------------------
double QmitkMIDASSingleViewWidget::GetMaxMagnification() const
{
  return m_MaximumMagnification;
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer QmitkMIDASSingleViewWidget::GetDataStorage() const
{
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetRememberSettingsPerLayout(bool remember)
{
  m_RememberSettingsPerLayout = remember;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::GetRememberSettingsPerLayout() const
{
  return m_RememberSettingsPerLayout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  m_MultiWidget->SetDataStorage(m_DataStorage);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetNavigationControllerEventListening(bool enabled)
{
  if (enabled)
  {
    m_MultiWidget->EnableNavigationControllerEventListening();
    m_MultiWidget->SetWidgetPlanesLocked(false);
  }
  else
  {
    m_MultiWidget->DisableNavigationControllerEventListening();
    m_MultiWidget->SetWidgetPlanesLocked(true);
  }
  m_NavigationControllerEventListening = enabled;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::GetNavigationControllerEventListening() const
{
  return m_NavigationControllerEventListening;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetDisplayInteractionsEnabled(bool enabled)
{
  m_MultiWidget->SetDisplayInteractionsEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::AreDisplayInteractionsEnabled() const
{
  return m_MultiWidget->AreDisplayInteractionsEnabled();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::AreCursorPositionsBound() const
{
  return m_MultiWidget->AreCursorPositionsBound();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetCursorPositionsBound(bool bound)
{
  m_MultiWidget->SetCursorPositionsBound(bound);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::AreScaleFactorsBound() const
{
  return m_MultiWidget->AreScaleFactorsBound();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetScaleFactorsBound(bool bound)
{
  m_MultiWidget->SetScaleFactorsBound(bound);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::RequestUpdate()
{
  m_MultiWidget->RequestUpdate();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::ResetRememberedPositions()
{
  for (int i = 0; i < MIDAS_ORIENTATION_NUMBER; i++)
  {
    m_SliceIndexes[Index(i)] = 0;
    m_TimeSteps[Index(i)] = 0;
  }
  for (int i = 0; i < MIDAS_LAYOUT_NUMBER; i++)
  {
    m_ScaleFactors[Index(i)] = 1.0;
    m_LayoutInitialised[Index(i)] = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetGeometry(mitk::Geometry3D::Pointer geometry)
{
  assert(geometry);
  m_Geometry = geometry;

  if (!m_IsBoundGeometryActive)
  {
    m_MultiWidget->SetGeometry(geometry);

    this->ResetRememberedPositions();

    m_SelectedPosition = m_MultiWidget->GetSelectedPosition();
    m_LastSelectedPosition = m_SelectedPosition;
    m_SecondLastSelectedPosition = m_SelectedPosition;
    m_CursorPosition = m_MultiWidget->GetCursorPosition();
    m_LastCursorPosition = m_CursorPosition;
  }

  emit GeometryChanged(this, geometry);
}


//-----------------------------------------------------------------------------
mitk::Geometry3D::Pointer QmitkMIDASSingleViewWidget::GetGeometry()
{
  assert(m_Geometry);
  return m_Geometry;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetBoundGeometry(mitk::Geometry3D::Pointer geometry)
{
  assert(geometry);
  m_BoundGeometry = geometry;

  if (m_IsBoundGeometryActive)
  {
    m_MultiWidget->SetGeometry(geometry);

    this->ResetRememberedPositions();

    m_SelectedPosition = m_MultiWidget->GetSelectedPosition();
    m_LastSelectedPosition = m_SelectedPosition;
    m_SecondLastSelectedPosition = m_SelectedPosition;
    m_CursorPosition = m_MultiWidget->GetCursorPosition();
    m_LastCursorPosition = m_CursorPosition;
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::IsBoundGeometryActive()
{
  return m_IsBoundGeometryActive;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetBoundGeometryActive(bool isBoundGeometryActive)
{
  if (isBoundGeometryActive == m_IsBoundGeometryActive)
  {
    // No change, nothing to do.
    return;
  }

  mitk::Geometry3D* geometry = isBoundGeometryActive ? m_BoundGeometry : m_Geometry;
  m_MultiWidget->SetGeometry(geometry);

  m_IsBoundGeometryActive = isBoundGeometryActive;
  //  m_Layout = MIDAS_LAYOUT_UNKNOWN;
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetSliceIndex(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetSliceIndex(orientation);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetSliceIndex(MIDASOrientation orientation, unsigned int sliceIndex)
{
  m_SliceIndexes[Index(m_Orientation)] = sliceIndex;
  if (m_Orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_MultiWidget->SetSliceIndex(orientation, sliceIndex);
  }
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetTimeStep() const
{
  return m_MultiWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetTimeStep(unsigned int timeStep)
{
  m_TimeSteps[Index(m_Orientation)] = timeStep;
  if (m_Orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_MultiWidget->SetTimeStep(timeStep);
  }
}


//-----------------------------------------------------------------------------
MIDASLayout QmitkMIDASSingleViewWidget::GetLayout() const
{
  return m_Layout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetLayout(MIDASLayout layout)
{
  if (layout != MIDAS_LAYOUT_UNKNOWN)
  {
    mitk::Geometry3D* geometry = m_IsBoundGeometryActive ? m_BoundGeometry : m_Geometry;

    // If for whatever reason, we have no geometry... bail out.
    if (!geometry)
    {
      return;
    }

    // If we have a currently valid layout/orientation, then store the current position, so we can switch back to it if necessary.
    m_SliceIndexes[Index(m_Orientation)] = this->GetSliceIndex(m_Orientation);
    m_TimeSteps[Index(m_Orientation)] = this->GetTimeStep();
    m_ScaleFactors[Index(m_Layout)] = m_MultiWidget->GetScaleFactor();
    m_LayoutInitialised[Index(m_Layout)] = true;

    // Store the currently selected position because the SetGeometry call resets it to the origin.
    mitk::Point3D selectedPosition = this->GetSelectedPosition();

    // This will initialise the whole QmitkStdMultiWidget according to the supplied geometry (normally an image).

    m_MultiWidget->SetGeometry(geometry);
    m_MultiWidget->SetLayout(layout);
    // Call Qt update to try and make sure we are painted at the right size.
    m_MultiWidget->update();

    // Restore the selected position if it was set before.
    if (selectedPosition[0] != 0.0 || selectedPosition[1] != 0.0 || selectedPosition[2] != 0.0)
    {
      m_MultiWidget->SetSelectedPosition(selectedPosition);
    }

    // Now store the current layout/orientation.
    MIDASOrientation orientation = this->GetOrientation();
    m_Orientation = orientation;
    m_Layout = layout;

    // Now, in MIDAS, which only shows 2D views, if we revert to a previous view,
    // we should go back to the same slice index, time step, cursor position on display, scale factor.
    bool hasBeenInitialised = m_LayoutInitialised[Index(layout)];
    if (m_RememberSettingsPerLayout && hasBeenInitialised)
    {
      if (orientation != MIDAS_ORIENTATION_UNKNOWN)
      {
        int sliceIndex = m_SliceIndexes[Index(orientation)];
        this->SetSliceIndex(orientation, sliceIndex);
        this->SetTimeStep(m_TimeSteps[Index(orientation)]);

        QmitkRenderWindow* renderWindow = m_MultiWidget->GetRenderWindow(orientation);
        emit SelectedPositionChanged(this, renderWindow, sliceIndex);
      }

      double scaleFactor = m_ScaleFactors[Index(layout)];
      this->SetScaleFactor(scaleFactor);
      emit ScaleFactorChanged(this, scaleFactor);
    }
    else
    {
      if (orientation == MIDAS_ORIENTATION_UNKNOWN)
      {
        orientation = MIDAS_ORIENTATION_AXIAL; // somewhat arbitrary.
      }

      unsigned int sliceIndex = this->GetSliceIndex(orientation);
      unsigned int timeStep = this->GetTimeStep();

      this->SetSliceIndex(orientation, sliceIndex);
      this->SetTimeStep(timeStep);
      QmitkRenderWindow* renderWindow = m_MultiWidget->GetRenderWindow(orientation);
      emit SelectedPositionChanged(this, renderWindow, sliceIndex);

      if (!hasBeenInitialised)
      {
        m_MultiWidget->FitToDisplay();
        hasBeenInitialised = true;
      }

      double magnification = m_MultiWidget->GetMagnification(renderWindow);
      this->SetMagnification(magnification);
      double scaleFactor = m_MultiWidget->GetScaleFactor();
//      this->SetScaleFactor(scaleFactor);
      m_LayoutInitialised[Index(layout)] = true;
      emit ScaleFactorChanged(this, scaleFactor);
    }
  }
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkMIDASSingleViewWidget::GetSelectedPosition() const
{
  return m_MultiWidget->GetSelectedPosition();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition)
{
  if (m_Layout != MIDAS_LAYOUT_UNKNOWN)
  {
    m_SelectedPosition = selectedPosition;
    m_LastSelectedPosition = selectedPosition;
    m_SecondLastSelectedPosition = selectedPosition;
    m_MultiWidget->SetSelectedPosition(selectedPosition);
  }
}


//-----------------------------------------------------------------------------
const mitk::Vector3D& QmitkMIDASSingleViewWidget::GetCursorPosition() const
{
  return m_MultiWidget->GetCursorPosition();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetCursorPosition(const mitk::Vector3D& cursorPosition)
{
  if (m_Layout != MIDAS_LAYOUT_UNKNOWN)
  {
    m_CursorPosition = cursorPosition;
    m_LastCursorPosition = cursorPosition;
    m_MultiWidget->SetCursorPosition(cursorPosition);
  }
}


//-----------------------------------------------------------------------------
double QmitkMIDASSingleViewWidget::GetMagnification() const
{
  return m_MultiWidget->GetMagnification();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetMagnification(double magnification)
{
  if (m_Layout != MIDAS_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetMagnification(magnification);
    m_ScaleFactors[Index(m_Layout)] = m_MultiWidget->GetScaleFactor();
  }
}


//-----------------------------------------------------------------------------
double QmitkMIDASSingleViewWidget::GetScaleFactor() const
{
  return m_MultiWidget->GetScaleFactor();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetScaleFactor(double scaleFactor)
{
  if (m_Layout != MIDAS_LAYOUT_UNKNOWN)
  {
    m_MultiWidget->SetScaleFactor(scaleFactor);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::paintEvent(QPaintEvent *event)
{
  QWidget::paintEvent(event);
  std::vector<QmitkRenderWindow*> renderWindows = this->GetVisibleRenderWindows();
  for (unsigned i = 0; i < renderWindows.size(); i++)
  {
    renderWindows[i]->GetVtkRenderWindow()->Render();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::InitializeStandardViews(const mitk::Geometry3D * geometry)
{
  m_MultiWidget->InitializeStandardViews(geometry);
}


//-----------------------------------------------------------------------------
std::vector<mitk::DataNode*> QmitkMIDASSingleViewWidget::GetWidgetPlanes()
{
  std::vector<mitk::DataNode*> result;
  result.push_back(m_MultiWidget->GetWidgetPlane1());
  result.push_back(m_MultiWidget->GetWidgetPlane2());
  result.push_back(m_MultiWidget->GetWidgetPlane3());
  return result;
}


//-----------------------------------------------------------------------------
int QmitkMIDASSingleViewWidget::GetSliceUpDirection(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetSliceUpDirection(orientation);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetDefaultSingleWindowLayout(MIDASLayout layout)
{
  m_SingleWindowLayout = layout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetDefaultMultiWindowLayout(MIDASLayout layout)
{
  m_MultiWindowLayout = layout;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::MoveAnterior()
{
  return this->MoveAnteriorPosterior(1);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::MovePosterior()
{
  return this->MoveAnteriorPosterior(-1);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::MoveAnteriorPosterior(int slices)
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
      actuallyDidSomething = true;
      emit SelectedPositionChanged(this, m_MultiWidget->GetRenderWindow(orientation), nextSliceIndex);
    }
  }

  return actuallyDidSomething;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::SwitchToAxial()
{
  this->SetLayout(MIDAS_LAYOUT_AXIAL);
  emit LayoutChanged(this, MIDAS_LAYOUT_AXIAL);
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::SwitchToSagittal()
{
  this->SetLayout(MIDAS_LAYOUT_SAGITTAL);
  emit LayoutChanged(this, MIDAS_LAYOUT_SAGITTAL);
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::SwitchToCoronal()
{
  this->SetLayout(MIDAS_LAYOUT_CORONAL);
  emit LayoutChanged(this, MIDAS_LAYOUT_CORONAL);
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::SwitchTo3D()
{
  this->SetLayout(MIDAS_LAYOUT_3D);
  emit LayoutChanged(this, MIDAS_LAYOUT_3D);
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::ToggleMultiWindowLayout()
{
  MIDASLayout nextLayout;

  if (::IsSingleWindowLayout(m_Layout))
  {
    nextLayout = m_MultiWindowLayout;
  }
  else
  {
    switch (this->GetOrientation())
    {
    case MIDAS_ORIENTATION_AXIAL:
      nextLayout = MIDAS_LAYOUT_AXIAL;
      break;
    case MIDAS_ORIENTATION_SAGITTAL:
      nextLayout = MIDAS_LAYOUT_SAGITTAL;
      break;
    case MIDAS_ORIENTATION_CORONAL:
      nextLayout = MIDAS_LAYOUT_CORONAL;
      break;
    case MIDAS_ORIENTATION_UNKNOWN:
      nextLayout = MIDAS_LAYOUT_3D;
      break;
    default:
      nextLayout = MIDAS_LAYOUT_CORONAL;
    }
  }

  // We have to switch back to the previous position because the double click should not change
  // neither the selected position nor the cursor position.
  this->SetSelectedPosition(m_SecondLastSelectedPosition);
  this->SetCursorPosition(m_LastCursorPosition);
//  m_MultiWidget->SetCursorPosition(m_LastCursorPosition);
//  m_MultiWidget->SetSelectedPosition(m_SecondLastSelectedPosition);

  this->SetLayout(nextLayout);
  emit LayoutChanged(this, nextLayout);

  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::ToggleCursor()
{
//  this->SetShow2DCursors(!this->GetShow2DCursors());

  return true;
}
