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

#include "itkConversionUtils.h"
#include "mitkPointUtils.h"
#include "mitkMIDASOrientationUtils.h"
#include "QmitkMIDASSingleViewWidget.h"
#include "QmitkMIDASStdMultiWidget.h"


//-----------------------------------------------------------------------------
QmitkMIDASSingleViewWidget::QmitkMIDASSingleViewWidget(QWidget *parent)
: QWidget(parent)
, m_DataStorage(NULL)
, m_RenderingManager(NULL)
, m_Layout(NULL)
, m_MultiWidget(NULL)
, m_IsBound(false)
, m_UnBoundGeometry(NULL)
, m_BoundGeometry(NULL)
, m_ActiveGeometry(NULL)
, m_MinimumMagnification(-5.0)
, m_MaximumMagnification(20.0)
, m_View(MIDAS_VIEW_UNKNOWN)
, m_Orientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberViewSettingsPerOrientation(false)
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
    mitk::DataStorage* dataStorage
    )
  : QWidget(parent)
, m_DataStorage(NULL)
, m_RenderingManager(NULL)
, m_Layout(NULL)
, m_MultiWidget(NULL)
, m_IsBound(false)
, m_UnBoundGeometry(NULL)
, m_BoundGeometry(NULL)
, m_ActiveGeometry(NULL)
, m_MinimumMagnification(minimumMagnification)
, m_MaximumMagnification(maximumMagnification)
, m_View(MIDAS_VIEW_UNKNOWN)
, m_Orientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberViewSettingsPerOrientation(false)
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
    m_SliceNumbers[i] = 0;
    m_TimeSliceNumbers[i] = 0;
  }
  for (int i = 0; i < MIDAS_VIEW_NUMBER * 2; i++)
  {
    for (int j = 0; j < 3; ++j)
    {
      m_CursorPositions[i][j] = 0.5;
    }
    m_Magnifications[i] = m_MinimumMagnification;
    m_ViewInitialised[i] = false;
  }

  // Create the main QmitkMIDASStdMultiWidget
  m_MultiWidget = new QmitkMIDASStdMultiWidget(this, NULL, m_RenderingManager, m_DataStorage);
  m_MultiWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  this->SetNavigationControllerEventListening(false);

  m_Layout = new QGridLayout(this);
  m_Layout->setObjectName(QString::fromUtf8("QmitkMIDASSingleViewWidget::m_Layout"));
  m_Layout->setContentsMargins(1, 1, 1, 1);
  m_Layout->setVerticalSpacing(0);
  m_Layout->setHorizontalSpacing(0);
  m_Layout->addWidget(m_MultiWidget);

  // Connect to QmitkMIDASStdMultiWidget, so we can listen for signals.
  connect(m_MultiWidget, SIGNAL(NodesDropped(QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(m_MultiWidget, SIGNAL(SelectedPositionChanged(QmitkRenderWindow*, int)), this, SLOT(OnSelectedPositionChanged(QmitkRenderWindow*, int)));
  connect(m_MultiWidget, SIGNAL(CursorPositionChanged(const mitk::Vector3D&)), this, SLOT(OnCursorPositionChanged(const mitk::Vector3D&)));
  connect(m_MultiWidget, SIGNAL(MagnificationChanged(double)), this, SLOT(OnMagnificationChanged(double)));
}


//-----------------------------------------------------------------------------
QmitkMIDASSingleViewWidget::~QmitkMIDASSingleViewWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnNodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  // Try not to emit the QmitkMIDASStdMultiWidget pointer.
  emit NodesDropped(window, nodes);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnSelectedPositionChanged(QmitkRenderWindow *window, int sliceNumber)
{
  emit SelectedPositionChanged(this, window, sliceNumber);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnCursorPositionChanged(const mitk::Vector3D& cursorPosition)
{
  emit CursorPositionChanged(this, cursorPosition);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::OnMagnificationChanged(double magnification)
{
  emit MagnificationChanged(this, magnification);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::IsSingle2DView() const
{
  return m_MultiWidget->IsSingle2DView();
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
unsigned int QmitkMIDASSingleViewWidget::GetMinSlice(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMinSlice(orientation);
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetMaxSlice(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMaxSlice(orientation);
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetMinTime() const
{
  return m_MultiWidget->GetMinTime();
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetMaxTime() const
{
  return m_MultiWidget->GetMaxTime();
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
void QmitkMIDASSingleViewWidget::SetRememberViewSettingsPerOrientation(bool remember)
{
  m_RememberViewSettingsPerOrientation = remember;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::GetRememberViewSettingsPerOrientation() const
{
  return m_RememberViewSettingsPerOrientation;
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
void QmitkMIDASSingleViewWidget::SetDisplayInteractionEnabled(bool enabled)
{
  m_MultiWidget->SetDisplayInteractionEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::IsDisplayInteractionEnabled() const
{
  return m_MultiWidget->IsDisplayInteractionEnabled();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::RequestUpdate()
{
  m_MultiWidget->RequestUpdate();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::StorePosition()
{
  MIDASView view = m_View;
  MIDASOrientation orientation = m_Orientation;

  m_SliceNumbers[Index(orientation)] = this->GetSliceNumber(orientation);
  m_TimeSliceNumbers[Index(orientation)] = this->GetTime();
  m_CursorPositions[Index(view)] = m_MultiWidget->GetCursorPosition();
  m_Magnifications[Index(view)] = m_MultiWidget->GetMagnification();
  m_ViewInitialised[Index(view)] = true;

  MITK_DEBUG << "QmitkMIDASSingleViewWidget::StorePosition is bound=" << m_IsBound \
      << ", current orientation=" << orientation \
      << ", view=" << view \
      << ", so storing slice=" << this->GetSliceNumber(orientation) \
      << ", time=" << this->GetTime() \
      << ", magnification=" << m_MultiWidget->GetMagnification() << std::endl;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::ResetCurrentPosition()
{
  m_SliceNumbers[Index(m_Orientation)] = 0;
  m_TimeSliceNumbers[Index(m_Orientation)] = 0;
  for (int j = 0; j < 3; ++j)
  {
    m_CursorPositions[Index(m_View)][j] = 0.5;
  }
  m_Magnifications[Index(m_View)] = m_MinimumMagnification;
  m_ViewInitialised[Index(m_View)] = false;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::ResetRememberedPositions()
{
  for (int i = 0; i < MIDAS_ORIENTATION_NUMBER; i++)
  {
    m_SliceNumbers[Index(i)] = 0;
    m_TimeSliceNumbers[Index(i)] = 0;
  }
  for (int i = 0; i < MIDAS_VIEW_NUMBER; i++)
  {
    for (int j = 0; j < 3; ++j)
    {
      m_CursorPositions[Index(i)][j] = 0.5;
    }
    m_Magnifications[Index(i)] = m_MinimumMagnification;
    m_ViewInitialised[Index(i)] = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetGeometry(mitk::Geometry3D::Pointer geometry)
{
  assert(geometry);
  m_UnBoundGeometry = geometry;

  this->ResetRememberedPositions();
  this->ResetCurrentPosition();
}


//-----------------------------------------------------------------------------
mitk::Geometry3D::Pointer QmitkMIDASSingleViewWidget::GetGeometry()
{
  assert(m_UnBoundGeometry);
  return m_UnBoundGeometry;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetBoundGeometry(mitk::Geometry3D::Pointer geometry)
{
  assert(geometry);
  m_BoundGeometry = geometry;

  this->ResetRememberedPositions();
  this->ResetCurrentPosition();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSingleViewWidget::GetBoundGeometryActive()
{
  return m_IsBound;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetBoundGeometryActive(bool isBound)
{
  if (isBound == m_IsBound)
  {
    // No change, nothing to do.
    return;
  }

  m_IsBound = isBound;
  m_View = MIDAS_VIEW_UNKNOWN;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetActiveGeometry()
{
  if (m_IsBound)
  {
    m_ActiveGeometry = m_BoundGeometry;
  }
  else
  {
    m_ActiveGeometry = m_UnBoundGeometry;
  }
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetSliceNumber(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetSliceNumber(orientation);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetSliceNumber(MIDASOrientation orientation, unsigned int sliceNumber)
{
  m_SliceNumbers[Index(m_Orientation)] = sliceNumber;
  if (m_Orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_MultiWidget->SetSliceNumber(orientation, sliceNumber);
  }
}


//-----------------------------------------------------------------------------
unsigned int QmitkMIDASSingleViewWidget::GetTime() const
{
  return m_MultiWidget->GetTime();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetTime(unsigned int timeSliceNumber)
{
  m_TimeSliceNumbers[Index(m_Orientation)] = timeSliceNumber;
  if (m_Orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_MultiWidget->SetTime(timeSliceNumber);
  }
}


//-----------------------------------------------------------------------------
MIDASView QmitkMIDASSingleViewWidget::GetView() const
{
  return m_View;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SwitchView(MIDASView view)
{
  m_MultiWidget->SetMIDASView(view, true);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetView(MIDASView view, bool fitToDisplay)
{
  if (view != MIDAS_VIEW_UNKNOWN)
  {
    // Makes sure that we do have an active active geometry.
    this->SetActiveGeometry();

    // If for whatever reason, we have no geometry... bail out.
    if (m_ActiveGeometry.IsNull())
    {
      return;
    }

    // If we have a currently valid view/orientation, then store the current position, so we can switch back to it if necessary.
    this->StorePosition();

    // Store the currently selected position because the SetGeometry call resets it to the origin.
    mitk::Point3D selectedPosition = this->GetSelectedPosition();

    // This will initialise the whole QmitkStdMultiWidget according to the supplied geometry (normally an image).
    m_MultiWidget->SetGeometry(m_ActiveGeometry); // Sets geometry on all 4 MITK views.
    m_MultiWidget->SetMIDASView(view, true);            // True to always rebuild layout.
    m_MultiWidget->update();                            // Call Qt update to try and make sure we are painted at the right size.
    if (fitToDisplay)
    {
      m_MultiWidget->Fit();                             // Fits the MITK DisplayGeometry to the current widget size.
    }

    // Restore the selected position if it was set before.
    if (selectedPosition[0] != 0.0 || selectedPosition[1] != 0.0 || selectedPosition[2] != 0.0)
    {
      m_MultiWidget->SetSelectedPosition(selectedPosition);
    }

    // Now store the current view/orientation.
    MIDASOrientation orientation = this->GetOrientation();
    m_Orientation = orientation;
    m_View = view;

    // Now, in MIDAS, which only shows 2D views, if we revert to a previous view,
    // we should go back to the same slice, time, cursor position on display, magnification.
    bool hasBeenInitialised = m_ViewInitialised[Index(view)];
    if (m_RememberViewSettingsPerOrientation && hasBeenInitialised)
    {
      if (orientation != MIDAS_ORIENTATION_UNKNOWN)
      {
        this->SetSliceNumber(orientation, m_SliceNumbers[Index(orientation)]);
        this->SetTime(m_TimeSliceNumbers[Index(orientation)]);
      }
      this->SetCursorPosition(m_CursorPositions[Index(view)]);
      this->SetMagnification(m_Magnifications[Index(view)]);
    }
    else
    {
      if (orientation == MIDAS_ORIENTATION_UNKNOWN)
      {
        orientation = MIDAS_ORIENTATION_AXIAL; // somewhat arbitrary.
      }

      unsigned int sliceNumber = this->GetSliceNumber(orientation);
      unsigned int timeStep = this->GetTime();
      double magnification = m_MultiWidget->FitMagnification();
      const mitk::Vector3D& cursorPosition = m_MultiWidget->GetCursorPosition();

      this->SetSliceNumber(orientation, sliceNumber);
      this->SetTime(timeStep);
      this->SetCursorPosition(cursorPosition);
      this->SetMagnification(magnification);
      m_ViewInitialised[Index(view)] = true;
    }
  } // end view != MIDAS_VIEW_UNKNOWN
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkMIDASSingleViewWidget::GetSelectedPosition() const
{
  return m_MultiWidget->GetSelectedPosition();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition)
{
  if (m_View != MIDAS_VIEW_UNKNOWN)
  {
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
  if (m_View != MIDAS_VIEW_UNKNOWN)
  {
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
  if (m_View != MIDAS_VIEW_UNKNOWN)
  {
    m_MultiWidget->SetMagnification(magnification);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::paintEvent(QPaintEvent *event)
{
  QWidget::paintEvent(event);
  std::vector<QmitkRenderWindow*> renderWindows = GetRenderWindows();
  for (unsigned i = 0; i < renderWindows.size(); i++)
  {
    renderWindows[i]->GetVtkRenderWindow()->Render();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSingleViewWidget::InitializeStandardViews(const mitk::Geometry3D * geometry )
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
