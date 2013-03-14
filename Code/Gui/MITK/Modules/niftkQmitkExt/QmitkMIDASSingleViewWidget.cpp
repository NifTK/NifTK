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
, m_MinimumMagnification(0)
, m_MaximumMagnification(0)
, m_CurrentView(MIDAS_VIEW_UNKNOWN)
, m_CurrentOrientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberViewSettingsPerOrientation(false)
{
  mitk::RenderingManager::Pointer renderingManager = mitk::RenderingManager::GetInstance();

  QString name("QmitkMIDASSingleViewWidget");
  this->Initialize(name, -5.0, 20.0, renderingManager, NULL);
}

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
, m_MinimumMagnification(0.0)
, m_MaximumMagnification(0.0)
, m_CurrentView(MIDAS_VIEW_UNKNOWN)
, m_CurrentOrientation(MIDAS_ORIENTATION_UNKNOWN)
, m_NavigationControllerEventListening(false)
, m_RememberViewSettingsPerOrientation(false)
{
  this->Initialize(windowName, minimumMagnification, maximumMagnification, renderingManager, dataStorage);
}

void QmitkMIDASSingleViewWidget::Initialize(QString windowName,
                double minimumMagnification,
                double maximumMagnification,
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
  m_MinimumMagnification = minimumMagnification;
  m_MaximumMagnification = maximumMagnification;

  this->setAcceptDrops(true);

  for (int i = 0; i < 6; i++)
  {
    m_PreviousSliceNumbers[i] = 0;
    m_PreviousTimeSliceNumbers[i] = 0;
    m_PreviousMagnificationFactors[i] = minimumMagnification;
    m_Initialised[i] = false;
  }
  for (int i = 6; i < 8; i++)
  {
    m_PreviousMagnificationFactors[i] = minimumMagnification;
    m_Initialised[i] = false;
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
  connect(m_MultiWidget, SIGNAL(PositionChanged(QmitkRenderWindow*, mitk::Index3D, mitk::Point3D, int, MIDASOrientation)), this, SLOT(OnPositionChanged(QmitkRenderWindow*,mitk::Index3D,mitk::Point3D, int, MIDASOrientation)));
  connect(m_MultiWidget, SIGNAL(MagnificationFactorChanged(QmitkRenderWindow*, double)), this, SLOT(OnMagnificationFactorChanged(QmitkRenderWindow*, double)));
}

QmitkMIDASSingleViewWidget::~QmitkMIDASSingleViewWidget()
{
}

void QmitkMIDASSingleViewWidget::OnNodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  // Try not to emit the QmitkMIDASStdMultiWidget pointer.
  emit NodesDropped(window, nodes);
}

void QmitkMIDASSingleViewWidget::OnPositionChanged(QmitkRenderWindow *window, mitk::Index3D voxelLocation, mitk::Point3D millimetreLocation, int sliceNumber, MIDASOrientation orientation)
{
  emit PositionChanged(this, window, voxelLocation, millimetreLocation, sliceNumber, orientation);
}

void QmitkMIDASSingleViewWidget::OnMagnificationFactorChanged(QmitkRenderWindow *window, double magnificationFactor)
{
  emit MagnificationFactorChanged(this, window, magnificationFactor);
}

bool QmitkMIDASSingleViewWidget::IsSingle2DView() const
{
  return m_MultiWidget->IsSingle2DView();
}

void QmitkMIDASSingleViewWidget::SetSelected(bool selected)
{
  m_MultiWidget->SetSelected(selected);
}

bool QmitkMIDASSingleViewWidget::IsSelected() const
{
  return m_MultiWidget->IsSelected();
}

void QmitkMIDASSingleViewWidget::SetSelectedRenderWindow(QmitkRenderWindow* renderWindow)
{
  m_MultiWidget->SetSelectedRenderWindow(renderWindow);
}

std::vector<QmitkRenderWindow*> QmitkMIDASSingleViewWidget::GetSelectedRenderWindows() const
{
  return m_MultiWidget->GetSelectedRenderWindows();
}

std::vector<QmitkRenderWindow*> QmitkMIDASSingleViewWidget::GetRenderWindows() const
{
  return m_MultiWidget->GetRenderWindows();
}

QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetAxialWindow() const
{
  return m_MultiWidget->GetRenderWindow1();
}

QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetSagittalWindow() const
{
  return m_MultiWidget->GetRenderWindow2();
}

QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetCoronalWindow() const
{
  return m_MultiWidget->GetRenderWindow3();
}

QmitkRenderWindow* QmitkMIDASSingleViewWidget::Get3DWindow() const
{
  return m_MultiWidget->GetRenderWindow4();
}

void QmitkMIDASSingleViewWidget::SetEnabled(bool enabled)
{
  m_MultiWidget->SetEnabled(enabled);
}

bool QmitkMIDASSingleViewWidget::IsEnabled() const
{
  return m_MultiWidget->IsEnabled();
}

void QmitkMIDASSingleViewWidget::SetDisplay2DCursorsLocally(bool visible)
{
  m_MultiWidget->SetDisplay2DCursorsLocally(visible);
}

bool QmitkMIDASSingleViewWidget::GetDisplay2DCursorsLocally() const
{
  return m_MultiWidget->GetDisplay2DCursorsLocally();
}

void QmitkMIDASSingleViewWidget::SetDisplay2DCursorsGlobally(bool visible)
{
  m_MultiWidget->SetDisplay2DCursorsGlobally(visible);
}

bool QmitkMIDASSingleViewWidget::GetDisplay2DCursorsGlobally() const
{
  return m_MultiWidget->GetDisplay2DCursorsGlobally();
}

void QmitkMIDASSingleViewWidget::SetDisplay3DViewInOrthoView(bool visible)
{
  m_MultiWidget->SetDisplay3DViewInOrthoView(visible);
}

bool QmitkMIDASSingleViewWidget::GetDisplay3DViewInOrthoView() const
{
  return m_MultiWidget->GetDisplay3DViewInOrthoView();
}

void QmitkMIDASSingleViewWidget::SetBackgroundColor(QColor color)
{
  m_MultiWidget->SetBackgroundColor(color);
}

QColor QmitkMIDASSingleViewWidget::GetBackgroundColor() const
{
  return m_MultiWidget->GetBackgroundColor();
}

unsigned int QmitkMIDASSingleViewWidget::GetMinSlice(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMinSlice(orientation);
}

unsigned int QmitkMIDASSingleViewWidget::GetMaxSlice(MIDASOrientation orientation) const
{
  return m_MultiWidget->GetMaxSlice(orientation);
}

unsigned int QmitkMIDASSingleViewWidget::GetMinTime() const
{
  return m_MultiWidget->GetMinTime();
}

unsigned int QmitkMIDASSingleViewWidget::GetMaxTime() const
{
  return m_MultiWidget->GetMaxTime();
}

bool QmitkMIDASSingleViewWidget::ContainsRenderWindow(QmitkRenderWindow *renderWindow) const
{
  return m_MultiWidget->ContainsRenderWindow(renderWindow);
}

QmitkRenderWindow* QmitkMIDASSingleViewWidget::GetRenderWindow(vtkRenderWindow *aVtkRenderWindow) const
{
  return m_MultiWidget->GetRenderWindow(aVtkRenderWindow);
}

MIDASOrientation QmitkMIDASSingleViewWidget::GetOrientation()
{
  return m_MultiWidget->GetOrientation();
}

void QmitkMIDASSingleViewWidget::FitToDisplay()
{
  m_MultiWidget->FitToDisplay();
}

void QmitkMIDASSingleViewWidget::SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible)
{
  m_MultiWidget->SetRendererSpecificVisibility(nodes, visible);
}

double QmitkMIDASSingleViewWidget::GetMinMagnification() const
{
  return this->m_MinimumMagnification;
}

double QmitkMIDASSingleViewWidget::GetMaxMagnification() const
{
  return this->m_MaximumMagnification;
}

mitk::DataStorage::Pointer QmitkMIDASSingleViewWidget::GetDataStorage() const
{
  return this->m_DataStorage;
}

void QmitkMIDASSingleViewWidget::SetRememberViewSettingsPerOrientation(bool remember)
{
  this->m_RememberViewSettingsPerOrientation = remember;
}

bool QmitkMIDASSingleViewWidget::GetRememberViewSettingsPerOrientation() const
{
  return this->m_RememberViewSettingsPerOrientation;
}

void QmitkMIDASSingleViewWidget::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  this->m_DataStorage = dataStorage;
  this->m_MultiWidget->SetDataStorage(this->m_DataStorage);
}

void QmitkMIDASSingleViewWidget::SetNavigationControllerEventListening(bool enabled)
{
  if (enabled)
  {
    this->m_MultiWidget->EnableNavigationControllerEventListening();
    this->m_MultiWidget->SetWidgetPlanesLocked(false);
  }
  else
  {
    this->m_MultiWidget->DisableNavigationControllerEventListening();
    this->m_MultiWidget->SetWidgetPlanesLocked(true);
  }
  this->m_NavigationControllerEventListening = enabled;
}

bool QmitkMIDASSingleViewWidget::GetNavigationControllerEventListening() const
{
  return m_NavigationControllerEventListening;
}

void QmitkMIDASSingleViewWidget::RequestUpdate()
{
  m_MultiWidget->RequestUpdate();
}

void QmitkMIDASSingleViewWidget::StorePosition()
{
  MIDASView view = m_CurrentView;
  MIDASOrientation orientation = m_CurrentOrientation;

  int sliceNumber = this->GetSliceNumber(orientation);
  int timeSliceNumber = this->GetTime();
  double magnificationFactor = this->m_MultiWidget->GetMagnificationFactor();

  if (view != MIDAS_VIEW_UNKNOWN && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    if (orientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      m_PreviousSliceNumbers[Index(orientation)] = sliceNumber;
      m_PreviousTimeSliceNumbers[Index(orientation)] = timeSliceNumber;

      MITK_DEBUG << "QmitkMIDASSingleViewWidget::StorePosition is bound=" << m_IsBound \
          << ", current orientation=" << orientation \
          << ", view=" << view \
          << ", so storing slice=" << sliceNumber \
          << ", time=" << timeSliceNumber \
          << ", magnification=" << magnificationFactor << std::endl;
    }
    // The magnification is stored also for the 3D render window.
    m_PreviousMagnificationFactors[Index(orientation)] = magnificationFactor;
  }
}

void QmitkMIDASSingleViewWidget::ResetCurrentPosition()
{
  MIDASOrientation orientation = m_CurrentOrientation;
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_PreviousSliceNumbers[Index(orientation)] = 0;
    m_PreviousTimeSliceNumbers[Index(orientation)] = 0;
  }
  m_PreviousMagnificationFactors[Index(orientation)] = this->m_MinimumMagnification;
}

void QmitkMIDASSingleViewWidget::ResetRememberedPositions()
{
  for (int i = 0; i < 3; i++)
  {
    m_PreviousSliceNumbers[Index(i)] = 0;
    m_PreviousTimeSliceNumbers[Index(i)] = 0;
    m_PreviousMagnificationFactors[Index(i)] = this->m_MinimumMagnification;
  }
  // The magnification is also stored for the 3D render window.
  m_PreviousMagnificationFactors[Index(3)] = this->m_MinimumMagnification;
}

void QmitkMIDASSingleViewWidget::SetGeometry(mitk::Geometry3D::Pointer geometry)
{
  assert(geometry);
  this->m_UnBoundGeometry = geometry;

  this->ResetRememberedPositions();
  this->ResetCurrentPosition();
}

mitk::Geometry3D::Pointer QmitkMIDASSingleViewWidget::GetGeometry()
{
  assert(this->m_UnBoundGeometry);
  return this->m_UnBoundGeometry;
}

void QmitkMIDASSingleViewWidget::SetBoundGeometry(mitk::Geometry3D::Pointer geometry)
{
  assert(geometry);
  this->m_BoundGeometry = geometry;

  this->ResetRememberedPositions();
  this->ResetCurrentPosition();
}

bool QmitkMIDASSingleViewWidget::GetBoundGeometryActive()
{
  return this->m_IsBound;
}

void QmitkMIDASSingleViewWidget::SetBoundGeometryActive(bool isBound)
{
  if (isBound == m_IsBound)
  {
    // No change, nothing to do.
    return;
  }

  this->m_IsBound = isBound;
  m_CurrentView = MIDAS_VIEW_UNKNOWN;
}

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

unsigned int QmitkMIDASSingleViewWidget::GetSliceNumber(MIDASOrientation orientation) const
{
  return this->m_MultiWidget->GetSliceNumber(orientation);
}

void QmitkMIDASSingleViewWidget::SetSliceNumber(MIDASOrientation orientation, unsigned int sliceNumber)
{
  if (m_CurrentOrientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    this->m_PreviousSliceNumbers[Index(m_CurrentOrientation)] = sliceNumber;
    this->m_MultiWidget->SetSliceNumber(orientation, sliceNumber);
  }
}

unsigned int QmitkMIDASSingleViewWidget::GetTime() const
{
  return this->m_MultiWidget->GetTime();
}

void QmitkMIDASSingleViewWidget::SetTime(unsigned int timeSliceNumber)
{
  if (m_CurrentOrientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    //  this->m_CurrentTimeSliceNumbers[Index(0)] = timeSliceNumber;
    this->m_PreviousTimeSliceNumbers[Index(m_CurrentOrientation)] = timeSliceNumber;
    this->m_MultiWidget->SetTime(timeSliceNumber);
  }
}

MIDASView QmitkMIDASSingleViewWidget::GetView() const
{
  return m_CurrentView;
}

void QmitkMIDASSingleViewWidget::SwitchView(MIDASView view)
{
  this->m_MultiWidget->SetMIDASView(view, true);
}

void QmitkMIDASSingleViewWidget::SetView(MIDASView view, bool fitToDisplay)
{
  if (view != MIDAS_VIEW_UNKNOWN)
  {
    // Makes sure that we do have an active active geometry.
    this->SetActiveGeometry();

    // If for whatever reason, we have no geometry... bail out.
    if (this->m_ActiveGeometry.IsNull())
    {
      return;
    }

    // If we have a currently valid view/orientation, then store the current position, so we can switch back to it if necessary.
    this->StorePosition();

    // This will initialise the whole QmitkStdMultiWidget according to the supplied geometry (normally an image).
    this->m_MultiWidget->SetGeometry(this->m_ActiveGeometry); // Sets geometry on all 4 MITK views.
    this->m_MultiWidget->SetMIDASView(view, true);            // True to always rebuild layout.
    this->m_MultiWidget->update();                            // Call Qt update to try and make sure we are painted at the right size.
    if (fitToDisplay)
    {
      this->m_MultiWidget->Fit();                             // Fits the MITK DisplayGeometry to the current widget size.
    }


    // Now store the current view/orientation.
    MIDASOrientation orientation = this->GetOrientation();
    m_CurrentOrientation = orientation;
    m_CurrentView = view;

    // Now, in MIDAS, which only shows 2D views, if we revert to a previous view,
    // we should go back to the same slice, time, magnification.
    bool hasBeenInitialised = m_Initialised[Index(orientation)];
    if (this->m_RememberViewSettingsPerOrientation && hasBeenInitialised)
    {
//      if (this->m_MultiWidget->IsSingle2DView())
      if (orientation < 3)
      {
        this->SetSliceNumber(orientation, m_PreviousSliceNumbers[Index(orientation)]);
        this->SetTime(m_PreviousTimeSliceNumbers[Index(orientation)]);
      }
      this->SetMagnificationFactor(m_PreviousMagnificationFactors[Index(orientation)]);
    }
    else
    {
      if (orientation == MIDAS_ORIENTATION_UNKNOWN)
      {
        orientation = MIDAS_ORIENTATION_AXIAL; // somewhat arbitrary.
      }

      unsigned int sliceNumber = this->GetSliceNumber(orientation);
      unsigned int timeStep = this->GetTime();
      double magnificationFactor = this->m_MultiWidget->FitMagnificationFactor();

      this->SetSliceNumber(orientation, sliceNumber);
      this->SetTime(timeStep);
      this->SetMagnificationFactor(magnificationFactor);
      m_Initialised[Index(orientation)] = true;
    }
  } // end view != MIDAS_VIEW_UNKNOWN
}

double QmitkMIDASSingleViewWidget::GetMagnificationFactor() const
{
  return this->m_PreviousMagnificationFactors[Index(m_CurrentOrientation)];
}

void QmitkMIDASSingleViewWidget::SetMagnificationFactor(double magnificationFactor)
{
  this->m_PreviousMagnificationFactors[Index(m_CurrentOrientation)] = magnificationFactor;
  this->m_MultiWidget->SetMagnificationFactor(magnificationFactor);
}

mitk::Point3D QmitkMIDASSingleViewWidget::GetSelectedPosition() const
{
  return this->m_MultiWidget->GetCrossPosition();
}

void QmitkMIDASSingleViewWidget::SetSelectedPosition(const mitk::Point3D &pos)
{
  this->m_MultiWidget->MoveCrossToPosition(pos);
}

void QmitkMIDASSingleViewWidget::paintEvent(QPaintEvent *event)
{
  QWidget::paintEvent(event);
  std::vector<QmitkRenderWindow*> renderWindows = GetRenderWindows();
  for (unsigned i = 0; i < renderWindows.size(); i++)
  {
    renderWindows[i]->GetVtkRenderWindow()->Render();
  }
}

void QmitkMIDASSingleViewWidget::InitializeStandardViews(const mitk::Geometry3D * geometry )
{
  this->m_MultiWidget->InitializeStandardViews(geometry);
}

std::vector<mitk::DataNode*> QmitkMIDASSingleViewWidget::GetWidgetPlanes()
{
  std::vector<mitk::DataNode*> result;
  result.push_back(this->m_MultiWidget->GetWidgetPlane1());
  result.push_back(this->m_MultiWidget->GetWidgetPlane2());
  result.push_back(this->m_MultiWidget->GetWidgetPlane3());
  return result;
}


int QmitkMIDASSingleViewWidget::GetSliceUpDirection(MIDASOrientation orientation) const
{
  return this->m_MultiWidget->GetSliceUpDirection(orientation);
}
