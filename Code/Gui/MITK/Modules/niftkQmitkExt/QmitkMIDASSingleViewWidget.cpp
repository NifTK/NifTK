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
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
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
, m_NavigationControllerEventListening(false)
, m_RememberViewSettingsPerOrientation(false)
{
  mitk::RenderingManager::Pointer renderingManager = mitk::RenderingManager::GetInstance();

  QString name("QmitkMIDASSingleViewWidget");
  this->Initialize(name, -5, 20, renderingManager, NULL);
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
, m_MinimumMagnification(0)
, m_MaximumMagnification(0)
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

  // We maintain "current slice, current magnification" for both bound and unbound views = 2 of each.
  for (unsigned int i = 0; i < 2; i++)
  {
    m_CurrentSliceNumbers.push_back(0);
    m_CurrentTimeSliceNumbers.push_back(0);
    m_CurrentMagnificationFactors.push_back(minimumMagnification);
    m_CurrentOrientations.push_back(MIDAS_ORIENTATION_UNKNOWN);
    m_CurrentViews.push_back(MIDAS_VIEW_UNKNOWN);
  }

  // But we have to remember the slice, magnification and orientation for 3 views unbound, then 3 views bound = 6 of each.
  for (int i = 0; i < 6; i++)
  {
    m_PreviousSliceNumbers.push_back(0);
    m_PreviousTimeSliceNumbers.push_back(0);
    m_PreviousMagnificationFactors.push_back(minimumMagnification);
    m_PreviousOrientations.push_back(MIDAS_ORIENTATION_UNKNOWN);
    m_PreviousViews.push_back(MIDAS_VIEW_UNKNOWN);
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
//  SetMagnificationFactor(magnificationFactor);
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

void QmitkMIDASSingleViewWidget::SetSelectedWindow(vtkRenderWindow* window)
{
  m_MultiWidget->SetSelectedWindow(window);
}

std::vector<QmitkRenderWindow*> QmitkMIDASSingleViewWidget::GetSelectedWindows() const
{
  return m_MultiWidget->GetSelectedWindows();
}

std::vector<QmitkRenderWindow*> QmitkMIDASSingleViewWidget::GetAllWindows() const
{
  return m_MultiWidget->GetAllWindows();
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

std::vector<vtkRenderWindow*> QmitkMIDASSingleViewWidget::GetAllVtkWindows() const
{
  return m_MultiWidget->GetAllVtkWindows();
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

bool QmitkMIDASSingleViewWidget::ContainsWindow(QmitkRenderWindow *window) const
{
  return m_MultiWidget->ContainsWindow(window);
}

bool QmitkMIDASSingleViewWidget::ContainsVtkRenderWindow(vtkRenderWindow *window) const
{
  return m_MultiWidget->ContainsVtkRenderWindow(window);
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

unsigned int QmitkMIDASSingleViewWidget::GetBoundUnboundOffset() const
{
  // So we have arrays of length 2, index=0 corresponds to 'Un-bound', and index=1 corresponds to 'Bound'.
  if (m_IsBound)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

unsigned int QmitkMIDASSingleViewWidget::GetBoundUnboundPreviousArrayOffset() const
{
  // So we have arrays of length 6, index=0-2 corresponds to 'Un-bound' for axial, sagittal, coronal,
  // and index=3-5 corresponds to 'Bound' for axial, sagittal, coronal.
  if (m_IsBound)
  {
    return 3;
  }
  else
  {
    return 0;
  }
}

void QmitkMIDASSingleViewWidget::StorePosition()
{
  unsigned int currentArrayOffset = this->GetBoundUnboundOffset();
  unsigned int previousArrayOffset = this->GetBoundUnboundPreviousArrayOffset();

  MIDASView view = m_CurrentViews[currentArrayOffset];
  MIDASOrientation orientation = m_CurrentOrientations[currentArrayOffset];

  int sliceNumber = m_CurrentSliceNumbers[currentArrayOffset];
  int timeSliceNumber = m_CurrentTimeSliceNumbers[currentArrayOffset];
  double magnificationFactor = m_CurrentMagnificationFactors[currentArrayOffset];

  if (view != MIDAS_VIEW_UNKNOWN && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    // Dodgy style: orientation is an enum, being used as an array index.

    m_PreviousSliceNumbers[previousArrayOffset + orientation] = sliceNumber;
    m_PreviousTimeSliceNumbers[previousArrayOffset + orientation] = timeSliceNumber;
    m_PreviousMagnificationFactors[previousArrayOffset + orientation] = magnificationFactor;
    m_PreviousOrientations[previousArrayOffset + orientation] = orientation;
    m_PreviousViews[previousArrayOffset + orientation] = view;

    MITK_DEBUG << "QmitkMIDASSingleViewWidget::StorePosition is bound=" << m_IsBound \
        << ", current orientation=" << orientation \
        << ", view=" << view \
        << ", so storing slice=" << sliceNumber \
        << ", time=" << timeSliceNumber \
        << ", magnification=" << magnificationFactor << std::endl;
  }
}

void QmitkMIDASSingleViewWidget::ResetCurrentPosition(unsigned int currentIndex)
{
  assert(currentIndex >=0);
  assert(currentIndex <=1);

  m_CurrentSliceNumbers[currentIndex] = 0;
  m_CurrentTimeSliceNumbers[currentIndex] = 0;
  m_CurrentMagnificationFactors[currentIndex] = this->m_MinimumMagnification;
  m_CurrentOrientations[currentIndex] = MIDAS_ORIENTATION_UNKNOWN;
  m_CurrentViews[currentIndex] = MIDAS_VIEW_UNKNOWN;
}

void QmitkMIDASSingleViewWidget::ResetRememberedPositions(unsigned int startIndex, unsigned int stopIndex)
{
  // NOTE: The positions array is off length 6, corresponding to
  // Unbound (axial, sagittal, coronal), Bound (axial, sagittal, coronal).

  assert(startIndex >= 0);
  assert(stopIndex >= 0);
  assert(startIndex <= 5);
  assert(stopIndex <= 5);
  assert(startIndex <= stopIndex);

  for (unsigned int i = startIndex; i <= stopIndex; i++)
  {
    m_PreviousSliceNumbers[i] = 0;
    m_PreviousTimeSliceNumbers[i] = 0;
    m_PreviousMagnificationFactors[i] = this->m_MinimumMagnification;
    m_PreviousOrientations[i] = MIDAS_ORIENTATION_UNKNOWN;
    m_PreviousViews[i] = MIDAS_VIEW_UNKNOWN;
  }
}

void QmitkMIDASSingleViewWidget::SetGeometry(mitk::Geometry3D::Pointer geometry)
{
  assert(geometry);
  this->m_UnBoundGeometry = geometry;

  if (!this->m_IsBound)
  {
    this->ResetRememberedPositions(0, 2);
    this->ResetCurrentPosition(0);
  }
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

  if (this->m_IsBound)
  {
    this->ResetRememberedPositions(3, 5);
    this->ResetCurrentPosition(1);
  }
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
  this->m_CurrentViews[this->GetBoundUnboundOffset()] = MIDAS_VIEW_UNKNOWN; // to force a reset.
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
  this->m_CurrentSliceNumbers[this->GetBoundUnboundOffset()] = sliceNumber;
  this->m_MultiWidget->SetSliceNumber(orientation, sliceNumber);
}

unsigned int QmitkMIDASSingleViewWidget::GetTime() const
{
  return this->m_MultiWidget->GetTime();
}

void QmitkMIDASSingleViewWidget::SetTime(unsigned int timeSliceNumber)
{
  this->m_CurrentTimeSliceNumbers[this->GetBoundUnboundOffset()] = timeSliceNumber;
  this->m_MultiWidget->SetTime(timeSliceNumber);
}

MIDASView QmitkMIDASSingleViewWidget::GetView() const
{
  return this->m_CurrentViews[this->GetBoundUnboundOffset()];
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
    this->m_CurrentViews[this->GetBoundUnboundOffset()] = view;
    MIDASOrientation orientation = this->GetOrientation();
    this->m_CurrentOrientations[this->GetBoundUnboundOffset()] = orientation;

    // Now, in MIDAS, which only shows 2D views, if we revert to a previous view,
    // we should go back to the same slice, time, magnification.
    if (this->m_RememberViewSettingsPerOrientation
        && this->m_MultiWidget->IsSingle2DView()
        && m_PreviousOrientations[this->GetBoundUnboundPreviousArrayOffset() + orientation] != MIDAS_ORIENTATION_UNKNOWN)
    {
      this->SetSliceNumber(orientation, m_PreviousSliceNumbers[this->GetBoundUnboundPreviousArrayOffset() + orientation]);
      this->SetTime(m_PreviousTimeSliceNumbers[this->GetBoundUnboundPreviousArrayOffset() + orientation]);
      this->SetMagnificationFactor(m_PreviousMagnificationFactors[this->GetBoundUnboundPreviousArrayOffset() + orientation]);
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
    }
  } // end view != MIDAS_VIEW_UNKNOWN
}

double QmitkMIDASSingleViewWidget::GetMagnificationFactor() const
{
  return this->m_CurrentMagnificationFactors[this->GetBoundUnboundOffset()];
}

void QmitkMIDASSingleViewWidget::SetMagnificationFactor(double magnificationFactor)
{
  this->m_CurrentMagnificationFactors[this->GetBoundUnboundOffset()] = magnificationFactor;
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
  std::vector<vtkRenderWindow*> vtkRenderWindows = GetAllVtkWindows();
  for (unsigned int i = 0; i < vtkRenderWindows.size(); i++)
  {
    vtkRenderWindows[i]->Render();
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
