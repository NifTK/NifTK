/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <QStackedLayout>
#include <QDebug>
#include "mitkFocusManager.h"
#include "mitkGlobalInteraction.h"
#include "mitkPointUtils.h"
#include "QmitkMIDASSingleViewWidget.h"
#include "QmitkRenderWindow.h"
#include "QmitkMIDASStdMultiWidget.h"
#include "vtkRenderer.h"
#include "vtkCamera.h"
#include "vtkMatrix4x4.h"
#include "vtkSmartPointer.h"
#include "itkConversionUtils.h"
#include "itkMatrix.h"
#include "itkSpatialOrientationAdapter.h"

QmitkMIDASSingleViewWidget::QmitkMIDASSingleViewWidget(
    QWidget *parent,
    QString windowName,
    int minimumMagnification,
    int maximumMagnification,
    mitk::DataStorage* dataStorage,
    mitk::RenderingManager* renderingManager
    )
  : QWidget(parent)
, m_DataStorage(NULL)
, m_RenderingManager(NULL)
, m_Layout(NULL)
, m_MultiWidget(NULL)
, m_IsBound(false)
, m_UnBoundTimeSlicedGeometry(NULL)
, m_BoundTimeSlicedGeometry(NULL)
, m_ActiveTimeSlicedGeometry(NULL)
, m_MinimumMagnification(minimumMagnification)
, m_MaximumMagnification(maximumMagnification)
, m_NavigationControllerEventListening(false)
, m_RememberViewSettingsPerOrientation(false)
{
  assert(dataStorage);
  m_DataStorage = dataStorage;

  assert(renderingManager);
  m_RenderingManager = renderingManager;

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
  m_MultiWidget = new QmitkMIDASStdMultiWidget(m_RenderingManager, m_DataStorage, this, NULL);
  this->SetNavigationControllerEventListening(false);

  m_Layout = new QGridLayout(this);
  m_Layout->setObjectName(QString::fromUtf8("QmitkMIDASSingleViewWidget::m_Layout"));
  m_Layout->setContentsMargins(1, 1, 1, 1);
  m_Layout->setVerticalSpacing(0);
  m_Layout->setHorizontalSpacing(0);
  m_Layout->addWidget(m_MultiWidget);

  // Connect to QmitkMIDASStdMultiWidget, so we can listen for signals.
  connect(m_MultiWidget, SIGNAL(NodesDropped(QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkMIDASStdMultiWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)));
  connect(m_MultiWidget, SIGNAL(PositionChanged(mitk::Point3D,mitk::Point3D)), this, SLOT(OnPositionChanged(mitk::Point3D,mitk::Point3D)));
}

QmitkMIDASSingleViewWidget::~QmitkMIDASSingleViewWidget()
{
}

void QmitkMIDASSingleViewWidget::OnNodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  // Try not to emit the QmitkMIDASStdMultiWidget pointer.
  emit NodesDropped(window, nodes);
}

void QmitkMIDASSingleViewWidget::OnPositionChanged(mitk::Point3D voxelLocation, mitk::Point3D millimetreLocation)
{
  emit PositionChanged(this, voxelLocation, millimetreLocation);
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

int QmitkMIDASSingleViewWidget::GetMinMagnification() const
{
  return this->m_MinimumMagnification;
}

int QmitkMIDASSingleViewWidget::GetMaxMagnification() const
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
  this->m_MultiWidget->SetDataStorage(dataStorage);
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
  int magnificationFactor = m_CurrentMagnificationFactors[currentArrayOffset];

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

void QmitkMIDASSingleViewWidget::SetGeometry(mitk::TimeSlicedGeometry::Pointer geometry)
{
  assert(geometry);
  this->m_UnBoundTimeSlicedGeometry = geometry;

  if (!this->m_IsBound)
  {
    this->ResetRememberedPositions(0, 2);
    this->ResetCurrentPosition(0);
  }
}

mitk::TimeSlicedGeometry::Pointer QmitkMIDASSingleViewWidget::GetGeometry()
{
  assert(this->m_UnBoundTimeSlicedGeometry);
  return this->m_UnBoundTimeSlicedGeometry;
}

void QmitkMIDASSingleViewWidget::SetBoundGeometry(mitk::TimeSlicedGeometry::Pointer geometry)
{
  assert(geometry);
  this->m_BoundTimeSlicedGeometry = geometry;

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
    m_ActiveTimeSlicedGeometry = m_BoundTimeSlicedGeometry;
  }
  else
  {
    m_ActiveTimeSlicedGeometry = m_UnBoundTimeSlicedGeometry;
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

void QmitkMIDASSingleViewWidget::SetView(MIDASView view, bool fitToDisplay)
{
  if (view != MIDAS_VIEW_UNKNOWN)
  {
    // Makes sure that we do have an active active geometry.
    this->SetActiveGeometry();

    // If we have a currently valid view/orientation, then store the current position, so we can switch back to it if necessary.
    this->StorePosition();

    // This will initialise the whole QmitkStdMultiWidget according to the supplied geometry (normally an image).
    this->m_MultiWidget->SetGeometry(this->m_ActiveTimeSlicedGeometry); // Sets geometry on all 4 MITK views.
    this->m_MultiWidget->SetMIDASView(view, true);                      // True to always rebuild layout.
    this->m_MultiWidget->update();                                      // Call Qt update to try and make sure we are painted at the right size.
    this->m_MultiWidget->Fit();                                         // Fits the MITK DisplayGeometry to the current widget size.

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
      int magnificationFactor = this->m_MultiWidget->FitMagnificationFactor();

      this->SetSliceNumber(orientation, sliceNumber);
      this->SetTime(timeStep);
      this->SetMagnificationFactor(magnificationFactor);
    }
  } // end view != MIDAS_VIEW_UNKNOWN
}

int QmitkMIDASSingleViewWidget::GetMagnificationFactor() const
{
  return this->m_CurrentMagnificationFactors[this->GetBoundUnboundOffset()];
}

void QmitkMIDASSingleViewWidget::SetMagnificationFactor(int magnificationFactor)
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
