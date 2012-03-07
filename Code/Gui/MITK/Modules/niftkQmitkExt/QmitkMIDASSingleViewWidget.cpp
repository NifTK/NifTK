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
#include "QmitkMIDASSingleViewWidget.h"
#include "QmitkRenderWindow.h"
#include "QmitkMIDASStdMultiWidget.h"
#include "vtkRenderer.h"
#include "vtkCamera.h"

QmitkMIDASSingleViewWidget::QmitkMIDASSingleViewWidget(
    QWidget *parent,
    QString windowName,
    int minimumMagnification,
    int maximumMagnification,
    mitk::DataStorage* dataStorage
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
, m_NavigationControllerEventListening(false)
{
  assert(dataStorage);
  m_DataStorage = dataStorage;

  this->setAcceptDrops(true);

  m_MinimumMagnification = minimumMagnification;
  m_MaximumMagnification = maximumMagnification;

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

  // Create our own RenderingManager, so we are NOT using the Global one.
  m_RenderingManager = mitk::RenderingManager::New();
  m_RenderingManager->SetConstrainedPaddingZooming(false);

  // Create the main QmitkMIDASStdMultiWidget, and pass in our OWN RenderingManager.
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

void QmitkMIDASSingleViewWidget::SetDisplay3DViewInOrthoView(bool visible)
{
  m_MultiWidget->SetDisplay3DViewInOrthoView(visible);
}

bool QmitkMIDASSingleViewWidget::GetDisplay3DViewInOrthoView() const
{
  return m_MultiWidget->GetDisplay3DViewInOrthoView();
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

void QmitkMIDASSingleViewWidget::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  this->m_DataStorage = dataStorage;
  this->m_RenderingManager->SetDataStorage(dataStorage);
  this->m_MultiWidget->SetDataStorage(dataStorage);
}

void QmitkMIDASSingleViewWidget::SetNavigationControllerEventListening(bool enabled)
{
  this->m_NavigationControllerEventListening = enabled;

  if (enabled)
  {
    this->m_MultiWidget->EnableNavigationControllerEventListening();
  }
  else
  {
    this->m_MultiWidget->DisableNavigationControllerEventListening();
  }
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

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
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

bool QmitkMIDASSingleViewWidget::GetBound()
{
  return this->m_IsBound;
}

void QmitkMIDASSingleViewWidget::SetBound(bool isBound)
{
  if (isBound == m_IsBound)
  {
    // No change, nothing to do.
    return;
  }

  this->m_IsBound = isBound;

  MIDASView view = this->m_CurrentViews[this->GetBoundUnboundOffset()]; // must come after this->m_IsBound = isBound so we pick up the orientation before views were bound
  this->m_CurrentViews[this->GetBoundUnboundOffset()] = MIDAS_VIEW_UNKNOWN; // to force a reset.

  this->SetActiveGeometry();
  this->SetView(view, false);
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

    // Store current settings if they were in fact for a valid orientation.
    this->StorePosition();

    // We Construct a mitk::Geometry to represent the maximum view to render.
    //
    // This is important, because unlike the ortho view, which expands its world
    // horizons when more data is added, in MIDAS we do not want this. It must be
    // exactly the size of the specified image (so that slice controllers make sense).
    //
    // Secondly, it must be in world coordinates, aligned with the standard
    // view where:
    //   as x increases we go Left->Right
    //   as y increases we go Posterior->Anterior
    //   as z increases we go Inferior->Superior.
    // In other words, it is determined by the transformed bounding box of the image geometry.
    //
    // Thirdly, it must be an image geometry.
    //
    // Fourthly, this is all a workaround due to the fact that MITK doesn't properly support rotated views.

    mitk::Point3D transformedOrigin;
    mitk::Point3D cornerPointsInImage[8];
    cornerPointsInImage[0] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, true, true);
    cornerPointsInImage[1] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, true, false);
    cornerPointsInImage[2] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, false, true);
    cornerPointsInImage[3] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, false, false);
    cornerPointsInImage[4] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, true, true);
    cornerPointsInImage[5] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, true, false);
    cornerPointsInImage[6] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, false, true);
    cornerPointsInImage[7] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, false, false);

    for (unsigned int i = 0; i < 8; i++)
    {
      MITK_DEBUG << "Matt, corner points in image=" << cornerPointsInImage[i] << std::endl;
    }
    transformedOrigin[0] = std::numeric_limits<float>::max();
    transformedOrigin[1] = std::numeric_limits<float>::max();
    transformedOrigin[2] = std::numeric_limits<float>::max();

    for (unsigned int i = 0; i < 8; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        if (cornerPointsInImage[i][j] < transformedOrigin[j])
        {
          transformedOrigin[j] = cornerPointsInImage[i][j];
        }
      }
    }

    // Do the same procedure for each axis, as each axis might be permuted and/or flipped/inverted.
    mitk::Vector3D originalSpacing = this->m_ActiveTimeSlicedGeometry->GetSpacing();
    mitk::Vector3D transformedSpacing;
    mitk::Point3D transformedExtent;

    for (unsigned int i = 0; i < 3; i++)
    {

      mitk::Vector3D axisVector = this->m_ActiveTimeSlicedGeometry->GetAxisVector(i);
      MITK_DEBUG << "Matt, axisVector=" << axisVector << std::endl;

      unsigned int axisInWorldSpace = 0;
      for (unsigned int j = 0; j < 3; j++)
      {
        if (fabs(axisVector[j]) > 0.5)
        {
          axisInWorldSpace = j;
        }
      }
      transformedSpacing[axisInWorldSpace] = originalSpacing[i];
      transformedExtent[axisInWorldSpace] = fabs(axisVector[axisInWorldSpace]/transformedSpacing[axisInWorldSpace]);
    }

    mitk::Geometry3D::BoundsArrayType transformedBoundingBox;
    transformedBoundingBox[0] = 0;
    transformedBoundingBox[1] = transformedExtent[0];
    transformedBoundingBox[2] = 0;
    transformedBoundingBox[3] = transformedExtent[1];
    transformedBoundingBox[4] = 0;
    transformedBoundingBox[5] = transformedExtent[2];

    mitk::TimeSlicedGeometry::Pointer timeSlicedTransformedGeometry = mitk::TimeSlicedGeometry::New();
    timeSlicedTransformedGeometry->InitializeEmpty(this->m_ActiveTimeSlicedGeometry->GetTimeSteps());
    //timeSlicedTransformedGeometry->SetImageGeometry(this->m_ActiveTimeSlicedGeometry->GetImageGeometry());
    timeSlicedTransformedGeometry->SetTimeBounds(this->m_ActiveTimeSlicedGeometry->GetTimeBounds());
    timeSlicedTransformedGeometry->SetEvenlyTimed(this->m_ActiveTimeSlicedGeometry->GetEvenlyTimed());
    timeSlicedTransformedGeometry->SetSpacing(transformedSpacing);
    timeSlicedTransformedGeometry->SetOrigin(transformedOrigin);
    timeSlicedTransformedGeometry->SetBounds(transformedBoundingBox);

    for (unsigned int i = 0; i < this->m_ActiveTimeSlicedGeometry->GetTimeSteps(); i++)
    {
      mitk::Geometry3D::Pointer transformedGeometry = mitk::Geometry3D::New();
      //transformedGeometry->SetImageGeometry(this->m_ActiveTimeSlicedGeometry->GetImageGeometry());
      transformedGeometry->SetSpacing(transformedSpacing);
      transformedGeometry->SetOrigin(transformedOrigin);
      transformedGeometry->SetBounds(transformedBoundingBox);
      transformedGeometry->SetTimeBounds(this->m_ActiveTimeSlicedGeometry->GetGeometry3D(i)->GetTimeBounds());

      timeSlicedTransformedGeometry->SetGeometry3D(transformedGeometry, i);
    }
    timeSlicedTransformedGeometry->UpdateInformation();

    MITK_DEBUG << "Matt, transformedSpacing=" << transformedSpacing << std::endl;
    MITK_DEBUG << "Matt, transformedOrigin=" << transformedOrigin << std::endl;
    MITK_DEBUG << "Matt, transformedBoundingBox=" << transformedBoundingBox << std::endl;
    MITK_DEBUG << "Matt, timeBounds=" << this->m_ActiveTimeSlicedGeometry->GetTimeBounds() << std::endl;

    this->m_CurrentViews[this->GetBoundUnboundOffset()] = view;
    this->m_MultiWidget->SetMIDASView(view, timeSlicedTransformedGeometry);
    this->m_MultiWidget->Fit();
/*
    mitk::SliceNavigationController::ViewDirection direction = mitk::SliceNavigationController::Original;
    if (orientation == MIDAS_VIEW_AXIAL)
    {
      direction = mitk::SliceNavigationController::Transversal;
      m_SliceNavigationController->Update(direction, false, false, true);
    }
    else if (orientation == MIDAS_VIEW_SAGITTAL)
    {
      direction = mitk::SliceNavigationController::Sagittal;
      m_SliceNavigationController->Update(direction, true, true, false);
    }
    else if (orientation == MIDAS_VIEW_CORONAL)
    {
      direction = mitk::SliceNavigationController::Frontal;
      m_SliceNavigationController->Update(direction, true, true, false);
    }

    unsigned int slice = std::numeric_limits<unsigned int>::min();
    unsigned int time = std::numeric_limits<unsigned int>::min();
    int magnification = std::numeric_limits<int>::min();

    if (m_PreviousViewOrientations[this->GetBoundUnboundPreviousArrayOffset() + orientation] == MIDAS_VIEW_UNKNOWN)
    {
      // No previous slice, so default to central slice.
      unsigned int steps = m_SliceNavigationController->GetSlice()->GetSteps();
      slice = (int)((steps - 1.0)/2.0);

      // No previous timestep, so default to zero.
      time = 0;

      // No previous magnification, so fit to display.

      // Use MITK to perform an initial Fit, then we adjust it,
      // by working out the effective scale, and adjusting it by zooming.
      mitk::DisplayGeometry::Pointer displayGeometry = baseRenderer->GetDisplayGeometry();
      displayGeometry->SetConstrainZoomingAndPanning(false);
      displayGeometry->Fit();

      // We do this with mitk::Point2D, so we have different values in X and Y, as images can be anisotropic.
      mitk::Point2D scaleFactorPixPerVoxel;
      mitk::Point2D scaleFactorPixPerMillimetres;
      this->GetScaleFactors(scaleFactorPixPerVoxel, scaleFactorPixPerMillimetres);

      // Now we scale these values so we get an integer number of pixels per voxel.
      mitk::Point2D targetScaleFactorPixPerVoxel;
      mitk::Point2D targetScaleFactorPixPerMillimetres;

      // Need to round the scale factors.
      for (int i = 0; i < 2; i++)
      {
        if (fitToDisplay)
        {
          targetScaleFactorPixPerVoxel[i] = scaleFactorPixPerVoxel[i];
          targetScaleFactorPixPerMillimetres[i] = scaleFactorPixPerMillimetres[i];
        }
        else
        {
          // If they are >= than 1, we round down towards 1
          // so you have less pixels per voxel, so image will appear smaller.
          if (scaleFactorPixPerVoxel[i] >= 1)
          {
            targetScaleFactorPixPerVoxel[i] = (int)(scaleFactorPixPerVoxel[i]);
          }
          else
          {
            // Otherwise, we still have to make image smaller to fit it on screen.
            //
            // For example, if the scale factor is 0.5, we invert it to get 2, which is an integer, so OK.
            // If however the scale factor is 0.4, we invert it to get 2.5 voxels per pixel, so we have
            // to round it up to 3, which means the image gets smaller (3 voxels fit into 1 pixel), and then
            // invert it to get the scale factor again.
            double tmp = 1.0 / scaleFactorPixPerVoxel[i];
            int roundedTmp = (int)(tmp + 0.5);
            tmp = 1.0 / (double) roundedTmp;
            targetScaleFactorPixPerVoxel[i] = tmp;
          }
          targetScaleFactorPixPerMillimetres[i] = scaleFactorPixPerMillimetres[i] * (targetScaleFactorPixPerVoxel[i]/scaleFactorPixPerVoxel[i]);
        }
      }

      // We may have anisotropic voxels, so find the axis that requires most scale factor change.
      int axisWithLargestDifference = 0;
      double largestDifference = std::numeric_limits<double>::min();
      for(int i = 0; i < 2; i++)
      {
        if (fabs(targetScaleFactorPixPerVoxel[i] - scaleFactorPixPerVoxel[i]) > largestDifference)
        {
          largestDifference = fabs(targetScaleFactorPixPerVoxel[i] - scaleFactorPixPerVoxel[i]);
          axisWithLargestDifference = i;
        }
      }

      // So the VTK display doesn't care about voxels, it is effectively isotropic, so you are scaling
      // in the millimetre space. So we pick one axis, and then calculate a scale factor, and apply it.
      double zoomScaleFactor = targetScaleFactorPixPerVoxel[axisWithLargestDifference] / scaleFactorPixPerVoxel[axisWithLargestDifference];
      this->ZoomDisplayAboutCentre(zoomScaleFactor);

      // See comments at top of header file
      if (targetScaleFactorPixPerVoxel[axisWithLargestDifference] > 0)
      {
        // So, if pixels per voxel = 2, midas magnification = 1.
        // So, if pixels per voxel = 1, midas magnification = 0. etc.
        magnification = targetScaleFactorPixPerVoxel[axisWithLargestDifference] - 1;
      }
      else
      {
        magnification = (int)(1.0 / targetScaleFactorPixPerVoxel[axisWithLargestDifference]) + 1;
      }

      this->SetSliceNumber(slice);
      this->SetTime(time);
      m_CurrentMagnificationFactors[this->GetBoundUnboundOffset()] = magnification;

      MITK_DEBUG << "QmitkMIDASSingleViewWidget::SetViewOrientation calculated slice=" << slice << ", time=" << time << ", magnification=" << magnification << std::endl;
    }
    else
    {

      unsigned int currentArrayOffset = this->GetBoundUnboundOffset();
      unsigned int previousArrayOffset = this->GetBoundUnboundPreviousArrayOffset();

      MIDASViewOrientation orientation = m_CurrentViewOrientations[currentArrayOffset];

      slice = m_PreviousSliceNumbers[previousArrayOffset + orientation];
      time = m_PreviousTimeSliceNumbers[orientation];
      magnification = m_PreviousMagnificationFactors[orientation];

      this->SetSliceNumber(slice);
      this->SetTime(time);
      this->SetMagnificationFactor(magnification);

      MITK_DEBUG << "QmitkMIDASSingleViewWidget::SetViewOrientation using previous settings, slice=" << slice << ", time=" << time << ", magnification=" << magnification << std::endl;
    }
    */
  }
}

int QmitkMIDASSingleViewWidget::GetMagnificationFactor(MIDASOrientation orientation) const
{
  return this->m_CurrentMagnificationFactors[this->GetBoundUnboundOffset()];
}

void QmitkMIDASSingleViewWidget::SetMagnificationFactor(MIDASOrientation orientation, int magnificationFactor)
{
  /*
  MITK_DEBUG << "Matt, requested magnificationFactor=" << magnificationFactor << std::endl;

  mitk::Point2D scaleFactorPixPerVoxel;
  mitk::Point2D scaleFactorPixPerMillimetres;
  this->GetScaleFactors(scaleFactorPixPerVoxel, scaleFactorPixPerMillimetres);

  MITK_DEBUG << "Matt, currently scaleFactorPixPerVoxel=" << scaleFactorPixPerVoxel << ", scaleFactorPixPerMillimetres=" << scaleFactorPixPerMillimetres << std::endl;

  double effectiveMagnificationFactor = 0;
  if (magnificationFactor >= 0)
  {
    effectiveMagnificationFactor = magnificationFactor + 1;
  }
  else
  {
    effectiveMagnificationFactor = magnificationFactor - 1;
    effectiveMagnificationFactor = fabs(1.0/effectiveMagnificationFactor);
  }

  MITK_DEBUG << "Matt, effectiveMagnificationFactor=" << effectiveMagnificationFactor << std::endl;

  mitk::Point2D targetScaleFactorPixPerMillimetres;

  // Need to scale both of the current scaleFactorPixPerVoxel[i]
  for (int i = 0; i < 2; i++)
  {
    targetScaleFactorPixPerMillimetres[i] = (effectiveMagnificationFactor / scaleFactorPixPerVoxel[i]) * scaleFactorPixPerMillimetres[i];
  }

  MITK_DEBUG << "Matt, targetScaleFactorPixPerMillimetres=" << targetScaleFactorPixPerMillimetres << std::endl;

  // Pick the one that has changed the least
  int axisWithLeastDifference = -1;
  double leastDifference = std::numeric_limits<double>::max();
  for(int i = 0; i < 2; i++)
  {
    if (fabs(targetScaleFactorPixPerMillimetres[i] - scaleFactorPixPerMillimetres[i]) < leastDifference)
    {
      leastDifference = fabs(targetScaleFactorPixPerMillimetres[i] - scaleFactorPixPerMillimetres[i]);
      axisWithLeastDifference = i;
    }
  }

  double zoomScaleFactor = targetScaleFactorPixPerMillimetres[axisWithLeastDifference]/scaleFactorPixPerMillimetres[axisWithLeastDifference];

  MITK_DEBUG << "Matt, axisWithLeastDifference=" << axisWithLeastDifference << std::endl;
  MITK_DEBUG << "Matt, zoomScaleFactor=" << zoomScaleFactor << std::endl;

  this->ZoomDisplayAboutCentre(zoomScaleFactor);
  this->m_CurrentMagnificationFactors[this->GetBoundUnboundOffset()] = magnificationFactor;
  this->RequestUpdate();
  */
}

void QmitkMIDASSingleViewWidget::ZoomDisplayAboutCentre(double scaleFactor)
{
  /*
  assert(this->m_SliceNavigationController);

  mitk::BaseRenderer::Pointer baseRenderer = this->m_SliceNavigationController->GetRenderer();
  assert(baseRenderer);

  mitk::DisplayGeometry::Pointer displayGeometry = baseRenderer->GetDisplayGeometry();
  assert(displayGeometry);

  mitk::Vector2D sizeInDisplayUnits = displayGeometry->GetSizeInDisplayUnits();
  mitk::Point2D centreOfDisplayInDisplayUnits;

  centreOfDisplayInDisplayUnits[0] = (sizeInDisplayUnits[0]-1.0)/2.0;
  centreOfDisplayInDisplayUnits[1] = (sizeInDisplayUnits[1]-1.0)/2.0;

  displayGeometry->Zoom(scaleFactor, centreOfDisplayInDisplayUnits);
  this->RequestUpdate();
  */
}

void QmitkMIDASSingleViewWidget::GetScaleFactors(mitk::Point2D &scaleFactorPixPerVoxel, mitk::Point2D &scaleFactorPixPerMillimetres)
{
  /*
  this->SetActiveGeometry();

  assert(this->m_SliceNavigationController);

  mitk::BaseRenderer::Pointer baseRenderer = this->m_SliceNavigationController->GetRenderer();
  assert(baseRenderer);

  mitk::DisplayGeometry::Pointer displayGeometry = baseRenderer->GetDisplayGeometry();
  assert(displayGeometry);

  mitk::Point3D cornerPointsInImage[8];
  cornerPointsInImage[0] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, true, true);
  cornerPointsInImage[1] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, true, false);
  cornerPointsInImage[2] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, false, true);
  cornerPointsInImage[3] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(true, false, false);
  cornerPointsInImage[4] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, true, true);
  cornerPointsInImage[5] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, true, false);
  cornerPointsInImage[6] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, false, true);
  cornerPointsInImage[7] = this->m_ActiveTimeSlicedGeometry->GetCornerPoint(false, false, false);

  scaleFactorPixPerVoxel[0] = std::numeric_limits<float>::max();
  scaleFactorPixPerVoxel[1] = std::numeric_limits<float>::max();

  // Take every combination of pairs of 3D corner points taken from the 8 corners of the geometry.
  for (unsigned int i = 0; i < 8; i++)
  {
    mitk::Point3D pointsInVoxels[2];

    for (unsigned int j = 1; j < 8; j++)
    {
      this->m_ActiveTimeSlicedGeometry->WorldToIndex(cornerPointsInImage[i], pointsInVoxels[0]);
      this->m_ActiveTimeSlicedGeometry->WorldToIndex(cornerPointsInImage[j], pointsInVoxels[1]);

      // We only want to pick pairs of points where the points differ in 3D
      // space along exactly one axis (i.e. no diagonals), and no duplicates.
      unsigned int differentVoxelIndexesCounter=0;
      int differentVoxelAxis = -1;

      for (unsigned int k = 0; k < 3; k++)
      {
        if (fabs(pointsInVoxels[1][k] - pointsInVoxels[0][k]) > 0.1)
        {
          differentVoxelIndexesCounter++;
          differentVoxelAxis = k;
        }
      }
      if (differentVoxelIndexesCounter == 1)
      {
        // So, for this pair of points, project to 2D
        mitk::Point2D displayPointInMillimetreCoordinates[2];
        mitk::Point2D displayPointInPixelCoordinates[2];

        displayGeometry->Map(cornerPointsInImage[i], displayPointInMillimetreCoordinates[0]);
        displayGeometry->WorldToDisplay(displayPointInMillimetreCoordinates[0], displayPointInPixelCoordinates[0]);

        displayGeometry->Map(cornerPointsInImage[j], displayPointInMillimetreCoordinates[1]);
        displayGeometry->WorldToDisplay(displayPointInMillimetreCoordinates[1], displayPointInPixelCoordinates[1]);

        // Similarly, we only want to pick pairs of points where the projected 2D points
        // differ in 2D display coordinates along exactly one axis.
        unsigned int differentDisplayIndexesCounter=0;
        int differentDisplayAxis = -1;

        for (unsigned int k = 0; k < 2; k++)
        {
          if (fabs(displayPointInPixelCoordinates[1][k] - displayPointInPixelCoordinates[0][k]) > 0.1)
          {
            differentDisplayIndexesCounter++;
            differentDisplayAxis = k;
          }
        }
        if (differentDisplayIndexesCounter == 1)
        {
          // We now have i,j corresponding to a pair of points that are different in
          // 1 axis in voxel space, and different in one axis in diplay space, we can
          // use them to calculate scale factors.
          double distanceInMillimetres = cornerPointsInImage[i].EuclideanDistanceTo(cornerPointsInImage[j]);
          double distanceInVoxels = pointsInVoxels[0].EuclideanDistanceTo(pointsInVoxels[1]);
          double distanceInPixels = displayPointInPixelCoordinates[0].EuclideanDistanceTo(displayPointInPixelCoordinates[1]);
          double scaleFactorInDisplayPixelsPerImageVoxel = distanceInPixels / distanceInVoxels;
          double scaleFactorInDisplayPixelsPerMillimetres = distanceInPixels / distanceInMillimetres;

          if (scaleFactorInDisplayPixelsPerImageVoxel < scaleFactorPixPerVoxel[differentDisplayAxis])
          {
            scaleFactorPixPerVoxel[differentDisplayAxis] = scaleFactorInDisplayPixelsPerImageVoxel;
            scaleFactorPixPerMillimetres[differentDisplayAxis] = scaleFactorInDisplayPixelsPerMillimetres;
            MITK_DEBUG << "Matt, i=" << i << ", world=" << cornerPointsInImage[i] << ", voxel=" << pointsInVoxels[0] << ", display=" << displayPointInPixelCoordinates[0] << std::endl;
            MITK_DEBUG << "Matt, j=" << j << ", world=" << cornerPointsInImage[j] << ", voxel=" << pointsInVoxels[1] << ", display=" << displayPointInPixelCoordinates[1] << std::endl;
            MITK_DEBUG << "Matt 3D axis of interest=" << differentVoxelAxis << ", 2D axis of interest=" << differentDisplayAxis << std::endl;
            MITK_DEBUG << "Matt updated axis=" << differentDisplayAxis << ", pix=" << distanceInPixels << ", vox=" << distanceInVoxels << ", mm=" << distanceInMillimetres << std::endl;
            MITK_DEBUG <<" Matt, scaleVox=" << scaleFactorPixPerVoxel << ", mm=" << scaleFactorPixPerMillimetres << std::endl;
          }
        }
      }
    }
  }
  MITK_DEBUG << "Matt, output = scaleFactorPixPerVoxel=" << scaleFactorPixPerVoxel << ", scaleFactorPixPerMillimetres=" << scaleFactorPixPerMillimetres << std::endl;
  */
}

