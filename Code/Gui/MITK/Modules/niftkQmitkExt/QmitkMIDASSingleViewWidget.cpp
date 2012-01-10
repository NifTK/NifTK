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

#include "QmitkMIDASSingleViewWidget.h"
#include "QmitkMIDASRenderWindow.h"

QmitkMIDASSingleViewWidget::QmitkMIDASSingleViewWidget(
    QWidget *parent,
    QString windowName,
    int minimumMagnification,
    int maximumMagnification)
  : QWidget(parent)
, m_RenderWindow(NULL)
, m_RenderWindowFrame(NULL)
, m_RenderWindowBackground(NULL)
, m_RenderingManager(NULL)
, m_SliceNavigationController(NULL)
, m_DataStorage(NULL)
, m_Geometry(NULL)
, m_Layout(NULL)
, m_SliceNumber(0)
, m_TimeSliceNumber(0)
, m_MagnificationFactor(minimumMagnification)
, m_ViewOrientation(MIDAS_VIEW_UNKNOWN)

{
  this->setAcceptDrops(true);

  m_BackgroundColor = QColor(255, 250, 240);  // that strange MIDAS background color.
  m_SelectedColor   = QColor(255, 0, 0);
  m_UnselectedColor = QColor(255, 255, 255);

  m_MinimumMagnification = minimumMagnification;
  m_MaximumMagnification = maximumMagnification;

  // Create the main QmitkMIDASRenderWindow, which registers it with the Global RenderingManager.
  m_RenderWindow = new QmitkMIDASRenderWindow(this, windowName);
  m_RenderWindow->setAcceptDrops(true);
  m_RenderWindow->setEnabled(false);
  m_RenderWindow->setVisible(false);

  // Remove from the Global RenderingManager before we get asked (by any another method, anywhere in application)
  // to render when this widget is not visible which causes InvalidDrawable on Mac, and invalid X windows and crash on Linux.
  mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_RenderWindow->GetVtkRenderWindow());

  m_RenderWindowFrame = mitk::RenderWindowFrame::New();
  m_RenderWindowFrame->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_RenderWindowFrame->Enable(
      m_UnselectedColor.redF(),
      m_UnselectedColor.greenF(),
      m_UnselectedColor.blueF()
      );

  m_RenderWindowBackground = mitk::GradientBackground::New();
  m_RenderWindowBackground->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_RenderWindowBackground->SetGradientColors(
      m_BackgroundColor.redF(),
      m_BackgroundColor.greenF(),
      m_BackgroundColor.blueF(),
      m_BackgroundColor.redF(),
      m_BackgroundColor.greenF(),
      m_BackgroundColor.blueF());
  m_RenderWindowBackground->Enable();

  // Sort out our Rendering controller stuff.
  m_RenderingManager = mitk::RenderingManager::New();
  m_SliceNavigationController = m_RenderWindow->GetSliceNavigationController();
  m_SliceNavigationController->SetRenderingManager(m_RenderingManager);

  // But we have to remember the slice, magnification and orientation for 3 views, so initilise these to invalid.
  for (int i = 0; i < 3; i++)
  {
    m_SliceNumbers.push_back(0);
    m_TimeSliceNumbers.push_back(0);
    m_MagnificationFactors.push_back(minimumMagnification);
    m_ViewOrientations.push_back(MIDAS_VIEW_UNKNOWN);
  }

  m_Layout = new QGridLayout(this);
  m_Layout->setObjectName(QString::fromUtf8("QmitkMIDASSingleViewWidget::m_Layout"));
  m_Layout->addWidget(m_RenderWindow, 0, 0);
}

QmitkMIDASSingleViewWidget::~QmitkMIDASSingleViewWidget()
{
}

void QmitkMIDASSingleViewWidget::SetContentsMargins(unsigned int margin)
{
  m_Layout->setContentsMargins(margin, margin, margin, margin);
}

void QmitkMIDASSingleViewWidget::SetSpacing(unsigned int spacing)
{
  m_Layout->setSpacing(spacing);
}

void QmitkMIDASSingleViewWidget::SetSelected(bool selected)
{
  if (selected)
  {
    this->m_RenderWindowFrame->Enable(
        m_SelectedColor.redF(),
        m_SelectedColor.greenF(),
        m_SelectedColor.blueF()
        );
  }
  else
  {
    this->m_RenderWindowFrame->Enable(
        m_UnselectedColor.redF(),
        m_UnselectedColor.greenF(),
        m_UnselectedColor.blueF()
        );
  }
}

void QmitkMIDASSingleViewWidget::SetSelectedColor(QColor color)
{
  this->m_SelectedColor = color;
}

QColor QmitkMIDASSingleViewWidget::GetSelectedColor() const
{
  return this->m_SelectedColor;
}

void QmitkMIDASSingleViewWidget::SetUnselectedColor(QColor color)
{
  this->m_UnselectedColor = color;
}

QColor QmitkMIDASSingleViewWidget::GetUnselectedColor() const
{
  return this->m_UnselectedColor;
}

void QmitkMIDASSingleViewWidget::SetBackgroundColor(QColor color)
{
  this->m_BackgroundColor = color;

  m_RenderWindowBackground->SetGradientColors(
      m_BackgroundColor.redF(),
      m_BackgroundColor.greenF(),
      m_BackgroundColor.blueF(),
      m_BackgroundColor.redF(),
      m_BackgroundColor.greenF(),
      m_BackgroundColor.blueF());
}

QColor QmitkMIDASSingleViewWidget::GetBackgroundColor() const
{
  return this->m_BackgroundColor;
}

unsigned int QmitkMIDASSingleViewWidget::GetMinSlice() const
{
  return this->m_SliceNavigationController->GetSlice()->GetRangeMin();
}

unsigned int QmitkMIDASSingleViewWidget::GetMaxSlice() const
{
  return this->m_SliceNavigationController->GetSlice()->GetRangeMax();
}

unsigned int QmitkMIDASSingleViewWidget::GetMinTime() const
{
  return this->m_SliceNavigationController->GetTime()->GetRangeMin();
}

unsigned int QmitkMIDASSingleViewWidget::GetMaxTime() const
{
  return this->m_SliceNavigationController->GetTime()->GetRangeMax();
}

int QmitkMIDASSingleViewWidget::GetMinMagnification() const
{
  return this->m_MinimumMagnification;
}

int QmitkMIDASSingleViewWidget::GetMaxMagnification() const
{
  return this->m_MaximumMagnification;
}

void QmitkMIDASSingleViewWidget::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  this->m_DataStorage = dataStorage;

  m_RenderWindow->GetRenderer()->SetDataStorage(dataStorage);
}

mitk::DataStorage::Pointer QmitkMIDASSingleViewWidget::GetDataStorage(mitk::DataStorage* dataStorage)
{
  return this->m_DataStorage;
}

bool QmitkMIDASSingleViewWidget::ContainsWindow(QmitkMIDASRenderWindow *window) const
{
  bool containsWindow = false;
  if (m_RenderWindow == window)
  {
    containsWindow = true;
  }
  return containsWindow;
}

bool QmitkMIDASSingleViewWidget::ContainsVtkRenderWindow(vtkRenderWindow *window) const
{
  bool containsWindow = false;
  if (m_RenderWindow->GetVtkRenderWindow() == window)
  {
    containsWindow = true;
  }
  return containsWindow;
}

QmitkMIDASRenderWindow* QmitkMIDASSingleViewWidget::GetRenderWindow() const
{
  return this->m_RenderWindow;
}

void QmitkMIDASSingleViewWidget::SetSliceNumber(unsigned int sliceNumber)
{
  this->m_SliceNavigationController->GetSlice()->SetPos(sliceNumber);
  this->m_SliceNumber = sliceNumber;
}

unsigned int QmitkMIDASSingleViewWidget::GetSliceNumber() const
{
  return this->m_SliceNumber;
}

void QmitkMIDASSingleViewWidget::SetTime(unsigned int timeSliceNumber)
{
  this->m_SliceNavigationController->GetTime()->SetPos(timeSliceNumber);
  this->m_TimeSliceNumber = timeSliceNumber;
}

unsigned int QmitkMIDASSingleViewWidget::GetTime() const
{
  return this->m_TimeSliceNumber;
}

void QmitkMIDASSingleViewWidget::SetMagnificationFactor(int magnificationFactor)
{

}

int QmitkMIDASSingleViewWidget::GetMagnificationFactor() const
{
  return this->m_MagnificationFactor;
}

void QmitkMIDASSingleViewWidget::RemoveFromRenderingManager()
{
  this->m_RenderWindow->setEnabled(false);
  this->m_RenderWindow->setVisible(false);
  this->m_RenderingManager->RemoveRenderWindow(this->m_RenderWindow->GetVtkRenderWindow());
}

void QmitkMIDASSingleViewWidget::AddToRenderingManager()
{
  this->m_RenderingManager->AddRenderWindow(this->m_RenderWindow->GetVtkRenderWindow());
  this->m_RenderWindow->setEnabled(true);
  this->m_RenderWindow->setVisible(true);
  this->RequestUpdate();
}

void QmitkMIDASSingleViewWidget::RequestUpdate()
{
  vtkRenderWindow* vtkWindow = m_RenderWindow->GetVtkRenderWindow();
  this->m_RenderingManager->RequestUpdate(vtkWindow);
}

void QmitkMIDASSingleViewWidget::ForceUpdate()
{
  vtkRenderWindow* vtkWindow = m_RenderWindow->GetVtkRenderWindow();
  this->m_RenderingManager->ForceImmediateUpdate(vtkWindow);
}

QmitkMIDASSingleViewWidget::MIDASViewOrientation QmitkMIDASSingleViewWidget::GetViewOrientation() const
{
  return this->m_ViewOrientation;
}

void QmitkMIDASSingleViewWidget::SetViewOrientation(MIDASViewOrientation orientation)
{
  if (   orientation != MIDAS_VIEW_UNKNOWN
      && this->m_Geometry != NULL)
  {
    // Store current settings
    if (m_RenderWindow->GetSliceNavigationController()->GetSlice()->GetPos() != 0)
    {
      m_SliceNumbers[this->m_ViewOrientation] = m_RenderWindow->GetSliceNavigationController()->GetSlice()->GetPos();
    }

    vtkRenderWindow* vtkWindow = m_RenderWindow->GetVtkRenderWindow();
    assert(vtkWindow);

    mitk::BaseRenderer::Pointer baseRenderer = this->m_SliceNavigationController->GetRenderer();
    assert(baseRenderer);

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

    mitk::Point3D transformedOrigin;
    mitk::Point3D cornerPointsInImage[8];
    cornerPointsInImage[0] = this->m_Geometry->GetCornerPoint(true, true, true);
    cornerPointsInImage[1] = this->m_Geometry->GetCornerPoint(true, true, false);
    cornerPointsInImage[2] = this->m_Geometry->GetCornerPoint(true, false, true);
    cornerPointsInImage[3] = this->m_Geometry->GetCornerPoint(true, false, false);
    cornerPointsInImage[4] = this->m_Geometry->GetCornerPoint(false, true, true);
    cornerPointsInImage[5] = this->m_Geometry->GetCornerPoint(false, true, false);
    cornerPointsInImage[6] = this->m_Geometry->GetCornerPoint(false, false, true);
    cornerPointsInImage[7] = this->m_Geometry->GetCornerPoint(false, false, false);

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
    mitk::Vector3D originalSpacing = this->m_Geometry->GetSpacing();
    mitk::Vector3D transformedSpacing;
    mitk::Point3D transformedExtent;

    for (unsigned int i = 0; i < 3; i++)
    {

      mitk::Vector3D axisVector = this->m_Geometry->GetAxisVector(i);
      mitk::Vector3D transformedAxis;

      this->m_Geometry->IndexToWorld(axisVector, transformedAxis);

      int axisInWorldSpace = -1;
      for (unsigned int j = 0; j < 3; j++)
      {
        if (fabs(transformedAxis[j]) > 0.5)
        {
          axisInWorldSpace = j;
        }
      }
      transformedSpacing[axisInWorldSpace] = originalSpacing[i];
      transformedExtent[axisInWorldSpace] = fabs(transformedAxis[axisInWorldSpace]);
    }

    mitk::Geometry3D::BoundsArrayType originalBoundingBox = this->m_Geometry->GetBounds();
    mitk::Geometry3D::BoundsArrayType transformedBoundingBox;
    transformedBoundingBox[0] = 0;
    transformedBoundingBox[1] = transformedExtent[0];
    transformedBoundingBox[2] = 0;
    transformedBoundingBox[3] = transformedExtent[1];
    transformedBoundingBox[4] = 0;
    transformedBoundingBox[5] = transformedExtent[2];

    mitk::Geometry3D::Pointer transformedGeometry = mitk::Geometry3D::New();
    transformedGeometry->SetImageGeometry(true);
    transformedGeometry->SetSpacing(transformedSpacing);
    transformedGeometry->SetOrigin(transformedOrigin);
    transformedGeometry->SetBounds(transformedBoundingBox);
    baseRenderer->SetWorldGeometry(transformedGeometry);

    // Set the view to the new orientation
    this->m_ViewOrientation = orientation;

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

    int slice = -1;
    if (m_ViewOrientations[orientation] == MIDAS_VIEW_UNKNOWN)
    {
      // No previous slice, so default to central slice.
      unsigned int steps = m_SliceNavigationController->GetSlice()->GetSteps();
      slice = (int)((steps - 1)/2);
    }
    else
    {
      slice = m_SliceNumbers[this->m_ViewOrientation];
    }
    this->SetSliceNumber(slice);

    baseRenderer->GetDisplayGeometry()->Fit();
    this->RequestUpdate();
  }
}

void QmitkMIDASSingleViewWidget::InitializeGeometry(mitk::Geometry3D::Pointer geometry)
{
  // Store the geometry for later
  this->m_Geometry = geometry;

  // If we reset these variables, then code that works out orientation should re-initialize,
  // as it is like we have had no previous geometry information before.
  for (unsigned int i = 0; i < 3; i++)
  {
    m_SliceNumbers[i] = 0;
    m_TimeSliceNumbers[i] = 0;
    m_MagnificationFactors[i] = this->m_MinimumMagnification;
    m_ViewOrientations[i] = MIDAS_VIEW_UNKNOWN;
  }
  this->m_ViewOrientation = MIDAS_VIEW_UNKNOWN;
}
