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

QmitkMIDASSingleViewWidget::QmitkMIDASSingleViewWidget(QWidget *parent, QString windowName)
  : QWidget(parent)
, m_Layout(NULL)
, m_RenderWindow(NULL)
, m_RenderWindowFrame(NULL)
, m_RenderWindowBackground(NULL)
, m_DataStorage(NULL)
, m_SliceNumber(-1)
, m_MagnificationFactor(-6)
, m_ViewOrientation(MIDAS_VIEW_UNKNOWN)
, m_Geometry(NULL)
{
  this->setAcceptDrops(true);

  m_BackgroundColor = QColor(255, 250, 240);  // that strange MIDAS background color.
  m_SelectedColor   = QColor(255, 0, 0);
  m_UnselectedColor = QColor(255, 255, 255);

  m_RenderWindow = new QmitkMIDASRenderWindow(this, windowName);
  m_RenderWindow->setAcceptDrops(true);
  m_RenderWindow->setEnabled(false);
  m_RenderWindow->setVisible(false);

  // Remove from the rendering manager before we get asked (by any another method)
  // to render when widget not visible, as otherwise you get "Invalid drawable" errors.
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

  // But we have to remember the slice, magnification and orientation for 3 views, so initilise these to invalid.
  for (int i = 0; i < 3; i++)
  {
    m_SliceNumbers.push_back(-1);
    m_MagnificationFactors.push_back(-6);
    m_Orientations.push_back(MIDAS_VIEW_UNKNOWN);
  }

  m_Layout = new QGridLayout(this);
  m_Layout->setObjectName(QString::fromUtf8("QmitkMIDASSingleViewWidget::m_Layout"));
  m_Layout->addWidget(m_RenderWindow, 0, 0);
}

QmitkMIDASSingleViewWidget::~QmitkMIDASSingleViewWidget()
{
}

void QmitkMIDASSingleViewWidget::SetContentsMargins(int margin)
{
  m_Layout->setContentsMargins(margin, margin, margin, margin);
}

void QmitkMIDASSingleViewWidget::SetSpacing(int spacing)
{
  m_Layout->setSpacing(spacing);
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

void QmitkMIDASSingleViewWidget::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  this->m_DataStorage = dataStorage;

  m_RenderWindow->GetRenderer()->SetDataStorage(dataStorage);
}

mitk::DataStorage::Pointer QmitkMIDASSingleViewWidget::GetDataStorage(mitk::DataStorage* dataStorage)
{
  return this->m_DataStorage;
}

QmitkMIDASRenderWindow* QmitkMIDASSingleViewWidget::GetRenderWindow() const
{
  return this->m_RenderWindow;
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

void QmitkMIDASSingleViewWidget::RemoveFromRenderingManager()
{
  m_RenderWindow->setEnabled(false);
  m_RenderWindow->setVisible(false);
  mitk::RenderingManager::GetInstance()->RemoveRenderWindow(this->m_RenderWindow->GetVtkRenderWindow());
}

void QmitkMIDASSingleViewWidget::AddToRenderingManager()
{
  mitk::RenderingManager::GetInstance()->AddRenderWindow(this->m_RenderWindow->GetVtkRenderWindow());
  m_RenderWindow->setEnabled(true);
  m_RenderWindow->setVisible(true);
}

void QmitkMIDASSingleViewWidget::InitializeGeometry(mitk::Geometry3D::Pointer geometry)
{
  // Store the geometry for later
  this->m_Geometry = geometry;

  // If we reset these variables, then code that works out orientation should re-initialize,
  // as it is like we have had no previous geometry information before.
  for (unsigned int i = 0; i < 3; i++)
  {
    m_SliceNumbers[i] = -1;
    m_MagnificationFactors[i] = -6;
    m_Orientations[i] = MIDAS_VIEW_UNKNOWN;
  }
  this->m_ViewOrientation = MIDAS_VIEW_UNKNOWN;
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

    // Update the current orientation, so the current settings are now considered previous settings.
    this->m_ViewOrientation = orientation;

    vtkRenderWindow* vtkWindow = m_RenderWindow->GetVtkRenderWindow();
    mitk::BaseRenderer::Pointer baseRenderer = mitk::BaseRenderer::GetInstance(vtkWindow);
    mitk::SliceNavigationController::Pointer controller = baseRenderer->GetSliceNavigationController();

    int slice = -1;
    if (m_SliceNumbers[orientation] == -1)
    {
      // No previous slice, so default to central slice.
      unsigned int steps = controller->GetSlice()->GetSteps();
      slice = (int)((steps - 1)/2);
    }
    else
    {
      slice = m_SliceNumbers[this->m_ViewOrientation];
    }

    // Set the view to the new orientation
    mitk::SliceNavigationController::ViewDirection direction = mitk::SliceNavigationController::Original;
    if (orientation == MIDAS_VIEW_AXIAL)
    {
      direction = mitk::SliceNavigationController::Transversal;
      controller->Update(direction, false, false, true);
    }
    else if (orientation == MIDAS_VIEW_SAGITTAL)
    {
      direction = mitk::SliceNavigationController::Sagittal;
      controller->Update(direction, true, true, false);
    }
    else if (orientation == MIDAS_VIEW_CORONAL)
    {
      direction = mitk::SliceNavigationController::Frontal;
      controller->Update(direction, true, true, false);
    }

    this->SetSliceNumber(slice);

    baseRenderer->GetDisplayGeometry()->Fit();
    mitk::RenderingManager::GetInstance()->RequestUpdate(vtkWindow);
  }
}

QmitkMIDASSingleViewWidget::MIDASViewOrientation QmitkMIDASSingleViewWidget::GetViewOrientation() const
{
  return this->m_ViewOrientation;
}

void QmitkMIDASSingleViewWidget::SetSliceNumber(int sliceNumber)
{
  vtkRenderWindow* vtkWindow = m_RenderWindow->GetVtkRenderWindow();
  mitk::BaseRenderer::Pointer baseRenderer = mitk::BaseRenderer::GetInstance(vtkWindow);
  mitk::SliceNavigationController::Pointer controller = baseRenderer->GetSliceNavigationController();

  controller->GetSlice()->SetPos(sliceNumber);
  this->m_SliceNumber = sliceNumber;

  mitk::RenderingManager::GetInstance()->RequestUpdate(vtkWindow);
}

int QmitkMIDASSingleViewWidget::GetSliceNumber() const
{
  return this->m_SliceNumber;
}

void QmitkMIDASSingleViewWidget::SetMagnificationFactor(int magnificationFactor)
{

}

int QmitkMIDASSingleViewWidget::GetMagnificationFactor() const
{
  return this->m_MagnificationFactor;
}

int QmitkMIDASSingleViewWidget::GetMinSlice() const
{
  return 0;
}

int QmitkMIDASSingleViewWidget::GetMaxSlice() const
{
  vtkRenderWindow* vtkWindow = m_RenderWindow->GetVtkRenderWindow();
  mitk::BaseRenderer::Pointer baseRenderer = mitk::BaseRenderer::GetInstance(vtkWindow);
  mitk::SliceNavigationController::Pointer controller = baseRenderer->GetSliceNavigationController();

  return controller->GetSlice()->GetSteps() -1;
}

int QmitkMIDASSingleViewWidget::GetMinMagnification() const
{
  return -5;
}

int QmitkMIDASSingleViewWidget::GetMaxMagnification() const
{
  return 20;
}

