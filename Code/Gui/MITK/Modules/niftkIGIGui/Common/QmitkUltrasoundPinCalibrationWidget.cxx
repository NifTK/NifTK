/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkUltrasoundPinCalibrationWidget.h"
#include <stdexcept>

//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::QmitkUltrasoundPinCalibrationWidget(
  const QString& inputTrackerDirectory,
  const QString& inputImageDirectory,
  const QString& outputMatrixDirectory,
  const QString& outputPointDirectory,  
  const unsigned int timingToleranceInMilliseconds,
  QWidget *parent)
: QVTKWidget(parent)
, m_TimingToleranceInMilliseconds(timingToleranceInMilliseconds)
{
  m_InputTrackerDirectory = inputTrackerDirectory;
  m_InputImageDirectory = inputImageDirectory;
  m_OutputMatrixDirectory = outputMatrixDirectory;
  m_OutputPointDirectory = outputPointDirectory;
  
  m_ImageViewer = vtkImageViewer::New();
  this->SetRenderWindow(m_ImageViewer->GetRenderWindow());
  m_ImageViewer->SetupInteractor(this->GetRenderWindow()->GetInteractor());
  m_ImageViewer->SetColorLevel(127.5);
  m_ImageViewer->SetColorWindow(255);  
  
  m_PNGReader = vtkPNGReader::New();
  m_ImageViewer->SetInputConnection(m_PNGReader->GetOutputPort());

  // Load all data, and set up the PNG reader to the first image.  
}


//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::~QmitkUltrasoundPinCalibrationWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::mousePressEvent(QMouseEvent* event)
{
  try 
  {
    this->StorePoint();
    event->accept();
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught std::exception:" << e.what();
    event->ignore();
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception:";
    event->ignore();
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::keyPressEvent(QKeyEvent* event)
{
  if (event->key() == Qt::Key_N)
  {
    this->NextImage();
    event->accept();    
  }
  else if (event->key() == Qt::Key_P)
  {
    this->PreviousImage();
    event->accept();
  }
  else if (event->key() == Qt::Key_Q)
  {
    this->QuitApplication();
    event->accept();
  }
  else
  {
    event->ignore();
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::QuitApplication()
{
  QApplication::exit(0);
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::NextImage()
{
  std::cerr << "Matt, next" << std::endl;  
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::PreviousImage()
{
  std::cerr << "Matt, previous" << std::endl;
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::StorePoint()
{
  std::cerr << "Matt, StorePoint" << std::endl;
}

