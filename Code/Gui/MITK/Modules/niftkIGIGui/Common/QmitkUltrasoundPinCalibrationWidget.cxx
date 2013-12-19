/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkUltrasoundPinCalibrationWidget.h"
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>
#include <mitkCoordinateAxesData.h>
#include <vtkSmartPointer.h>
#include <mitkPointUtils.h>

//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::QmitkUltrasoundPinCalibrationWidget(
  const QString& inputTrackerDirectory,
  const QString& inputImageDirectory,
  const QString& outputMatrixDirectory,
  const QString& outputPointDirectory,  
  QObject *parent)
{
  setupUi(this);
  m_InputTrackerDirectory = inputTrackerDirectory;
  m_InputImageDirectory = inputImageDirectory;
  m_OutputMatrixDirectory = outputMatrixDirectory;
  m_OutputPointDirectory = outputPointDirectory;
}


//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::~QmitkUltrasoundPinCalibrationWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::mousePressEvent(QMouseEvent* event)
{
  std::cerr << "Matt, mouse pressed" << std::endl;  
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::keyPressEvent(QKeyEvent* event)
{
  std::cerr << "Matt, key pressed" << std::endl;    
}


