/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkUltrasoundPinCalibrationWidget_h
#define QmitkUltrasoundPinCalibrationWidget_h

#include "niftkIGIGuiExports.h"
#include "ui_QmitkUltrasoundPinCalibrationWidget.h"
#include <QVTKWidget.h>
#include <QString>
#include <QKeyEvent>
#include <vtkImageViewer.h>
#include <vtkSmartPointer.h>
#include <vtkPNGReader.h>

/**
 * \class QmitkUltrasoundPinCalibrationWidget
 * \brief Very prototypyish.. don't copy this one.
 */
class NIFTKIGIGUI_EXPORT QmitkUltrasoundPinCalibrationWidget : public QVTKWidget 
{
  Q_OBJECT

public:

  QmitkUltrasoundPinCalibrationWidget(
    const QString& inputTrackerDirectory,
    const QString& inputImageDirectory,
    const QString& outputMatrixDirectory,
    const QString& outputPointDirectory,
    const unsigned int timingToleranceInMilliseconds,
    QWidget *parent = 0
  );
  virtual ~QmitkUltrasoundPinCalibrationWidget();

  /**
   * \brief When mouse is pressed, we store the 2D pixel location, in a single line, in a new file.
   * 
   * This will only save if the timing difference between an ultrasound image and the tracking data
   * is within a certain tolerance. An error message is raised if there is no tracking information.
   */
  virtual void mousePressEvent(QMouseEvent* event);
  
  /**
   * \brief Responds to key press events.
   *
   * Valid keys are
   * <pre>
   *   N = next image
   *   P = previous image
   *   Q = quit application
   * </pre>
   */
  virtual void keyPressEvent(QKeyEvent* event);
  
private slots:
  
private:

  QString m_InputTrackerDirectory;
  QString m_InputImageDirectory;
  QString m_OutputMatrixDirectory;
  QString m_OutputPointDirectory;
  vtkSmartPointer<vtkImageViewer> m_ImageViewer;
  vtkSmartPointer<vtkPNGReader> m_PNGReader;
  const unsigned int m_TimingToleranceInMilliseconds;
  
  void NextImage();
  void PreviousImage();
  void QuitApplication();
  void StorePoint();
};

#endif // QmitkUltrasoundPinCalibrationWidget_h
