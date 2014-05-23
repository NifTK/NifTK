/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkVideoPointPickingWidget_h
#define QmitkVideoPointPickingWidget_h

#include "niftkIGIGuiExports.h"
#include <QVTKWidget.h>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QEvent>
#include <QString>
#include <vtkImageViewer.h>
#include <vtkSmartPointer.h>
#include <mitkVideoTrackerMatching.h>
#include <mitkTrackingMatrixTimeStamps.h>

/**
 * \class QmitkVideoPointPickingWidget
 * \brief Very very prototypyish.. don't copy this one, copy QmitkUltrasoundPinCalibrationWidget instead.
 */
class NIFTKIGIGUI_EXPORT QmitkVideoPointPickingWidget : public QVTKWidget 
{
  Q_OBJECT

public:

  QmitkVideoPointPickingWidget(
    const QString& inputTrackerDirectory,
    const QString& inputVideoDirectory,
    const QString& outputPointDirectory,
    const unsigned long long timingToleranceInMilliseconds,
    const bool& skipForward,
    const bool& multiPointMode,
    const bool& skipExistingFrames,
    const unsigned int& samplingFrequency,
    QWidget *parent = 0
  );
  virtual ~QmitkVideoPointPickingWidget();

  /**
   * \brief When mouse is pressed, we store the 2D pixel location, in a single line, in a 
   * file carrying the time stamp of the video frame. 
   */
  virtual void mousePressEvent(QMouseEvent* event);
  
  /**
   * \brief Responds to key press events.
   *
   * Valid keys are
   * <pre>
   *   N = next image
   *   Q = quit application
   * </pre>
   */
  virtual void keyPressEvent(QKeyEvent* event);
  
  /**
   * \brief To Force the size of the window.
   */
  virtual void enterEvent(QEvent* event);

private slots:
  
private:

  const QString m_InputTrackerDirectory;
  const QString m_InputVideoDirectory;
  const QString m_OutputPointDirectory;
  const unsigned long long m_TimingToleranceInMilliseconds;
  const bool m_SkipForward;
  const bool m_MultiPointMode;
  const bool m_SkipExistingFrames;
  const unsigned int m_SamplingFrequency;

  vtkSmartPointer<vtkImageViewer> m_ImageViewer;
  mitk::VideoTrackerMatching::Pointer m_Matcher;
  mitk::TrackingMatrixTimeStamps m_TrackingTimeStamps;
  int m_ImageWidth;
  int m_ImageHeight;
  unsigned long int m_ImageFileCounter;
  unsigned long int m_PointsOutputCounter;

  void NextImage();
  void QuitApplication();
  void StorePoint(QMouseEvent* event);
  void ShowImage(const unsigned long int& imageNumber);
  void CreateDir(const std::string& dir);
};

#endif // QmitkVideoPointPickingWidget_h
