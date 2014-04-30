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
#include <vtkPNGReader.h>
#include <mitkTrackingMatrixTimeStamps.h>

/**
 * \class QmitkVideoPointPickingWidget
 * \brief Very prototypyish.. don't copy this one.
 */
class NIFTKIGIGUI_EXPORT QmitkVideoPointPickingWidget : public QVTKWidget 
{
  Q_OBJECT

public:

  QmitkVideoPointPickingWidget(
    const QString& inputTrackerDirectory,
    const QString& inputImageDirectory,
    const QString& outputMatrixDirectory,
    const QString& outputPointDirectory,
    const unsigned long long timingToleranceInMilliseconds,
    const bool& skipForward,
    const bool& multiPointMode,
    QWidget *parent = 0
  );
  virtual ~QmitkVideoPointPickingWidget();

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
  
  /**
   * \brief To Force the size of the window.
   */
  virtual void enterEvent(QEvent* event);

private slots:
  
private:

  const QString m_InputTrackerDirectory;
  const QString m_InputImageDirectory;
  const QString m_OutputMatrixDirectory;
  const QString m_OutputPointDirectory;
  const unsigned long long m_TimingToleranceInMilliseconds;
  const bool m_SkipForward;
  const bool m_MultiPointMode;

  vtkSmartPointer<vtkImageViewer> m_ImageViewer;
  vtkSmartPointer<vtkPNGReader> m_PNGReader;
  mitk::TrackingMatrixTimeStamps m_TrackingTimeStamps;
  std::vector<std::string> m_ImageFiles;
  int m_ImageWidth;
  int m_ImageHeight;
  unsigned long int m_ImageFileCounter;
  unsigned long int m_PointsOutputCounter;

  void NextImage();
  void PreviousImage();
  void QuitApplication();
  void StorePoint(QMouseEvent* event);
  void ShowImage(const unsigned long int& imageNumber);
  void CreateDir(const std::string& dir);
};

#endif // QmitkVideoPointPickingWidget_h
