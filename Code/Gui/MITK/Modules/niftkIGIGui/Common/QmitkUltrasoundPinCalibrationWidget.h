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
#include <QVTKWidget.h>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QEvent>
#include <QString>
#include <vtkImageViewer.h>
#include <vtkSmartPointer.h>
#include <vtkImageReader2.h>

/**
 * \class QmitkUltrasoundPinCalibrationWidget
 * \brief Very prototypyish.. don't copy this one.
 */
class NIFTKIGIGUI_EXPORT QmitkUltrasoundPinCalibrationWidget : public QVTKWidget 
{
  Q_OBJECT

public:

  QmitkUltrasoundPinCalibrationWidget(
    const QString& inputImageDirectory,
    const QString& outputPointDirectory,
    QWidget *parent = 0
  );
  virtual ~QmitkUltrasoundPinCalibrationWidget();

  /**
   * \brief When mouse is pressed, we store the timestamp, 2D pixel location, in a single line, in a new file, whose filename is timestamp.
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

  void SetPNG ( bool );

private slots:
  
private:

  const QString m_InputImageDirectory;
  const QString m_OutputPointDirectory;

  vtkSmartPointer<vtkImageViewer> m_ImageViewer;
  vtkSmartPointer<vtkImageReader2> m_ImageReader;
  std::vector<std::string> m_ImageFiles;
  int m_ImageWidth;
  int m_ImageHeight;
  unsigned long int m_ImageFileCounter;
  unsigned long int m_PointsOutputCounter;
  bool m_PNG; //by default we look for nii, but we can read png

  void NextImage();
  void PreviousImage();
  void QuitApplication();
  void StorePoint(QMouseEvent* event);
  void ShowImage(const unsigned long int& imageNumber);
  void CreateDir(const std::string& dir);
};

#endif // QmitkUltrasoundPinCalibrationWidget_h
