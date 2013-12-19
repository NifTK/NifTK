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

/**
 * \class QmitkUltrasoundPinCalibrationWidget
 * \brief Very prototypyish.. don't copy this one.
 */
class NIFTKIGIGUI_EXPORT QmitkUltrasoundPinCalibrationWidget : public QVTKWidget, public Ui_QmitkUltrasoundPinCalibrationWidget
{
  Q_OBJECT

public:

  QmitkUltrasoundPinCalibrationWidget(
    const QString& inputTrackerDirectory,
    const QString& inputImageDirectory,
    const QString& outputMatrixDirectory,
    const QString& outputPointDirectory,
    QObject *parent = 0
  );
  virtual ~QmitkUltrasoundPinCalibrationWidget();

private slots:

  virtual void mousePressEvent(QMouseEvent* event);
  virtual void keyPressEvent(QKeyEvent* event);
  
private:

  QString m_InputTrackerDirectory;
  QString m_InputImageDirectory;
  QString m_OutputMatrixDirectory;
  QString m_OutputPointDirectory;
};

#endif // QmitkUltrasoundPinCalibrationWidget_h
