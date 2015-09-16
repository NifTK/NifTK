/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkVideoTestClient_h
#define QmitkVideoTestClient_h

#include "niftkIGIGuiExports.h"
#include <QWidget>
#include <QKeyEvent>
#include <QEvent>
#include <QString>
#include <vtkCalibratedModelRenderingPipeline.h>

/**
 * \class QmitkVideoTestClient
 * \brief Harness to grab images via OpenCV and send via NiftyLinkTcpClient.
 */
class NIFTKIGIGUI_EXPORT QmitkVideoTestClient : public QWidget
{
  Q_OBJECT

public:

  QmitkVideoTestClient(
    const std::string& hostname,
    const int& portNumber,
    const int& numberOfSeconds,
    QWidget *parent = 0
  );

  virtual ~QmitkVideoTestClient();

public slots:

  void Run();

private:

};

#endif
