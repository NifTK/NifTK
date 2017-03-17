/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkBKMedicalDataSourceWorker_h
#define niftkBKMedicalDataSourceWorker_h

#include <QObject>
#include <QImage>
#include <QTcpSocket>

namespace niftk
{

/**
* \class BKMedicalDataSourceWorker
* \brief Main Worker object for BKMedicalDataSourceService that runs in
* a separate QThread.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class BKMedicalDataSourceWorker : public QObject
{

  Q_OBJECT

public:
  BKMedicalDataSourceWorker();
  ~BKMedicalDataSourceWorker();

  void ConnectToHost(QString address, int port);

public slots:

  void ReceiveImages();

signals:

  void ImageReceived(QImage);

private:

  QTcpSocket m_Socket;

};

} // end namespace

#endif
