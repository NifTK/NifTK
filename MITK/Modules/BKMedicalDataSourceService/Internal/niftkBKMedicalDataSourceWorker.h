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
#include <QByteArray>

namespace niftk
{

/**
* \class BKMedicalDataSourceWorker
* \brief Main Worker object for BKMedicalDataSourceService that runs in a separate QThread.
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
  void ErrorGenerated(QString);

private:

  size_t GenerateCommandMessage(const std::string& message);
  bool SendCommandMessage(const std::string& message);
  std::string ReceiveResponseMessage(const size_t& expectedSize);
  void ReceiveImage(QImage& image);

  int           m_Timeout;
  QTcpSocket    m_Socket;
  QByteArray    m_IntermediateBuffer;
  char          m_OutgoingMessageBuffer[256];
  char          m_ImageBuffer[1024*1024*4];
  int           m_ImageSize[2];
  QVector<QRgb> m_DefaultLUT;
};

} // end namespace

#endif
