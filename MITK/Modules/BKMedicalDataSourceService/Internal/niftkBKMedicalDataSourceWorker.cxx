/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBKMedicalDataSourceWorker.h"
#include <mitkLogMacros.h>
#include <mitkExceptionMacro.h>
#include <QThread>

namespace niftk
{

//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::BKMedicalDataSourceWorker()
: m_Timeout(1000)
{
  for (int i = 0; i < 256; i++)
  {
    m_DefaultLUT.push_back(qRgb(i, i, i));
  }
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::~BKMedicalDataSourceWorker()
{
  std::string message = "QUERY:GRAB_FRAME \"OFF\",30;";
  bool sentOK = this->SendCommandMessage(message);
  if (!sentOK)
  {
    MITK_ERROR << "Failed to send:" << message
               << ", but we are in destructor anyway.";
  }
  std::string response = this->ReceiveResponseMessage(4); // Should be ACK;
  if (response.empty())
  {
    MITK_ERROR << "Failed to parse response for:" << message
               << ", but we are in destructor anyway.";
  }
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ConnectToHost(QString address, int port)
{
  m_Socket.connectToHost(address, port);
  if (!m_Socket.waitForConnected(m_Timeout))
  {
    mitkThrow() << "Timed out while trying to connect to:" << address.toStdString() << ":" << port;
  }
  if (!m_Socket.isValid())
  {
    mitkThrow() << "Socket is not valid.";
  }
  if (!m_Socket.isOpen())
  {
    mitkThrow() << "Socket is not open.";
  }
  if (!m_Socket.isReadable())
  {
    mitkThrow() << "Socket is not readable.";
  }

  std::string message = "QUERY:US_WIN_SIZE;";
  bool sentOK = this->SendCommandMessage(message);
  if (!sentOK)
  {
    mitkThrow() << "Failed to send message '" << message
                << "'', to extract image size and socket status=" << m_Socket.errorString().toStdString();
  }

  std::string response = this->ReceiveResponseMessage(25);
  if (response.empty())
  {
    mitkThrow() << "Failed to parse response message.";
  }
  sscanf(response.c_str(), "DATA:US_WIN_SIZE %d,%d;", &m_ImageSize[0], &m_ImageSize[1]);
  MITK_INFO << "BK Medical image size:" << m_ImageSize[0] << ", " << m_ImageSize[1];

  if (m_ImageSize[0] < 1 || m_ImageSize[1] < 1)
  {
    mitkThrow() << "Invalid BK Medical image size.";
  }

  message = "QUERY:GRAB_FRAME \"ON\",2;";
  sentOK = this->SendCommandMessage(message);
  if (!sentOK)
  {
    mitkThrow() << "Failed to send message '" << message
                << "', to start streaming, socket status=" << m_Socket.errorString().toStdString();
  }
  response = this->ReceiveResponseMessage(4); // Should be ACK;
  if (response.empty())
  {
    MITK_ERROR << "Failed to parse response for:" << message
               << ", but we are in destructor anyway.";
  }
}


//-----------------------------------------------------------------------------
size_t BKMedicalDataSourceWorker::GenerateCommandMessage(const std::string& message)
{
  size_t counter = 0;
  m_OutgoingMessageBuffer[counter++] = 0x01;
  for (int i = 0; i < message.size(); i++)
  {
    m_OutgoingMessageBuffer[counter++] = message[i];
  }
  m_OutgoingMessageBuffer[counter++] = 0x04;
  return counter;
}


//-----------------------------------------------------------------------------
bool BKMedicalDataSourceWorker::SendCommandMessage(const std::string& message)
{
  size_t messageSize = this->GenerateCommandMessage(message);
  size_t sentSize = m_Socket.write(m_OutgoingMessageBuffer, messageSize);
  bool wasWritten = m_Socket.waitForBytesWritten(m_Timeout);

  bool isOK = true;
  if (sentSize != messageSize)
  {
    MITK_ERROR << "Failed to send message:" << message << " due to size mismatch:" << messageSize << " != " << sentSize;
    isOK = false;
  }
  if (!wasWritten)
  {
    MITK_ERROR << "Failed to send message:" << message << " due to socket error:" << m_Socket.errorString().toStdString();
    isOK = false;
  }

  MITK_INFO << "BKMedicalDataSourceWorker:sent:" << message << ", status:" << isOK;
  return isOK;
}


//-----------------------------------------------------------------------------
std::string BKMedicalDataSourceWorker::ReceiveResponseMessage(const size_t& expectedSize)
{
  std::string result;
  unsigned int counter = 0;
  size_t actualSize = expectedSize + 2; // due to start and end terminator.

  while(counter < actualSize)
  {
    qint64 bytesAvailable = m_Socket.bytesAvailable();
    if (bytesAvailable > 0)
    {
      QByteArray tmpData = m_Socket.readAll();
      if (tmpData.size() != bytesAvailable)
      {
        MITK_ERROR << "Failed to read " << bytesAvailable << " message bytes from socket.";
      }
      m_IntermediateBuffer.append(tmpData);
      if (m_IntermediateBuffer.size() >= actualSize)
      {
        const char* data = m_IntermediateBuffer.constData();

        while (counter < actualSize)
        {
          if (data[counter] != 0x01 && data[counter] != 0x04)
          {
            result += data[counter];
          }
          counter++;
        }
        m_IntermediateBuffer.remove(0, actualSize);
      }
    }
    else
    {
      if (!m_Socket.waitForReadyRead(-1))
      {
        MITK_ERROR << "Failed while waiting for socket, due to:" << m_Socket.errorString().toStdString();
      }
    }
  }

  if (result.size() != expectedSize)
  {
    MITK_ERROR << "Failed to read message of size:" << expectedSize;
  }

  MITK_INFO << "BKMedicalDataSourceWorker:received:" << result ;
  return result;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ReceiveImage(QImage& image)
{
  unsigned int minimumSize = m_ImageSize[0] * m_ImageSize[1] + 20;
  int preceedingChar = 0;
  int startingChar = 0;
  int startImageChar = 0;
  int endImageChar = 0;
  int terminatingChar = 0;
  int imageSize = 0;
  int dataSize = 0;

  bool imageAvailable = false;
  while(!imageAvailable)
  {
    qint64 bytesAvailable = m_Socket.bytesAvailable();
    if (bytesAvailable < minimumSize)
    {
      if (!m_Socket.waitForReadyRead(-1))
      {
        MITK_ERROR << "Failed while waiting on socket, due to:" << m_Socket.errorString().toStdString();
      }
    }
    else
    {
      QByteArray tmpData;
      tmpData.resize(bytesAvailable);

      qint64 bytesRead = m_Socket.read(tmpData.data(), bytesAvailable);

      if (bytesRead == bytesAvailable)
      {
        m_IntermediateBuffer.append(tmpData);

        preceedingChar = m_IntermediateBuffer.indexOf(0x01);
        startingChar = m_IntermediateBuffer.indexOf('#');
        startImageChar = startingChar + 4;
        terminatingChar = m_IntermediateBuffer.indexOf(0x04);
        endImageChar = terminatingChar - 2;
        dataSize =  terminatingChar - preceedingChar + 1;
        imageSize = endImageChar - startImageChar + 1;

        if (   preceedingChar >= 0
            && startingChar >= 0
            && startImageChar >= 0
            && endImageChar >= 0
            && terminatingChar >= 0
            && startingChar > preceedingChar
            && startImageChar > startingChar
            && endImageChar > startImageChar
            && terminatingChar > endImageChar
           )
        {
          if (   image.width() != m_ImageSize[0]
              || image.height() != m_ImageSize[1]
             )
          {
            // Image should either be grey scale or RGBA/ARGB ??? spec isnt so clear.
            if (imageSize < (m_ImageSize[0] * m_ImageSize[1] * 4))
            {
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
              QImage tmpImage(m_ImageSize[0], m_ImageSize[1], QImage::Format_Grayscale8);
#else
              QImage tmpImage(m_ImageSize[0], m_ImageSize[1], QImage::Format_Indexed8);
              tmpImage.setColorTable(m_DefaultLUT);
#endif
              image = tmpImage;
            }
            else
            {
              QImage tmpImage(m_ImageSize[0], m_ImageSize[1], QImage::Format_ARGB32);
              image = tmpImage;
            }
          }

          // Filling QImage with data from socket.
          // Assumes data is tightly packed.
          char *startImageData = &(m_IntermediateBuffer.data()[startImageChar]);
          char *endImageData = &(m_IntermediateBuffer.data()[endImageChar + 1]);
          char *rp = startImageData;
          char *wp = reinterpret_cast<char*>(image.bits());
          while (rp != endImageData)
          {
            if (*rp == 0x27)
            {
              rp++;            // skip escape char
              *wp = 255-(*rp); // invert it
            }
            else
            {
              *wp = *rp; // just copy
            }
            rp++;
            wp++;
          }

          m_IntermediateBuffer.remove(preceedingChar, dataSize);
          return;
        } // end extracting image
      }
    }
  }
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ReceiveImages()
{
  QImage image;
  while(true)
  {
    this->ReceiveImage(image);
    if (image.width() > 0 && image.height() > 0)
    {
      emit ImageReceived(image);
    }
  }
}

} // end namespace
