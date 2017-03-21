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
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::BKMedicalDataSourceWorker(const int& timeOut, const int& framesPerSecond)
: m_Lock(QMutex::Recursive)
, m_Timeout(timeOut)
, m_FramesPerSecond(framesPerSecond)
, m_RequestStopStreaming(false)
, m_IsStreaming(false)
{
  for (int i = 0; i < 256; i++)
  {
    m_DefaultLUT.push_back(qRgb(i, i, i));
  }
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::~BKMedicalDataSourceWorker()
{
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::RequestStopStreaming()
{
  QMutexLocker locker(&m_Lock);

  m_RequestStopStreaming = true;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::StopStreaming()
{
  QMutexLocker locker(&m_Lock);

  std::ostringstream message;
  message << "QUERY:GRAB_FRAME \"OFF\"," << m_FramesPerSecond << ";";
  bool sentOK = this->SendCommandMessage(message.str());
  if (!sentOK)
  {
    MITK_ERROR << "Failed to send:" << message.str()
               << ", but we are stopping anyway.";
  }
/*
 * I think we can ignore this for now. The incoming buffer will
 * be queued up with images. So, we would have to process all images,
 * and then hunt for an ACK. But we don't really care, as this is only
 * called when shutting down the service.
 *
  std::string response = this->ReceiveResponseMessage(4); // Should be ACK;
  if (response.empty())
  {
    MITK_ERROR << "Failed to parse response for:" << message.str()
               << ", but we are stopping anyway.";
  }
*/
  m_IsStreaming = false;
  m_RequestStopStreaming = false;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ConnectToHost(QString address, int port)
{
  QMutexLocker locker(&m_Lock);

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

  std::ostringstream streamMessage;
  streamMessage << "QUERY:GRAB_FRAME \"ON\"," << m_FramesPerSecond << ";";
  sentOK = this->SendCommandMessage(streamMessage.str());
  if (!sentOK)
  {
    mitkThrow() << "Failed to send message '" << streamMessage.str()
                << "', to start streaming, socket status=" << m_Socket.errorString().toStdString();
  }
  response = this->ReceiveResponseMessage(4); // Should be ACK;
  if (response.empty())
  {
    mitkThrow() << "Failed to parse acknowledgement to turn streaming on.";
  }

  m_IsStreaming = true;
}


//-----------------------------------------------------------------------------
int BKMedicalDataSourceWorker::FindFirstANotPreceededByB(const QByteArray& buf,
                                                         const char& a,
                                                         const char& b)
{
  int indexOf = 0;
  int startingPosition = 0;

  while (indexOf != -1)
  {
    indexOf = buf.indexOf(a, startingPosition);
    if (indexOf != -1)
    {
      // first character, can't be preceeded by 'b', so is valid.
      if (indexOf == 0)
      {
        return indexOf;
      }
      // If last character is preceeded by 'b', no point continuing.
      else if (indexOf == buf.size() - 1 && buf.at(indexOf-1) == b)
      {
        return -1;
      }
      // This is the valid case: 'a' not preceeded by 'b'.
      else if (indexOf != 0 && buf.at(indexOf-1) != b)
      {
        return indexOf;
      }
      else
      {
        // Keep searching from the next position.
        startingPosition = indexOf + 1;
      }
    }
  }
  return indexOf;
}


//-----------------------------------------------------------------------------
size_t BKMedicalDataSourceWorker::GenerateCommandMessage(const std::string& message)
{
  QMutexLocker locker(&m_Lock);

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
  QMutexLocker locker(&m_Lock);

  size_t messageSize = this->GenerateCommandMessage(message);
  size_t sentSize = m_Socket.write(m_OutgoingMessageBuffer, messageSize);
  bool wasWritten = m_Socket.waitForBytesWritten(m_Timeout);

  bool isOK = true;
  if (sentSize != messageSize)
  {
    MITK_ERROR << "Failed to send message:" << message
               << " due to size mismatch:" << messageSize << " != " << sentSize;
    isOK = false;
  }
  if (!wasWritten)
  {
    MITK_ERROR << "Failed to send message:" << message
               << " due to socket error:" << m_Socket.errorString().toStdString();
    isOK = false;
  }

  MITK_INFO << "BKMedicalDataSourceWorker:sent:" << message << ", status:" << isOK;
  return isOK;
}


//-----------------------------------------------------------------------------
std::string BKMedicalDataSourceWorker::ReceiveResponseMessage(const size_t& expectedSize)
{
  QMutexLocker locker(&m_Lock);

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

  MITK_INFO << "BKMedicalDataSourceWorker:received:" << result;
  return result;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ReceiveImage(QImage& image)
{
  QMutexLocker locker(&m_Lock);

  unsigned int minimumSize = m_ImageSize[0] * m_ImageSize[1] + 20;
  int preceedingChar = 0;
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

        preceedingChar = this->FindFirstANotPreceededByB(m_IntermediateBuffer,
                                                         0x01,
                                                         0x27);
        startImageChar = preceedingChar + 5;
        terminatingChar = this->FindFirstANotPreceededByB(m_IntermediateBuffer,
                                                          0x04,
                                                          0x27);
        endImageChar = terminatingChar - 2;
        dataSize =  terminatingChar - preceedingChar + 1;
        imageSize = endImageChar - startImageChar + 1;

        if (   preceedingChar >= 0
            && startImageChar >= 0
            && endImageChar >= 0
            && terminatingChar >= 0
            && dataSize > 0
            && imageSize > 0
            && endImageChar > startImageChar
            && terminatingChar > preceedingChar
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
            // See page 9 of 142 in BK doc PS12640-44
            if (   (*rp == 0x27 && *(rp+1) == ~(0x1))
                || (*rp == 0x27 && *(rp+1) == ~(0x4))
                || (*rp == 0x27 && *(rp+1) == ~(0x27))
               )
            {
              rp++;         // skip escape char
              *wp = ~(*rp); // invert data
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
  while(m_IsStreaming)
  {
    {
      QMutexLocker locker(&m_Lock);

      // This blocks if no data, so effectively this thread waits.
      this->ReceiveImage(image);
      if (image.width() > 0 && image.height() > 0)
      {
        emit ImageReceived(image);
      }

      // If another thread (e.g. GUI) has requested to stop,
      // we send this stop request, which ultimately sets m_IsStreaming to false.
      if (m_RequestStopStreaming)
      {
        this->StopStreaming();
      }
    }
  }
}

} // end namespace
