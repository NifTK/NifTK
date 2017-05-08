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
#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::BKMedicalDataSourceWorker(const int& timeOut,
                                                     const int& framesPerSecond)
: m_Timeout(timeOut)
, m_FramesPerSecond(framesPerSecond)
, m_Socket(nullptr)
, m_RequestStopStreaming(false)
, m_IsStreaming(false)
{
  for (int i = 0; i < 256; i++)
  {
    m_DefaultLUT.push_back(qRgb(i, i, i));
  }
  m_Socket = new QTcpSocket(this);
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::~BKMedicalDataSourceWorker()
{
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::DisconnectFromHost()
{
  if (m_Socket->state() == QTcpSocket::ConnectedState)
  {
    m_Socket->disconnectFromHost();
  }
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::RequestStop()
{
  m_RequestStopStreaming = true;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::StopStreaming()
{
  std::ostringstream message;
  message << "QUERY:GRAB_FRAME \"OFF\"," << m_FramesPerSecond << ";";
  bool sentOK = this->SendCommandMessage(message.str());
  if (!sentOK)
  {
    MITK_ERROR << "Failed to send:" << message.str()
               << ", but we are stopping anyway.";
  }

  m_IsStreaming = false;
  m_RequestStopStreaming = false;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::StartStreaming()
{
  std::ostringstream streamMessage;
  streamMessage << "QUERY:GRAB_FRAME \"ON\"," << m_FramesPerSecond << ";";
  bool sentOK = this->SendCommandMessage(streamMessage.str());
  if (!sentOK)
  {
    mitkThrow() << "Failed to send message '" << streamMessage.str()
                << "', to start streaming, socket status=" << m_Socket->errorString().toStdString();
  }
  std::string response = this->ReceiveResponseMessage(4); // Should be ACK;
  if (response.empty())
  {
    mitkThrow() << "Failed to parse acknowledgement to turn streaming on.";
  }

  m_IsStreaming = true;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ConnectToHost(const QString& address, const int& port)
{
  m_Socket->connectToHost(address, port);
  if (!m_Socket->waitForConnected(m_Timeout))
  {
    mitkThrow() << "Timed out while trying to connect to:" << address.toStdString() << ":" << port;
  }
  if (!m_Socket->isValid())
  {
    mitkThrow() << "Socket is not valid.";
  }
  if (!m_Socket->isOpen())
  {
    mitkThrow() << "Socket is not open.";
  }
  if (!m_Socket->isReadable())
  {
    mitkThrow() << "Socket is not readable.";
  }

  std::string message = "QUERY:US_WIN_SIZE;";
  bool sentOK = this->SendCommandMessage(message);
  if (!sentOK)
  {
    mitkThrow() << "Failed to send message '" << message
                << "'', to extract image size and socket status=" << m_Socket->errorString().toStdString();
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
}


//-----------------------------------------------------------------------------
int BKMedicalDataSourceWorker::FindFirstANotPreceededByB(const int& startingPosition,
                                                         const QByteArray& buf,
                                                         const char& a,
                                                         const char& b)
{
  int indexOf = 0;
  int sp = startingPosition;

  while (indexOf != -1)
  {
    indexOf = buf.indexOf(a, sp);
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
        sp = indexOf + 1;
      }
    }
  }
  return indexOf;
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
  size_t sentSize = m_Socket->write(m_OutgoingMessageBuffer, messageSize);
  bool wasWritten = m_Socket->waitForBytesWritten(m_Timeout);

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
               << " due to socket error:" << m_Socket->errorString().toStdString();
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
    qint64 bytesAvailable = m_Socket->bytesAvailable();
    if (bytesAvailable > 0)
    {
      QByteArray tmpData = m_Socket->readAll();
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
      if (!m_Socket->waitForReadyRead(-1))
      {
        MITK_ERROR << "Failed while waiting for socket, due to:" << m_Socket->errorString().toStdString();
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
  // According to spec, reply is:
  // DATA:GRAB_FRAME #6227332<b><b><b><b> …;
  unsigned int minimumSize = m_ImageSize[0] * m_ImageSize[1] + 22;
  int preceedingChar = 0;
  int hashChar = 0;
  int sizeOfDataChar = 0;
  int startImageChar = 0;
  int endImageChar = 0;
  int terminatingChar = 0;
  int imageSize = 0;
  int dataSize = 0;

  bool imageAvailable = false;
  while(!imageAvailable)
  {
    qint64 bytesAvailable = m_Socket->bytesAvailable();
    if (bytesAvailable < minimumSize)
    {
      if (!m_Socket->waitForReadyRead(-1))
      {
        MITK_ERROR << "Failed while waiting on socket, due to:" << m_Socket->errorString().toStdString();
      }
    }
    else
    {
      QByteArray tmpData;
      tmpData.resize(bytesAvailable);

      qint64 bytesRead = m_Socket->read(tmpData.data(), bytesAvailable);

      if (bytesRead == bytesAvailable)
      {
        m_IntermediateBuffer.append(tmpData);

        preceedingChar = this->FindFirstANotPreceededByB(0,
                                                         m_IntermediateBuffer,
                                                         0x01,
                                                         0x27);

        if (preceedingChar >= 0)
        {
          terminatingChar = this->FindFirstANotPreceededByB(preceedingChar,
                                                            m_IntermediateBuffer,
                                                            0x04,
                                                            0x27);

          if (terminatingChar > preceedingChar)
          {

            int imageMessageIndex = m_IntermediateBuffer.indexOf("DATA:GRAB_FRAME", preceedingChar+1);
            if (   imageMessageIndex != -1             // i.e. it was found
                && imageMessageIndex > preceedingChar  // it was after the preceeding char
                && imageMessageIndex < terminatingChar // and before terminating char (i.e. not in a subsequent message).
               )
            {
              hashChar = m_IntermediateBuffer.indexOf('#', preceedingChar);

              sizeOfDataChar = hashChar + 1;

              startImageChar = sizeOfDataChar
                             + (m_IntermediateBuffer[sizeOfDataChar] - '0') // as we are dealing with ASCII codes.
                             + 1  // to move onto next char
                             + 4; // timestamp = 4 bytes.


              endImageChar = terminatingChar - 2;
              dataSize =  terminatingChar - preceedingChar + 1;
              imageSize = endImageChar - startImageChar + 1;

              if (   startImageChar >= 0
                  && endImageChar > startImageChar
                  && imageSize > 0
                  && dataSize > 0
                  && dataSize > imageSize
                  && (    imageSize == m_ImageSize[0]*m_ImageSize[1]
                       || imageSize == m_ImageSize[0]*m_ImageSize[1]*4
                     ) // image must be grey-scale or RGBA.
                 )
              {
                // This means our argument QImage has not been initialised yet.
                if (   image.width()  != m_ImageSize[0]
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
                unsigned char uc = 0;
                unsigned char ucp1 = 0;
                unsigned char uc1 = 1;
                unsigned char uc4 = 4;
                unsigned char uc27 = 27;
                unsigned char ucN1 = ~uc1;
                unsigned char ucN4 = ~uc4;
                unsigned char ucN27 = ~uc27;

                while (rp != endImageData)
                {
                  uc = *(reinterpret_cast<unsigned char*>(rp));
                  ucp1 = *(reinterpret_cast<unsigned char*>(rp) + 1);

                  // See page 9 of 142 in BK doc PS12640-44
                  if (   (uc == uc27 && ucp1 == ucN1)
                      || (uc == uc27 && ucp1 == ucN4)
                      || (uc == uc27 && ucp1 == ucN27)
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

                m_IntermediateBuffer.remove(0, terminatingChar+1);
                return;
              }
              else
              {
                MITK_WARN << "Received an image message, but it was the wrong size.";
                m_IntermediateBuffer.remove(0, terminatingChar+1);
              }
            }
            else
            {
              MITK_WARN << "Received a non-image message, which I wasn't expecting.";
              m_IntermediateBuffer.remove(0, terminatingChar+1);
            }
          }
          else
          {
            MITK_DEBUG << "Failed to find end of message character. This is OK if message is still incoming.";
          }
        }
        else
        {
          MITK_WARN << "Failed to find start of message character. This suggests there is junk in the buffer.";
          m_IntermediateBuffer.clear();
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::Start()
{
  // The Start() method is called from a non-GUI thread.
  // So we start the streaming inside this method.
  this->StartStreaming();

  while(m_IsStreaming)
  {
    {
      // If another thread (e.g. GUI) has requested to stop,
      // we send this stop request, which ultimately sets m_IsStreaming
      // to false, thereby ending this loop.
      if (m_RequestStopStreaming)
      {
        this->StopStreaming();
      }
      else
      {
        // This blocks if no data, so effectively this thread waits.
        this->ReceiveImage(m_Image);

        // Signal to the BKMedicalDataSourceService to do something with this new data.
        if (m_Image.width() > 0 && m_Image.height() > 0)
        {
          emit ImageReceived(m_Image);
        }
      }
    } // scope for QMutexLocker
  }

  this->DisconnectFromHost();
  emit Finished();
}

} // end namespace
