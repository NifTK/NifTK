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
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::~BKMedicalDataSourceWorker()
{
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
  m_OutgoingMessageBuffer[counter++]=0x04;
  return counter;
}


//-----------------------------------------------------------------------------
bool BKMedicalDataSourceWorker::SendCommandMessage(const std::string& message)
{
  size_t messageSize = this->GenerateCommandMessage(message);
  size_t sentSize = m_Socket.write(m_CommandMessageBuffer, messageSize);
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

  return isOK;
}


//-----------------------------------------------------------------------------
std::string BKMedicalDataSourceWorker::ReceiveResponseMessage(const size_t& expectedSize)
{
  std::string result;
  return result;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ReceiveImages()
{
  std::string message = "QUERY:US_WIN_SIZE;";
  bool sentOK = this->SendCommandMessage(message);
  MITK_INFO << "Sent:" << message << ", result=" << sentOK;
}

} // end namespace
