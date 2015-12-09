/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkClientDataSourceService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker NiftyLinkClientDataSourceService::s_Lock;


//-----------------------------------------------------------------------------
NiftyLinkClientDataSourceService::NiftyLinkClientDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage
    )
: NiftyLinkDataSourceService(QString("NLClient-") + QString::number(s_Lock.GetNextSourceNumber()),
                             factoryName, properties, dataStorage)
{
  qRegisterMetaType<niftk::NiftyLinkMessageContainer::Pointer>("niftk::NiftyLinkMessageContainer::Pointer");

  int portNumber = 0;
  if(!properties.contains("port"))
  {
    mitkThrow() << "Port number not specified!";
  }
  portNumber = (properties.value("port")).toInt();

  QString hostName;
  if(!properties.contains("host"))
  {
    mitkThrow() << "Host name not specified!";
  }
  hostName = properties.value("host").toString();

  m_Client = new NiftyLinkTcpClient();

  bool ok = false;
  ok = QObject::connect(m_Client, SIGNAL(Connected(QString, int)), this, SLOT(OnConnected(QString, int)));
  assert(ok);
  ok = QObject::connect(m_Client, SIGNAL(Disconnected(QString, int)), this, SLOT(OnDisconnected(QString, int)));
  assert(ok);
  ok = QObject::connect(m_Client, SIGNAL(SocketError(QString, int, QAbstractSocket::SocketError, QString)), this, SLOT(OnSocketError(QString, int, QAbstractSocket::SocketError, QString)));
  assert(ok);
  ok = QObject::connect(m_Client, SIGNAL(ClientError(QString, int, QString)), this, SLOT(OnClientError(QString, int, QString)));
  assert(ok);
  ok = QObject::connect(m_Client, SIGNAL(MessageReceived(NiftyLinkMessageContainer::Pointer)), this, SLOT(OnMessageReceived(NiftyLinkMessageContainer::Pointer)), Qt::DirectConnection);
  assert(ok);

  m_Client->ConnectToHost(hostName, portNumber);

  QString deviceName = this->GetName();
  m_ClientNumber = (deviceName.remove(0, 9)).toInt(); // Should match string NLClient- above
}


//-----------------------------------------------------------------------------
NiftyLinkClientDataSourceService::~NiftyLinkClientDataSourceService()
{
  this->StopCapturing();

  bool ok = false;
  ok = QObject::disconnect(m_Client, SIGNAL(Connected(QString, int)), this, SLOT(OnConnected(QString, int)));
  assert(ok);
  ok = QObject::disconnect(m_Client, SIGNAL(Disconnected(QString, int)), this, SLOT(OnDisconnected(QString, int)));
  assert(ok);
  ok = QObject::disconnect(m_Client, SIGNAL(SocketError(QString, int, QAbstractSocket::SocketError, QString)), this, SLOT(OnSocketError(QString, int, QAbstractSocket::SocketError, QString)));
  assert(ok);
  ok = QObject::disconnect(m_Client, SIGNAL(ClientError(QString, int, QString)), this, SLOT(OnClientError(QString, int, QString)));
  assert(ok);
  ok = QObject::disconnect(m_Client, SIGNAL(MessageReceived(niftk::NiftyLinkMessageContainer::Pointer)), this, SLOT(OnMessageReceived(niftk::NiftyLinkMessageContainer::Pointer)));
  assert(ok);

  s_Lock.RemoveSource(m_ClientNumber);
}


//-----------------------------------------------------------------------------
void NiftyLinkClientDataSourceService::OnConnected(QString hostName, int portNumber)
{
  MITK_INFO << "Client OnConnected:" << hostName.toStdString() << ":" << portNumber;
}


//-----------------------------------------------------------------------------
void NiftyLinkClientDataSourceService::OnDisconnected(QString hostName, int portNumber)
{
  MITK_INFO << "Client OnDisconnected:" << hostName.toStdString() << ":" << portNumber;
}


//-----------------------------------------------------------------------------
void NiftyLinkClientDataSourceService::OnSocketError(QString hostName, int portNumber, QAbstractSocket::SocketError errorCode, QString errorString)
{
  MITK_INFO << "Client OnSocketError:" << hostName.toStdString() << ":" << portNumber << ", " << errorString.toStdString();

}


//-----------------------------------------------------------------------------
void NiftyLinkClientDataSourceService::OnClientError(QString hostName, int portNumber, QString errorString)
{
  MITK_INFO << "Client OnClientError:" << hostName.toStdString() << ":" << portNumber;
}


//-----------------------------------------------------------------------------
void NiftyLinkClientDataSourceService::OnMessageReceived(NiftyLinkMessageContainer::Pointer message)
{
  MITK_INFO << "Client, OnMessageReceived";
  this->MessageReceived(message);
}


} // end namespace
