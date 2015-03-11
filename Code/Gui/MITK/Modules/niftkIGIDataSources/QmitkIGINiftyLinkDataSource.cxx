/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINiftyLinkDataSource.h"

//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSource::QmitkIGINiftyLinkDataSource(mitk::DataStorage* storage, niftk::NiftyLinkTcpServer* server)
: QmitkIGIDataSource(storage)
, m_Server(server)
, m_ClientDescriptor(NULL)
{
  qRegisterMetaType<niftk::NiftyLinkMessageContainer::Pointer>("niftk::NiftyLinkMessageContainer::Pointer");

  if (m_Server == NULL)
  {
    m_Server = new niftk::NiftyLinkTcpServer();
    m_UsingSomeoneElsesServer = false;
  }
  else
  {
    m_UsingSomeoneElsesServer = true;
  }
  connect(m_Server, SIGNAL(ClientConnected(int)), this, SLOT(ClientConnected()));
  connect(m_Server, SIGNAL(ClientDisconnected(int)), this, SLOT(ClientDisconnected()));
  connect(m_Server, SIGNAL(MessageReceived(int, niftk::NiftyLinkMessageContainer::Pointer)), this, SLOT(InterpretMessage(int, niftk::NiftyLinkMessageContainer::Pointer)));
  if (m_Server != NULL)
  {
    this->ClientConnected();
  }
}


//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSource::~QmitkIGINiftyLinkDataSource()
{
  if ( m_UsingSomeoneElsesServer )
  {
    m_Server = NULL;
  }
  if (m_Server != NULL )
  {
    delete m_Server;
  }
  if (m_ClientDescriptor != NULL)
  {
    delete m_ClientDescriptor;
  }
}


//-----------------------------------------------------------------------------
int QmitkIGINiftyLinkDataSource::GetPort() const
{
  int result = -1;
  if (m_Server != NULL)
  {
    result = m_Server->serverPort();
  }
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkIGINiftyLinkDataSource::ListenOnPort(int portNumber)
{
  bool isListening = m_Server->listen(QHostAddress::LocalHost, portNumber);
  if (isListening)
  {
    this->SetStatus("Listening");
  }
  else
  {
    this->SetStatus("Listening Failed");
  }
  emit DataSourceStatusUpdated(this->GetIdentifier());
  return isListening;
}


//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSource::ClientConnected()
{
  this->SetStatus("Client Connected");
  emit DataSourceStatusUpdated(this->GetIdentifier());
}


//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSource::ClientDisconnected()
{
  this->SetStatus("Listening");
  emit DataSourceStatusUpdated(this->GetIdentifier());
}


//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSource::ProcessClientInfo(niftk::NiftyLinkClientDescriptor* clientInfo)
{
  this->SetClientDescriptor(clientInfo);

  this->SetName(clientInfo->GetDeviceName().toStdString());
  this->SetType(clientInfo->GetDeviceType().toStdString());

  QString descr = QString("Address=") +  clientInfo->GetClientIP()
      + QString(":") + clientInfo->GetClientPort();
  
  // Don't set description for trackers
  if ( clientInfo->GetDeviceType() != "Tracker" ) 
    this->SetDescription(descr.toStdString());

  QString deviceInfo;
  deviceInfo.append("Client connected:");
  deviceInfo.append("  Device name: ");
  deviceInfo.append(clientInfo->GetDeviceName());
  deviceInfo.append("\n");

  deviceInfo.append("  Device type: ");
  deviceInfo.append(clientInfo->GetDeviceType());
  deviceInfo.append("\n");

  deviceInfo.append("  Communication type: ");
  deviceInfo.append(clientInfo->GetCommunicationType());
  deviceInfo.append("\n");

  deviceInfo.append("  Port name: ");
  deviceInfo.append(clientInfo->GetPortName());
  deviceInfo.append("\n");

  deviceInfo.append("  Client ip: ");
  deviceInfo.append(clientInfo->GetClientIP());
  deviceInfo.append("\n");

  deviceInfo.append("  Client port: ");
  deviceInfo.append(clientInfo->GetClientPort());
  deviceInfo.append("\n");

  qDebug() << deviceInfo;
  emit DataSourceStatusUpdated(this->GetIdentifier());
}
