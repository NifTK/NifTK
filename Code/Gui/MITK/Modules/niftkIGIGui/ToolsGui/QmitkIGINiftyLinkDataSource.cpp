/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkIGINiftyLinkDataSource.h"

//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSource::QmitkIGINiftyLinkDataSource()
: m_Socket(NULL)
, m_ClientDescriptor(NULL)
, m_UsingSomeoneElsesSocket(false)
{
  m_Socket = new OIGTLSocketObject();
  connect(m_Socket, SIGNAL(clientConnectedSignal()), this, SLOT(ClientConnected()));
  connect(m_Socket, SIGNAL(clientDisconnectedSignal()), this, SLOT(ClientDisconnected()));
  connect(m_Socket, SIGNAL(messageReceived(OIGTLMessage::Pointer )), this, SLOT(InterpretMessage(OIGTLMessage::Pointer )), Qt::DirectConnection);
}
//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSource::QmitkIGINiftyLinkDataSource(OIGTLSocketObject *socket)
: m_Socket(socket)
, m_ClientDescriptor(NULL)
, m_UsingSomeoneElsesSocket(true)
{
  connect(m_Socket, SIGNAL(clientConnectedSignal()), this, SLOT(ClientConnected()));
  connect(m_Socket, SIGNAL(clientDisconnectedSignal()), this, SLOT(ClientDisconnected()));
  connect(m_Socket, SIGNAL(messageReceived(OIGTLMessage::Pointer )), this, SLOT(InterpretMessage(OIGTLMessage::Pointer )));
}



//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSource::~QmitkIGINiftyLinkDataSource()
{
  if (m_Socket != NULL && ! m_UsingSomeoneElsesSocket)
  {
    delete m_Socket;
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
  if (m_Socket != NULL)
  {
    result = m_Socket->getPort();
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSource::SendMessage(OIGTLMessage::Pointer msg)
{
  if (m_Socket != NULL)
  {
    m_Socket->sendMessage(msg);
  }
}


//-----------------------------------------------------------------------------
bool QmitkIGINiftyLinkDataSource::ListenOnPort(int portNumber)
{
  bool isListening = m_Socket->listenOnPort(portNumber);
  if (isListening)
  {
    this->SetStatus("Listening");
  }
  else
  {
    this->SetStatus("Listening Failed");
  }
  DataSourceStatusUpdated.Send(this->GetIdentifier());
  return isListening;
}

//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSource::ClientConnected()
{
  this->SetStatus("Client Connected");
  DataSourceStatusUpdated.Send(this->GetIdentifier());
}


//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSource::ClientDisconnected()
{
  this->SetStatus("Listening");
  DataSourceStatusUpdated.Send(this->GetIdentifier());
}


//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSource::ProcessClientInfo(ClientDescriptorXMLBuilder* clientInfo)
{
  this->SetClientDescriptor(clientInfo);

  this->SetName(clientInfo->getDeviceName().toStdString());
  this->SetType(clientInfo->getDeviceType().toStdString());

  QString descr = QString("Address=") +  clientInfo->getClientIP()
      + QString(":") + clientInfo->getClientPort();
  
  //Don't set description for trackers
  if ( clientInfo->getDeviceType() != "Tracker" ) 
    this->SetDescription(descr.toStdString());

  QString deviceInfo;
  deviceInfo.append("Client connected:");
  deviceInfo.append("  Device name: ");
  deviceInfo.append(clientInfo->getDeviceName());
  deviceInfo.append("\n");

  deviceInfo.append("  Device type: ");
  deviceInfo.append(clientInfo->getDeviceType());
  deviceInfo.append("\n");

  deviceInfo.append("  Communication type: ");
  deviceInfo.append(clientInfo->getCommunicationType());
  deviceInfo.append("\n");

  deviceInfo.append("  Port name: ");
  deviceInfo.append(clientInfo->getPortName());
  deviceInfo.append("\n");

  deviceInfo.append("  Client ip: ");
  deviceInfo.append(clientInfo->getClientIP());
  deviceInfo.append("\n");

  deviceInfo.append("  Client port: ");
  deviceInfo.append(clientInfo->getClientPort());
  deviceInfo.append("\n");

  qDebug() << deviceInfo;
  DataSourceStatusUpdated.Send(this->GetIdentifier());
}
//-----------------------------------------------------------------------------
OIGTLSocketObject* QmitkIGINiftyLinkDataSource::GetSocket()
{
  return this->m_Socket;
}
