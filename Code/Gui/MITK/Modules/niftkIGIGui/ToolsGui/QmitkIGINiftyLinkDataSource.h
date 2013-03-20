/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGINIFTYLINKDATASOURCE_H
#define QMITKIGINIFTYLINKDATASOURCE_H

#include "niftkIGIGuiExports.h"
#include "QmitkIGIDataSource.h"
#include <NiftyLinkSocketObject.h>
#include <Common/NiftyLinkXMLBuilder.h>

/**
 * \class QmitkIGINiftyLinkDataSource
 * \brief Base class for IGI Data Sources that are receiving networked input
 * from NiftyLink. NiftyLink uses Qt, so this class is in the Qt library, and named
 * Qmitk.
 */
class NIFTKIGIGUI_EXPORT QmitkIGINiftyLinkDataSource : public QmitkIGIDataSource
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGINiftyLinkDataSource, QmitkIGIDataSource);

  /**
   * \brief Sets the socket pointer.
   */
  itkSetObjectMacro(Socket, NiftyLinkSocketObject);

  /**
   * \brief Gets the socket pointer.
   */
  itkGetConstMacro(Socket, NiftyLinkSocketObject*);

  /**
   * \brief Sets the Client Descriptor XML.
   */
  itkSetObjectMacro(ClientDescriptor, ClientDescriptorXMLBuilder);

  /**
   * \brief Gets the Client Descriptor XML.
   */
  itkGetConstMacro(ClientDescriptor, ClientDescriptorXMLBuilder*);

  /**
   * \brief Returns the port number that this tool is using or -1 if no socket is available.
   */
  int GetPort() const;

  /**
   * \brief Tells this object to start listening on a given port number.
   */
  bool ListenOnPort(int portNumber);

  /**
   * \brief If there is a socket associated with this tool, will send the message.
   */
  void SendMessage(NiftyLinkMessage::Pointer msg);

  /**
   * \brief Get the Associated Socket
   */
  NiftyLinkSocketObject* GetSocket();

protected:

  QmitkIGINiftyLinkDataSource(); // Purposefully hidden.
  QmitkIGINiftyLinkDataSource(NiftyLinkSocketObject *socket); // Purposefully hidden.
  virtual ~QmitkIGINiftyLinkDataSource(); // Purposefully hidden.

  QmitkIGINiftyLinkDataSource(const QmitkIGINiftyLinkDataSource&); // Purposefully not implemented.
  QmitkIGINiftyLinkDataSource& operator=(const QmitkIGINiftyLinkDataSource&); // Purposefully not implemented.

  /**
   * \brief When client information is received we update the DataSource member variables, and dump info to console.
   */
  void ProcessClientInfo(ClientDescriptorXMLBuilder* clientInfo);

protected slots:

  /**
   * \brief Slot called when socket connects.
   */
  virtual void ClientConnected();

  /**
   * \brief Slot called when socket disconnects.
   */
  virtual void ClientDisconnected();

  /**
   * \brief Main message handler routine for this tool, that subclasses must implement.
   */
  virtual void InterpretMessage(NiftyLinkMessage::Pointer msg) {};

private:

  NiftyLinkSocketObject       *m_Socket;
  ClientDescriptorXMLBuilder  *m_ClientDescriptor;
  bool                         m_UsingSomeoneElsesSocket;

}; // end class

#endif

