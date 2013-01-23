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

#ifndef QMITKIGINIFTYLINKDATASOURCE_H
#define QMITKIGINIFTYLINKDATASOURCE_H

#include "niftkIGIGuiExports.h"
#include "QmitkIGIDataSource.h"
#include <OIGTLSocketObject.h>
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
  itkSetObjectMacro(Socket, OIGTLSocketObject);

  /**
   * \brief Gets the socket pointer.
   */
  itkGetConstMacro(Socket, OIGTLSocketObject*);

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
   * \brief Tells this object to start listening on a given port number, using an 
   * existing socket
   */
  bool ListenOnPort(igtl::Socket::Pointer socket, int portNumber);

  /**
   * \brief If there is a socket associated with this tool, will send the message.
   */
  void SendMessage(OIGTLMessage::Pointer msg);

  /**
   * \brief Get the Associated Socket
   */
  OIGTLSocketObject* GetSocket();

protected:

  QmitkIGINiftyLinkDataSource(); // Purposefully hidden.
  QmitkIGINiftyLinkDataSource(OIGTLSocketObject *socket); // Purposefully hidden.
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
  virtual void InterpretMessage(OIGTLMessage::Pointer msg) {};

private:

  OIGTLSocketObject           *m_Socket;
  ClientDescriptorXMLBuilder  *m_ClientDescriptor;

}; // end class

#endif

