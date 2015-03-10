/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGINiftyLinkDataSource_h
#define QmitkIGINiftyLinkDataSource_h

#include "niftkIGIDataSourcesExports.h"
#include "QmitkIGIDataSource.h"
#include <NiftyLinkTcpServer.h>
#include <NiftyLinkMessageContainer.h>
#include <NiftyLinkXMLBuilder.h>
#include <igtlTimeStamp.h>

/**
 * \class QmitkIGINiftyLinkDataSource
 * \brief Base class for IGI Data Sources that are receiving networked input
 * from NiftyLink. NiftyLink uses Qt, so this class is in the Qt library, and named Qmitk.
 */
class NIFTKIGIDATASOURCES_EXPORT QmitkIGINiftyLinkDataSource : public QmitkIGIDataSource
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGINiftyLinkDataSource, QmitkIGIDataSource);

  /**
   * \brief Sets the server pointer.
   */
  itkSetObjectMacro(Server, niftk::NiftyLinkTcpServer);

  /**
   * \brief Gets the server pointer.
   */
  itkGetConstMacro(Server, niftk::NiftyLinkTcpServer*);

  /**
   * \brief Sets the Client Descriptor XML.
   */
  itkSetObjectMacro(ClientDescriptor, niftk::NiftyLinkClientDescriptor);

  /**
   * \brief Gets the Client Descriptor XML.
   */
  itkGetConstMacro(ClientDescriptor, niftk::NiftyLinkClientDescriptor*);

  /**
   * \brief Returns the port number that this tool is using or -1 if no server is available.
   */
  int GetPort() const;

  /**
   * \brief Tells this object to start listening on a given port number.
   */
  bool ListenOnPort(int portNumber);

protected:

  /**
   * \brief Constructor where socket creation is optional.
   * \param socket if NULL a new socket will be created.
   */
  QmitkIGINiftyLinkDataSource(mitk::DataStorage* storage, niftk::NiftyLinkTcpServer *server); // Purposefully hidden.
  virtual ~QmitkIGINiftyLinkDataSource(); // Purposefully hidden.

  QmitkIGINiftyLinkDataSource(const QmitkIGINiftyLinkDataSource&); // Purposefully not implemented.
  QmitkIGINiftyLinkDataSource& operator=(const QmitkIGINiftyLinkDataSource&); // Purposefully not implemented.

  /**
   * \brief When client information is received we update the DataSource member variables, and dump info to console.
   */
  void ProcessClientInfo(niftk::NiftyLinkClientDescriptor* clientInfo);

protected slots:

  /**
   * \brief Slot called when client connects.
   */
  virtual void ClientConnected();

  /**
   * \brief Slot called when client disconnects.
   */
  virtual void ClientDisconnected();

  /**
   * \brief Main message handler routine for this tool, that subclasses must implement.
   */
  virtual void InterpretMessage(niftk::NiftyLinkMessageContainer::Pointer msg) {}

private:

  niftk::NiftyLinkTcpServer         *m_Server;
  niftk::NiftyLinkClientDescriptor  *m_ClientDescriptor;
  bool                               m_UsingSomeoneElsesServer;

}; // end class

#endif
