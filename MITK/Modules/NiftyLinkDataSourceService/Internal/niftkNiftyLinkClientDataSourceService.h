/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkNiftyLinkClientDataSourceService_h
#define niftkNiftyLinkClientDataSourceService_h

#include "niftkNiftyLinkDataSourceService.h"
#include <niftkIGIDataSourceLocker.h>
#include <NiftyLinkTcpClient.h>

namespace niftk
{

class NiftyLinkClientDataSourceService : public NiftyLinkDataSourceService {

  Q_OBJECT

public:

  mitkClassMacroItkParent(NiftyLinkClientDataSourceService, NiftyLinkDataSourceService);
  mitkNewMacro3Param(NiftyLinkClientDataSourceService, QString,
                     const IGIDataSourceProperties&, mitk::DataStorage::Pointer);

protected:

  NiftyLinkClientDataSourceService(QString factoryName,
                                   const IGIDataSourceProperties& properties,
                                   mitk::DataStorage::Pointer dataStorage
                                  );
  virtual ~NiftyLinkClientDataSourceService();

private slots:

  void OnConnected(QString hostName, int portNumber);
  void OnDisconnected(QString hostName, int portNumber);
  void OnSocketError(QString hostName, int portNumber, QAbstractSocket::SocketError errorCode, QString errorString);
  void OnClientError(QString hostName, int portNumber, QString errorString);
  void OnMessageReceived(NiftyLinkMessageContainer::Pointer message);

private:

  NiftyLinkClientDataSourceService(const NiftyLinkClientDataSourceService&); // deliberately not implemented
  NiftyLinkClientDataSourceService& operator=(const NiftyLinkClientDataSourceService&); // deliberately not implemented

  static niftk::IGIDataSourceLocker               s_Lock;
  int                                             m_ClientNumber;
  niftk::NiftyLinkTcpClient                      *m_Client;
};

} // end namespace


#endif
