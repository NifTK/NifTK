/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkNiftyLinkServerDataSourceService_h
#define niftkNiftyLinkServerDataSourceService_h

#include "niftkNiftyLinkDataSourceService.h"
#include <niftkIGIDataSourceLocker.h>
#include <NiftyLinkTcpServer.h>

namespace niftk
{

class NiftyLinkServerDataSourceService : public NiftyLinkDataSourceService {

  Q_OBJECT

public:

  mitkClassMacroItkParent(NiftyLinkServerDataSourceService, NiftyLinkDataSourceService);
  mitkNewMacro3Param(NiftyLinkServerDataSourceService, QString, const IGIDataSourceProperties&, mitk::DataStorage::Pointer);

protected:

  NiftyLinkServerDataSourceService(QString factoryName,
                                   const IGIDataSourceProperties& properties,
                                   mitk::DataStorage::Pointer dataStorage
                                  );
  virtual ~NiftyLinkServerDataSourceService();

private slots:

  void OnClientConnected(int portNumber);
  void OnClientDisconnected(int portNumber);
  void OnSocketError(int portNumber, QAbstractSocket::SocketError errorCode, QString errorString);
  void OnMessageReceived(int portNumber, niftk::NiftyLinkMessageContainer::Pointer message);

private:

  NiftyLinkServerDataSourceService(const NiftyLinkServerDataSourceService&); // deliberately not implemented
  NiftyLinkServerDataSourceService& operator=(const NiftyLinkServerDataSourceService&); // deliberately not implemented

  static niftk::IGIDataSourceLocker               s_Lock;
  int                                             m_ServerNumber;
  niftk::NiftyLinkTcpServer                      *m_Server;

};

} // end namespace


#endif
