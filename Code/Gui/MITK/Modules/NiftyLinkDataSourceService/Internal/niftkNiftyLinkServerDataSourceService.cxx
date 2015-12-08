/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkServerDataSourceService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker NiftyLinkServerDataSourceService::s_Lock;


//-----------------------------------------------------------------------------
NiftyLinkServerDataSourceService::NiftyLinkServerDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage
    )
: NiftyLinkDataSourceService(QString("NLServer-") + QString::number(s_Lock.GetNextSourceNumber()),
                             factoryName, properties, dataStorage)
{
  QString deviceName = this->GetName();
  m_ServerNumber = (deviceName.remove(0, 9)).toInt(); // Should match string NLServer- above
}


//-----------------------------------------------------------------------------
NiftyLinkServerDataSourceService::~NiftyLinkServerDataSourceService()
{
  this->StopCapturing();

  s_Lock.RemoveSource(m_ServerNumber);
}

} // end namespace
