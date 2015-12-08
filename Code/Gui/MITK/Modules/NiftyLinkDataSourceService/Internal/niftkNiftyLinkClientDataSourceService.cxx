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
: NiftyLinkDataSourceService((QString("NLClient-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                             factoryName, properties, dataStorage)
{
  QString deviceName = this->GetName();
  m_ClientNumber = (deviceName.remove(0, 9)).toInt(); // Should match string NLClient- above

}


//-----------------------------------------------------------------------------
NiftyLinkClientDataSourceService::~NiftyLinkClientDataSourceService()
{
  this->StopCapturing();

  s_Lock.RemoveSource(m_ServerNumber);
}

} // end namespace
