/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDataSourceFactory.h"
#include "niftkMITKTrackerDataSourceService.h"
#include <niftkMITKTrackerDialog.h>
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKTrackerDataSourceFactory::MITKTrackerDataSourceFactory(QString name)
: IGIDataSourceFactoryServiceI(name,
                               true, // can configure port number
                               true  // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
MITKTrackerDataSourceFactory::~MITKTrackerDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* MITKTrackerDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::MITKTrackerDialog(parent, this->GetName());
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* MITKTrackerDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> MITKTrackerDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceFactory::ExtractProperties(const IGIDataSourceProperties& properties,
    mitk::SerialCommunication::PortNumber& outputPortNumber,
    std::string& outputFileName) const
{
  int portNumber = 0;
  if(!properties.contains("port"))
  {
    mitkThrow() << "Port number not specified!";
  }
  portNumber = (properties.value("port")).toInt();
  outputPortNumber = static_cast<mitk::SerialCommunication::PortNumber>(portNumber);

  if(!properties.contains("file"))
  {
    mitkThrow() << "Configuration file not specified!";
  }
  std::string fileName = (properties.value("file")).toString().toStdString();
  if (fileName.size() == 0)
  {
    mitkThrow() << "Empty configuration file name specified!";
  }
  outputFileName = fileName;
}

} // end namespace
