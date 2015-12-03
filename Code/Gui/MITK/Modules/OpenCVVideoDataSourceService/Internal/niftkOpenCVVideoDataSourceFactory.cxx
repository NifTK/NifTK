/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVVideoDataSourceFactory.h"
#include "niftkOpenCVVideoDataSourceService.h"
#include <niftkIPPortDialog.h>
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
OpenCVVideoDataSourceFactory::OpenCVVideoDataSourceFactory()
: IGIDataSourceFactoryServiceI("OpenCV Frame Grabber",
                               true, // don't need to configure at startup
                               false // don't need to configure when running
                               )
{
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceFactory::~OpenCVVideoDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer OpenCVVideoDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const QMap<QString, QVariant>& properties) const
{
  niftk::OpenCVVideoDataSourceService::Pointer serviceInstance
      = OpenCVVideoDataSourceService::New(this->GetName(), dataStorage);

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* OpenCVVideoDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::IPPortDialog(parent);
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* OpenCVVideoDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
std::vector<std::string> OpenCVVideoDataSourceFactory::GetLegacyClassNames() const
{
  std::vector<std::string> names;
  names.push_back("QmitkIGIOpenCVDataSource");
  return names;
}

} // end namespace
