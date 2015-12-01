/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkOpenCVVideoDataSourceFactory_h
#define niftkOpenCVVideoDataSourceFactory_h

#include <niftkIGIDataSourceFactoryServiceI.h>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceFactory
* \brief Factory class to create OpenCVVideoDataSources.
 */
class OpenCVVideoDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  OpenCVVideoDataSourceFactory();
  virtual ~OpenCVVideoDataSourceFactory();

  /**
  * \see IGIDataSourceFactoryServiceI::Create()
  */
  virtual IGIDataSourceI::Pointer Create(mitk::DataStorage::Pointer dataStorage) override;

  /**
  * \see IGIDataSourceFactoryServiceI::Create()
  */
  virtual IGIDataSourceI::Pointer Create(const std::string& name,
                                         mitk::DataStorage::Pointer dataStorage) override;

  /**
  * \brief Returns "QmitkIGIOpenCVDataSource"
  */
  virtual std::vector<std::string> GetLegacyClassNames() const override;
};

} // end namespace

#endif
