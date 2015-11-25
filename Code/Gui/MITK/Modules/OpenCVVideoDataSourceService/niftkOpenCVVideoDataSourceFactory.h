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

#include "niftkOpenCVVideoDataSourceServiceExports.h"
#include <niftkIGIDataSourceFactoryServiceI.h>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceFactory
* \brief Factory class to create OpenCVVideoDataSources.
 */
class NIFTKOPENCVVIDEODATASOURCESERVICE_EXPORT OpenCVVideoDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  OpenCVVideoDataSourceFactory();
  virtual ~OpenCVVideoDataSourceFactory();

  virtual IGIDataSourceServiceI* Create(mitk::DataStorage::Pointer dataStorage) override;

};

} // end namespace

#endif
