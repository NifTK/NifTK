/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKPolarisVicraDataSourceFactory_h
#define niftkMITKPolarisVicraDataSourceFactory_h

#include "niftkMITKTrackerDataSourceFactory.h"

namespace niftk
{

/**
* \class MITKPolarisVicraDataSourceFactory
* \brief Class to create Polaris Vicra trackers.
 */
class MITKPolarisVicraDataSourceFactory : public MITKTrackerDataSourceFactory
{

public:

  MITKPolarisVicraDataSourceFactory();
  virtual ~MITKPolarisVicraDataSourceFactory();

  /**
  * \brief Actually creates the tracker.
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const override;
};

} // end namespace

#endif
