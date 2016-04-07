/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKAuroraCubeDataSourceFactory_h
#define niftkMITKAuroraCubeDataSourceFactory_h

#include "niftkMITKTrackerDataSourceFactory.h"

namespace niftk
{

/**
* \class MITKAuroraCubeDataSourceFactory
* \brief Class to create Aurora Cube trackers.
 */
class MITKAuroraCubeDataSourceFactory : public MITKTrackerDataSourceFactory
{

public:

  MITKAuroraCubeDataSourceFactory();
  virtual ~MITKAuroraCubeDataSourceFactory();

  /**
  * \brief Actually creates the tracker.
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const override;
};

} // end namespace

#endif
