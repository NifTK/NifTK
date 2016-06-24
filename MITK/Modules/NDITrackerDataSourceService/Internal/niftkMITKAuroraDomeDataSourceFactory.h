/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKAuroraDomeDataSourceFactory_h
#define niftkMITKAuroraDomeDataSourceFactory_h

#include "niftkMITKTrackerDataSourceFactory.h"

namespace niftk
{

/**
* \class MITKAuroraDomeDataSourceFactory
* \brief Class to create Aurora Dome trackers.
 */
class MITKAuroraDomeDataSourceFactory : public MITKTrackerDataSourceFactory
{

public:

  MITKAuroraDomeDataSourceFactory();
  virtual ~MITKAuroraDomeDataSourceFactory();

  /**
  * \brief Actually creates the tracker.
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const override;
};

} // end namespace

#endif
