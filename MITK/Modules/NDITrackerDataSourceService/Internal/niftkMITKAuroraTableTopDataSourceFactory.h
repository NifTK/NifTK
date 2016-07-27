/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKAuroraTableTopDataSourceFactory_h
#define niftkMITKAuroraTableTopDataSourceFactory_h

#include "niftkMITKTrackerDataSourceFactory.h"

namespace niftk
{

/**
* \class MITKAuroraTableTopDataSourceFactory
* \brief Class to create Aurora Table Top trackers.
 */
class MITKAuroraTableTopDataSourceFactory : public MITKTrackerDataSourceFactory
{

public:

  MITKAuroraTableTopDataSourceFactory();
  virtual ~MITKAuroraTableTopDataSourceFactory();

  /**
  * \brief Actually creates the tracker.
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateInitialisationDialog()
  *
  * All NDI trackers need the port (USB port) number, baud rate and config file settings at startup.
  */
  virtual IGIInitialisationDialog* CreateInitialisationDialog(QWidget *parent) const override;

};

} // end namespace

#endif
