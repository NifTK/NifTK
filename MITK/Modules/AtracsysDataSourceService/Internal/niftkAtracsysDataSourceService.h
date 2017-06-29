/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkAtracsysDataSourceService_h
#define niftkAtracsysDataSourceService_h

#include <niftkIGITrackerDataSourceService.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include <niftkAtracsysTracker.h>

namespace niftk
{

/**
* \class AtracsysDataSourceService
* \brief Provides an interface to an Atracsys Fusion Track 500,
* as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class AtracsysDataSourceService : public IGITrackerDataSourceService
{

public:

  mitkClassMacroItkParent(AtracsysDataSourceService, IGITrackerDataSourceService)
  mitkNewMacro3Param(AtracsysDataSourceService, QString,
                     const IGIDataSourceProperties&, mitk::DataStorage::Pointer)

protected:

  AtracsysDataSourceService(QString factoryName,
                            const IGIDataSourceProperties& properties,
                            mitk::DataStorage::Pointer dataStorage
                           );

  virtual ~AtracsysDataSourceService();

private:

  AtracsysDataSourceService(const AtracsysDataSourceService&); // deliberately not implemented
  AtracsysDataSourceService& operator=(const AtracsysDataSourceService&); // deliberately not implemented

  niftk::IGIDataSourceGrabbingThread*    m_DataGrabbingThread;

}; // end class

} // end namespace

#endif
