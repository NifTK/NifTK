/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysTracker.h"
#include <mitkLogMacros.h>
#include <mitkExceptionMacro.h>
#include <ftkInterface.h>
#include <helpers.hpp>

namespace niftk
{

//-----------------------------------------------------------------------------
class AtracsysTrackerPrivate
{

public:

  AtracsysTrackerPrivate(AtracsysTracker* q,
                         std::string toolConfigFileName);
  ~AtracsysTrackerPrivate();

  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > GetTrackingData();

private:

  void CheckError(ftkLibrary lib);

  ftkLibrary       m_Lib;
  AtracsysTracker *m_Container;
};


//-----------------------------------------------------------------------------
AtracsysTrackerPrivate::AtracsysTrackerPrivate(AtracsysTracker* t,
                                               std::string toolConfigFileName
                                               )
: m_Container(t)
, m_Lib(nullptr)
{

  m_Lib = ftkInit();
  if ( m_Lib == nullptr )
  {
    mitkThrow() << "Cannot initialize Atracsys driver.";
  }

  DeviceData device;
  device.SerialNumber = 0uLL;

  ftkError err = ftkEnumerateDevices( m_Lib, fusionTrackEnumerator, &device );
  if ( err != FTK_OK )
  {
    this->CheckError(m_Lib);
  }

  if ( device.SerialNumber == 0uLL )
  {
    mitkThrow() << "No Atracsys device connected.";
  }

  uint64 sn( device.SerialNumber );
  MITK_INFO << "Connected to Atracsys SN:" << sn;
}


//-----------------------------------------------------------------------------
AtracsysTrackerPrivate::~AtracsysTrackerPrivate()
{
  if (m_Lib != nullptr)
  {
    ftkClose( &m_Lib );
  }
}


//-----------------------------------------------------------------------------
void AtracsysTrackerPrivate::CheckError(ftkLibrary lib)
{
  ftkErrorExt ext;
  ftkError err = ftkGetLastError( lib, &ext );
  if ( err == FTK_OK ) // means we successfully retrieved the error message.
  {
    std::string message;
    if ( ext.isError() )
    {
      ext.errorString( message );
      MITK_ERROR << "AtracsysTrackerPrivate:" << message;
      mitkThrow() << message;
    }
    if ( ext.isWarning() )
    {
      ext.warningString( message );
      MITK_WARN << "AtracsysTrackerPrivate:" << message;
    }
    ext.messageStack( message );
    if ( message.size() > 0u )
    {
      MITK_INFO << "AtracsysTrackerPrivate:Stack:\n" << message;
    }
  }
}


//-----------------------------------------------------------------------------
std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > AtracsysTrackerPrivate::GetTrackingData()
{
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > results;
  return results;
}


//-----------------------------------------------------------------------------
AtracsysTracker::AtracsysTracker(mitk::DataStorage::Pointer dataStorage,
                                 std::string toolConfigFileName)
: niftk::IGITracker(dataStorage, toolConfigFileName, 330)
, m_Tracker(new AtracsysTrackerPrivate(this))
{
  MITK_INFO << "Creating AtracsysTracker";
}


//-----------------------------------------------------------------------------
AtracsysTracker::~AtracsysTracker()
{
  MITK_INFO << "Destroying AtracsysTracker";
}


//-----------------------------------------------------------------------------
std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > AtracsysTracker::GetTrackingData()
{
  return m_Tracker->GetTrackingData();
}

} // end namespace
