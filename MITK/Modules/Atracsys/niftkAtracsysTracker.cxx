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

  AtracsysTrackerPrivate(const AtracsysTracker* q,
                         const std::vector<std::string>& toolGeometryFileNames);
  ~AtracsysTrackerPrivate();

  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > GetTrackingData();

private:

  void CheckError(ftkLibrary lib);

  const AtracsysTracker          *m_Container;
  const std::vector<std::string>  m_GeometryFiles;
  ftkLibrary                      m_Lib;

};


//-----------------------------------------------------------------------------
AtracsysTrackerPrivate::AtracsysTrackerPrivate(const AtracsysTracker* t,
                                               const std::vector<std::string>& toolGeometryFileNames
                                              )
: m_Container(t)
, m_GeometryFiles(toolGeometryFileNames)
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
, m_Tracker(nullptr)
{
  MITK_INFO << "Creating AtracsysTracker";

  std::vector<std::string> fileNames;
  for (int i = 0; i < m_NavigationToolStorage->GetToolCount(); i++)
  {
    mitk::NavigationTool::Pointer tool = m_NavigationToolStorage->GetTool(i);
    fileNames.push_back(tool->GetCalibrationFile());
  }

  m_Tracker.reset(new AtracsysTrackerPrivate(this, fileNames));
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
