/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyCalVideoCalibrationManager.h"
#include <mitkExceptionMacro.h>
#include <niftkNiftyCalTypes.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::NiftyCalVideoCalibrationManager()
: m_DataStorage(nullptr)
, m_LeftImageNode(nullptr)
, m_RightImageNode(nullptr)
, m_TrackingTransformNode(nullptr)
, m_MinimumNumberOfSnapshotsForCalibrating(5)
{
}


//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::~NiftyCalVideoCalibrationManager()
{

}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetDataStorage(
    const mitk::DataStorage::Pointer& storage)
{
  if (storage.IsNull())
  {
    mitkThrow() << "Null DataStorage passed";
  }
  m_DataStorage = storage;
}


//-----------------------------------------------------------------------------
unsigned int NiftyCalVideoCalibrationManager::GetNumberOfSnapshots() const
{
  return 1;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Restart()
{
  MITK_INFO << "Restart";
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::Grab()
{
  MITK_INFO << "Grabbing data.";
  return true;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::UnGrab()
{
  MITK_INFO << "Removing last snapshot.";
}


//-----------------------------------------------------------------------------
double NiftyCalVideoCalibrationManager::Calibrate()
{
  MITK_INFO << "Calibrating.";
  int j=0;
  for (int i = 0; i < 1000000000; i++)
  {
    j++;
  }
  MITK_INFO << "Calibrating - DONE";
  return 0;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Save(const std::string dirName)
{
  MITK_INFO << "Saving calibration to " << dirName;
  niftk::NiftyCalTimeType time;
}

} // end namespace
