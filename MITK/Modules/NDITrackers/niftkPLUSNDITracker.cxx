/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPLUSNDITracker.h"
#include <mitkExceptionMacro.h>
#include <qextserialenumerator.h>
#include "ndicapi_serial.h"

namespace niftk
{

//-----------------------------------------------------------------------------
PLUSNDITracker::PLUSNDITracker(mitk::DataStorage::Pointer dataStorage,
                               std::string portName,
                               mitk::TrackingDeviceData deviceData,
                               std::string toolConfigFileName,
                               int preferredFramesPerSecond,
                               int baudRate,
                               int measurementVolumeNumber
                               )
: NDITracker(dataStorage, portName, deviceData, toolConfigFileName, preferredFramesPerSecond)
{
  m_SuppressUpateErrorsAfterNRepeats = 5;
  m_UpdateErrorRepeatCounter = 0;

  // Baseclass will unpack the config file.
  // We must connect to tracker and start tracking.
  for (int i = 0; i < m_NavigationToolStorage->GetToolCount(); i++)
  {
    niftk::NDICAPITracker::NdiToolDescriptor descriptor;
    descriptor.PortEnabled = false;
    descriptor.PortHandle = 0;
    descriptor.VirtualSROM = nullptr;
    descriptor.WiredPortNumber = -1;

    mitk::NavigationTool::Pointer tool = m_NavigationToolStorage->GetTool(i);
    if (tool.IsNull())
    {
      mitkThrow() << "Tool " << i << ", is NULL, suggesting a corrupt tool storage file.";
    }

    std::string sromFileName = tool->GetCalibrationFile();
    if (sromFileName.empty() || sromFileName == "none") // its an aurora tool
    {
      int wpn = -1;
      try
      {
        wpn = std::stoi(tool->GetIdentifier()); // Should throw on failure.
      }
      catch (std::invalid_argument& e)
      {
        mitkThrow() << "Caught '" << e.what()
          << "', which probably means the Identifier field is not an integer"
          << " in the MITK IGTToolStorage file.";
      }

      descriptor.WiredPortNumber = wpn;
    }
    else // its a spectra wireless tool. Wired not supported.
    {
      if(m_Tracker.ReadSromFromFile(descriptor, sromFileName.c_str()) != niftk::NDICAPITracker::PLUS_SUCCESS)
      {
        mitkThrow() << "Failed to read SROM from " << sromFileName;
      }
    }
    m_Tracker.NdiToolDescriptors.insert(std::pair<std::string,
      niftk::NDICAPITracker::NdiToolDescriptor>(tool->GetToolName(), descriptor));
  }

  std::string pName = ConvertPortNameToPortIndexPlusOne(m_PortName);
  m_Tracker.SetSerialPort(std::stoi(pName));
  m_Tracker.SetBaudRate(baudRate);
  m_Tracker.SetMeasurementVolumeNumber(measurementVolumeNumber);

  if (m_Tracker.InternalConnect() != niftk::NDICAPITracker::PLUS_SUCCESS)
  {
    MITK_WARN << "Caught error when connecting to tracker, but carrying on regardless. Check log file." << std::endl;
    return;
  }

  if (m_Tracker.InternalStartRecording() != niftk::NDICAPITracker::PLUS_SUCCESS)
  {
    MITK_WARN << "Caught error when starting tracking, but carrying on regardless. Check log file." << std::endl;
    return;
  }
}


//-----------------------------------------------------------------------------
PLUSNDITracker::~PLUSNDITracker()
{
  // Don't throw exceptions from destructor. Just log stuff.
  if (m_Tracker.InternalStopRecording() != niftk::NDICAPITracker::PLUS_SUCCESS)
  {
    MITK_ERROR << "PLUSNDITracker::An error occured while stopping recording, check log file.";
    return;
  }

  if (m_Tracker.InternalDisconnect() != niftk::NDICAPITracker::PLUS_SUCCESS)
  {
    MITK_ERROR << "PLUSNDITracker::An error occured while disconnecting, check log file.";
    return;
  }
}


//-----------------------------------------------------------------------------
std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > PLUSNDITracker::GetTrackingData()
{
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > result;

  if (m_Tracker.InternalUpdate() != niftk::NDICAPITracker::PLUS_SUCCESS)
  {
    if ( m_UpdateErrorRepeatCounter == m_SuppressUpateErrorsAfterNRepeats )
    {
      MITK_ERROR << "PLUSNDITracker::An error occured while retrieving matrices, " <<  m_UpdateErrorRepeatCounter << " times, suppressing further messages";
    }
    if ( m_UpdateErrorRepeatCounter < m_SuppressUpateErrorsAfterNRepeats )
    {
      MITK_ERROR << "PLUSNDITracker::An error occured while retrieving matrices, check log file.";
    }
    m_UpdateErrorRepeatCounter++;
    return result;
  }
  m_UpdateErrorRepeatCounter = 0;
  mitk::Point4D rotationQuaternion;
  mitk::Vector3D translation;
  std::map<std::string, std::vector<double> > tmpQuaternions = m_Tracker.GetTrackerQuaternions();
  std::map<std::string, std::vector<double> >::const_iterator iter;
  for (iter = tmpQuaternions.begin(); iter != tmpQuaternions.end(); ++iter)
  {
    rotationQuaternion[0] = (*iter).second[0];
    rotationQuaternion[1] = (*iter).second[1];
    rotationQuaternion[2] = (*iter).second[2];
    rotationQuaternion[3] = (*iter).second[3];
    translation[0] = (*iter).second[4];
    translation[1] = (*iter).second[5];
    translation[2] = (*iter).second[6];
    std::pair<mitk::Point4D, mitk::Vector3D> transformAsQuaternion(rotationQuaternion, translation);
    result.insert(std::pair<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >((*iter).first, transformAsQuaternion));
  }

  return result;
}


//-----------------------------------------------------------------------------
std::string PLUSNDITracker::ConvertPortNameToPortIndexPlusOne(const std::string& name) const
{

  std::string result = "";

#ifdef _WIN32 // The name argument contains just the COM port number (without the 'COM') which is what NDICAPI wants.
  result = name;

#elif __APPLE__  // convert the /dev/cu.... to an index in the list, which NDICAPI converts back.
  QStringList ports = getAvailableSerialPorts();
  int indexOfPort = ports.indexOf(QString::fromStdString(name));
  if (indexOfPort != -1)
  {
    result = QString::number(indexOfPort + 1).toStdString();
  }
#else
  // See NifTK/MITK/Modules/NDICAPI/ndicapi/ndicapi_serial.h
  if (name == NDI_DEVICE0)
  {
    result = "1";
  }
  else if (name == NDI_DEVICE1)
  {
    result = "2";
  }
  else if (name == NDI_DEVICE2)
  {
    result = "3";
  }
  else if (name == NDI_DEVICE3)
  {
    result = "4";
  }
  else if (name == NDI_DEVICE4)
  {
    result = "5";
  }
  else if (name == NDI_DEVICE5)
  {
    result = "6";
  }
  else if (name == NDI_DEVICE6)
  {
    result = "7";
  }
  else if (name == NDI_DEVICE7)
  {
    result = "8";
  }
  else
  {
    mitkThrow() << "Invalid port name:" << name;
  }
#endif

  return result;
}

} // end namespace
