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

namespace niftk
{

//-----------------------------------------------------------------------------
PLUSNDITracker::PLUSNDITracker(mitk::DataStorage::Pointer dataStorage,
                               std::string portName,
                               mitk::TrackingDeviceData deviceData,
                               std::string toolConfigFileName,
                               int preferredFramesPerSecond
                               )
: NDITracker(dataStorage, portName, deviceData, toolConfigFileName, preferredFramesPerSecond)
{
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
    std::string sromFileName = tool->GetCalibrationFile();
    if (sromFileName.empty()) // its an aurora tool
    {
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

  m_Tracker.SetBaudRate(115200);
#ifdef _WIN32
  m_Tracker.SetSerialPort(std::stoi(portName));
#else
  mitkThrow() << "Not implemented yet";
#endif

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
std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > PLUSNDITracker::GetTrackingData()
{
  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > result;

  if (m_Tracker.InternalUpdate() != niftk::NDICAPITracker::PLUS_SUCCESS)
  {
    MITK_ERROR << "PLUSNDITracker::An error occured while retrieving matrices, check log file."; 
    return result;
  }
  std::map<std::string, std::vector<double> > tmpMatrices = m_Tracker.GetTrackerMatrices(); // an empty result is not an error. 

  // Convert the standard STL vector to a VTK matrix.
  std::map<std::string, std::vector<double> >::const_iterator iter;
  for (iter = tmpMatrices.begin(); iter != tmpMatrices.end(); ++iter)
  {
    vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    for (int r = 0; r < 4; r++)
    {
      for (int c = 0; c < 4; c++)
      {
        matrix->SetElement(r, c, (*iter).second[r*4 + c]);
      }
    }
    matrix->Transpose(); // Matrix comes out with bottom row containing the translation!

    result.insert(std::pair<std::string, vtkSmartPointer<vtkMatrix4x4> >((*iter).first, matrix));
  }

  return result;
}

} // end namespace
