/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIUtils.h"
#include <QFile>
#include <mitkSTLFileReader.h>
#include <igtlStringMessage.h>
#include <NiftyLinkSocketObject.h>
#include <Common/NiftyLinkXMLBuilder.h>


//-----------------------------------------------------------------------------
mitk::Surface::Pointer LoadSurfaceFromSTLFile(const QString& surfaceFilename)
{
  mitk::Surface::Pointer toolSurface;

  QFile surfaceFile(surfaceFilename);

  if(surfaceFile.exists())
  {
    mitk::STLFileReader::Pointer stlReader = mitk::STLFileReader::New();

    try
    {
      stlReader->SetFileName(surfaceFilename.toStdString().c_str());
      stlReader->Update();//load surface
      toolSurface = stlReader->GetOutput();
    }
    catch (std::exception& e)
    {
      MBI_ERROR<<"Could not load surface for tool!";
      MBI_ERROR<< e.what();
      throw e;
    }
  }

  return toolSurface;
}


//-----------------------------------------------------------------------------
QString CreateTestDeviceDescriptor()
{
  TrackerClientDescriptor tcld;
  tcld.SetDeviceName("NDI Polaris Vicra");
  tcld.SetDeviceType("Tracker");
  tcld.SetCommunicationType("Serial");
  tcld.SetPortName("Tracker not connected");
  tcld.SetClientIP(GetLocalHostAddress());
  tcld.SetClientPort(QString::number(3200));
  //tcld.AddTrackerTool("8700302.rom");
  tcld.AddTrackerTool("8700338.rom");
  //tcld.AddTrackerTool("8700339.rom");
  tcld.AddTrackerTool("8700340.rom");

  return tcld.GetXMLAsString();
}
