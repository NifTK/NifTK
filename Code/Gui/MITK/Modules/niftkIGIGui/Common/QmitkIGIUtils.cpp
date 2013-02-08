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
#include <OIGTLSocketObject.h>
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
  tcld.setDeviceName("NDI Polaris Vicra");
  tcld.setDeviceType("Tracker");
  tcld.setCommunicationType("Serial");
  tcld.setPortName("Tracker not connected");
  tcld.setClientIP(getLocalHostAddress());
  tcld.setClientPort(QString::number(3200));
  //tcld.addTrackerTool("8700302.rom");
  tcld.addTrackerTool("8700338.rom");
  //tcld.addTrackerTool("8700339.rom");
  tcld.addTrackerTool("8700340.rom");

  return tcld.getXMLAsString();
}
