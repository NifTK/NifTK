/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8038 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <berryStarter.h>
#include <Poco/Util/MapConfiguration.h>
#include <QApplication>
#include "mitkDRCCoreObjectFactory.h"

int main(int argc, char** argv)
{
  // Create a QApplication instance first
  QApplication myApp(argc, argv);
  myApp.setApplicationName("NiftyView");
  myApp.setOrganizationName("CMIC");

  // This causes a real problem on windows, as it brings up an annoying error window.
  // We get VTK errors from the Thumbnail widget, as it switches orientation (axial, coronal, sagittal).
  // So, for now we block them completely.  This could be command line driven, or just done on Windows.
  vtkObject::GlobalWarningDisplayOff();

  // This is a DRC specific override (could make it controlled by command line params).
  // It takes care of registering the default MITK core object factories, which includes
  // the ITK based file reader. It then hunts down the ITK based file reader, and kills
  // it, and replaces it with a more DRC suitable one.
  RegisterDRCCoreObjectFactory();

  // These paths replace the .ini file and are tailored for installation
  // packages created with CPack. If a .ini file is presented, it will
  // overwrite the settings in MapConfiguration
  Poco::Path basePath(argv[0]);
  basePath.setFileName("");
  
  Poco::Path provFile(basePath);
  provFile.setFileName("NiftyView.provisioning");

  Poco::Util::MapConfiguration* sbConfig(new Poco::Util::MapConfiguration());
  sbConfig->setString(berry::Platform::ARG_PROVISIONING, provFile.toString());
  sbConfig->setString(berry::Platform::ARG_APPLICATION, "uk.ac.ucl.cmic.gui.qt.niftyview");
  return berry::Starter::Run(argc, argv, sbConfig);
}
