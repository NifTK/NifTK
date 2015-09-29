/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkBaseApplication.h>

#include <QVariant>

/**
 * \file NiftyIGI.cxx
 * \brief Main entry point for NiftyIGI application.
 */

int main(int argc, char** argv)
{
  // Create a QApplication instance first
  mitk::BaseApplication myApp(argc, argv);
  myApp.setApplicationName("NiftyIGI");
  myApp.setOrganizationName("CMIC");

  myApp.setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.gui.qt.niftyigi");

  return myApp.run();
}
