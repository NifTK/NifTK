/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

//#include "../NifTKApplication.h"
#include <mitkBaseApplication.h>


#include <QVariant>

/**
 * \file NiftyView.cxx
 * \brief Main entry point for NiftyView application.
 */

int main(int argc, char** argv)
{
  // Create a QApplication instance first
  mitk::BaseApplication myApp(argc, argv);
  myApp.setApplicationName("NiftyView");
  myApp.setOrganizationName("CMIC");

  myApp.setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.gui.qt.niftyview");

  return myApp.run();
}
