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

/// \file NiftyView.cxx
/// \brief Main entry point for NiftyView application.
int main(int argc, char** argv)
{
  mitk::BaseApplication myApp(argc, argv);
  myApp.setApplicationName("NiftyView");
  myApp.setOrganizationName("CMIC");

  myApp.setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftyview");

  /// We disable processing command line arguments by MITK so that we can introduce
  /// new options. See the uk.ac.ucl.cmic.commonapps plugin activator for details.
  myApp.setProperty("applicationArgs.processByMITK", false);

  return myApp.run();
}
