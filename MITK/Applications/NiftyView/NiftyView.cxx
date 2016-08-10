/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBaseApplication.h>

#include <QVariant>

/// \file NiftyView.cxx
/// \brief Main entry point for NiftyView application.
int main(int argc, char** argv)
{
  niftk::BaseApplication app(argc, argv);
  app.setApplicationName("NiftyView");
  app.setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftyview");

  return app.run();
}
