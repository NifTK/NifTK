/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkBaseApplication.h>

#include <Poco/Util/OptionException.h>

#include <mitkLogMacros.h>

#include <QVariant>


class NiftyMIDASApplication : public mitk::BaseApplication
{
public:

  NiftyMIDASApplication(int argc, char** argv)
  : mitk::BaseApplication(argc, argv)
  {
    this->setApplicationName("NiftyMIDAS");
    this->setOrganizationName("CMIC");
    this->setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.gui.qt.niftymidas");
  }

  int run() override
  {
    try
    {
      this->init(this->getArgc(), this->getArgv());
    }
    catch (Poco::Util::UnknownOptionException& e)
    {
      MITK_WARN << e.displayText();
    }

    return Poco::Util::Application::run();
  }
};

/**
 * \file NiftyMIDAS.cxx
 * \brief Main entry point for NiftyMIDAS application.
 */

int main(int argc, char** argv)
{
  NiftyMIDASApplication application(argc, argv);

  return application.run();
}
