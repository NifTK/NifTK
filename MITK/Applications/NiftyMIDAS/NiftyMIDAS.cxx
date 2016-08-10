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


namespace niftk
{

class BaseApplication : public mitk::BaseApplication
{
public:

  static QString PROP_PERSPECTIVE;

  BaseApplication(int argc, char **argv)
    : mitk::BaseApplication(argc, argv)
  {
    this->setOrganizationName("CMIC");

    /// We disable processing command line arguments by MITK so that we can introduce
    /// new options. See the uk.ac.ucl.cmic.commonapps plugin activator for details.
    this->setProperty("applicationArgs.processByMITK", false);
  }

  /// Define command line arguments
  /// @param options
  void defineOptions(Poco::Util::OptionSet& options) override
  {
    mitk::BaseApplication::defineOptions(options);

    Poco::Util::Option perspectiveOption("perspective", "", "the initial window perspective");
    perspectiveOption.argument("<perspective>").binding(PROP_PERSPECTIVE.toStdString());
    options.addOption(perspectiveOption);

  }

};

QString BaseApplication::PROP_PERSPECTIVE = "applicationArgs.perspective";

class NiftyMIDAS : public BaseApplication
{
public:
  NiftyMIDAS(int argc, char **argv)
    : BaseApplication(argc, argv)
  {
    this->setApplicationName("NiftyMIDAS");
    this->setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftymidas");
  }
};

}

/// \file NiftyMIDAS.cxx
/// \brief Main entry point for NiftyMIDAS application.
int main(int argc, char** argv)
{
  niftk::NiftyMIDAS app(argc, argv);

  return app.run();
}
