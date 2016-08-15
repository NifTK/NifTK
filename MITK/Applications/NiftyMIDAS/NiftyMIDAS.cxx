/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBaseApplication.h>

#include <Poco/Util/HelpFormatter.h>
#include <Poco/Util/Option.h>

#include <QVariant>


namespace niftk
{

class NiftyMIDAS : public BaseApplication
{
public:
  static const QString PROP_VIEWER_NUMBER;
  static const QString PROP_DRAG_AND_DROP;

  NiftyMIDAS(int argc, char **argv)
    : BaseApplication(argc, argv)
  {
    this->setApplicationName("NiftyMIDAS");
    this->setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftymidas");
  }

  /// \brief Define command line arguments
  /// \param options
  void defineOptions(Poco::Util::OptionSet& options) override
  {
    BaseApplication::defineOptions(options);

    Poco::Util::Option viewersOption("viewer-number", "v",
        "\n"
        "Sets the number of viewers.\n"
        "Viewers can be arranged in rows and columns. By default (without this option),\n"
        "there is one viewer. You can specify the required rows and numbers in the form\n"
        "<rows>x<columns>. The row number can be omitted. If only one number is given, the\n"
        "viewers will appear side-by-side.\n");
    viewersOption.argument("[<rows>x]<columns>").binding(PROP_VIEWER_NUMBER.toStdString());
    options.addOption(viewersOption);

    Poco::Util::Option dragAndDropOption("drag-and-drop", "D",
        "\n"
        "Opens the data nodes in the given viewers. By default (without this option),\n"
        "the data nodes are added to the Data Manager, but not loaded into any viewer.\n"
        "\n"
        "<data nodes> is a comma separated list of data node names, as given with the\n"
        "'--open' option. At least one node has to be given.\n"
        "\n"
        "<viewers> is a comma separated list of viewer indices. A viewer index is in the\n"
        "form '[<row>x]<column>' where the indexing starts from 1. If the <viewers> part is\n"
        "omitted, the data nodes will be loaded into the first viewer.\n");
    dragAndDropOption.argument("<data nodes>[:<viewers>]").repeatable(true);
    dragAndDropOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &NiftyMIDAS::HandleRepeatableOption));
    options.addOption(dragAndDropOption);
  }

  virtual Poco::Util::HelpFormatter* CreateHelpFormatter() override
  {
    Poco::Util::HelpFormatter* helpFormatter = BaseApplication::CreateHelpFormatter();

    std::string footer = helpFormatter->getFooter();

    QString examples =
        "\n"
        "The following command opens two reference images and one segmentation, and opens\n"
        "them in two viewers, side-by-side:\n"
        "\n"
        "    ${commandName} -o mask:segmentation.nii -o timeline1:image1.nii -o timeline2:image2.nii \\\n"
        "        --viewer-number 2 --drag-and-drop timeline1,mask:1 --drag-and-drop timeline2,mask:2\n"
        "\n"
        "Note that 'mask' has to be specified before the reference images so that it is\n"
        "placed in an upper layer than the reference images.\n";

    examples.replace("${commandName}", QString::fromStdString(this->commandName()));

    footer += examples.toStdString();

    helpFormatter->setFooter(footer);

    return helpFormatter;
  }

};

const QString NiftyMIDAS::PROP_VIEWER_NUMBER = "applicationArgs.viewer-number";
const QString NiftyMIDAS::PROP_DRAG_AND_DROP = "applicationArgs.drag-and-drop";

}

/// \file NiftyMIDAS.cxx
/// \brief Main entry point for NiftyMIDAS application.
int main(int argc, char** argv)
{
  niftk::NiftyMIDAS app(argc, argv);

  return app.run();
}
