/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBaseApplication.h>

#include <cctype>
#include <iostream>
#include <sstream>

#include <Poco/Util/HelpFormatter.h>

#include <QString>
#include <QStringList>
#include <QVariant>

namespace niftk
{

const QString BaseApplication::PROP_OPEN = "applicationArgs.open";
const QString BaseApplication::PROP_DERIVES_FROM = "applicationArgs.derives-from";
const QString BaseApplication::PROP_PERSPECTIVE = "applicationArgs.perspective";


BaseApplication::BaseApplication(int argc, char **argv)
  : mitk::BaseApplication(argc, argv)
{
  this->setOrganizationName("CMIC");

  /// We disable processing command line arguments by MITK so that we can introduce
  /// new options. See the uk.ac.ucl.cmic.commonapps plugin activator for details.
  this->setProperty("applicationArgs.processByMITK", false);
}

void BaseApplication::defineOptions(Poco::Util::OptionSet& options)
{
  mitk::BaseApplication::defineOptions(options);

  this->ReformatOptionDescriptions(options);

  Poco::Util::Option openOption("open", "o",
      "\n"
      "Opens a file with the given name.\n");
  openOption.argument("<name>:<file>").repeatable(true);
  openOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(openOption);

  Poco::Util::Option derivesFromOption("derives-from", "d",
      "\n"
      "Makes the data nodes derive from the given source data.\n"
      "The data nodes will appear as children of the source data in the Data Manager.\n");
  derivesFromOption.argument("<source data name>:<data name>[,<data name>]...").repeatable(true);
  derivesFromOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(derivesFromOption);

  Poco::Util::Option perspectiveOption("perspective", "",
      "\n"
      "The initial window perspective.\n");
  perspectiveOption.argument("<perspective>").binding(PROP_PERSPECTIVE.toStdString());
  options.addOption(perspectiveOption);
}

void BaseApplication::HandleRepeatableOption(const std::string& name, const std::string& value)
{
  QString propertyName = "applicationArgs.";
  propertyName.append(QString::fromStdString(name));

  QStringList valueList = this->getProperty(propertyName).toStringList();
  valueList.append(QString::fromStdString(value));
  this->setProperty(propertyName, QVariant::fromValue(valueList));
}

void BaseApplication::ReformatOptionDescriptions(Poco::Util::OptionSet& options)
{
  /// We have to redefine the help option so that we can apply different formatting.
  Poco::Util::Option& helpOption = const_cast<Poco::Util::Option&>(options.getOption("help"));
  helpOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::PrintHelp));

  /// We apply some style tweaks on all existing options (those defined by mitk::BaseApplication):
  ///   - we add an extra new line before and after the description,
  ///   - capitalise the first letter of the description
  ///   - add a period after the description if there was not any.
  for (const Poco::Util::Option& constOption: options)
  {
    Poco::Util::Option& mutableOption = const_cast<Poco::Util::Option&>(constOption);
    std::string description = mutableOption.description();
    if (!description.empty())
    {
      description[0] = std::toupper(description[0]);
      description.insert(description.begin(), '\n');
      if (description[description.length() - 1] == '.')
      {
        description.append("\n");
      }
      else
      {
        description.append(".\n");
      }
    }
    mutableOption.description(description);
  }
}


void BaseApplication::PrintHelp(const std::string& /*name*/, const std::string& /*value*/)
{
  Poco::Util::HelpFormatter help(this->options());
  help.setCommand(this->commandName());
  help.setUsage("[<option>]... [<image file>]... [<project file>]...");
  help.setHeader(
      "\n"
      "The descriptions of the available options are given below. Any other argument\n"
      "is regarded as a data file that is to be opened. Images files will appear with\n"
      "their original name in the Data Manager. If you want to see them with a different\n"
      "name, use the '--open' option.\n"
      "\n"
      "By default, image files are opened in the current application, and each project\n"
      "file is opened in a new application instance. This behaviour can be configured\n"
      "in the application preferences.\n"
      "\n"
      "You can also use space instead of '=' to specify the value for any options.\n"
      "\n"
      "\n"
      "Options:");

  std::string commandName = this->commandName();

  std::stringstream footerStream;
  footerStream
      << "Examples:\n"
         "\n"
         "    " << commandName << " --open T1:/path/to/image.nii.gz --open mask:/path/to/hippo.nii.gz --derives-from T1:mask\n"
         "\n"
         "This command will open 'image.nii.gz' as 'T1' and 'hippo.nii.gz' as 'mask',\n"
         "and it makes 'mask' a derived image ('child') of T1.\n"
         "The next command is equivalent, with shorter notations:"
         "    " << commandName << " -o T1:/path/to/image.nii.gz -o mask:/path/to/mask.nii.gz -d T1:mask\n";

  help.setFooter(footerStream.str());
  help.setUnixStyle(true);
  help.setIndent(4);
  help.setWidth(160);
  help.format(std::cout);

  std::exit(EXIT_OK);
}

}
