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
const QString BaseApplication::PROP_PROPERTY = "applicationArgs.property";
const QString BaseApplication::PROP_PERSPECTIVE = "applicationArgs.perspective";


BaseApplication::BaseApplication(int argc, char **argv)
  : mitk::BaseApplication(argc, argv)
{
  this->setOrganizationName("CMIC");
}

void BaseApplication::defineOptions(Poco::Util::OptionSet& options)
{
  mitk::BaseApplication::defineOptions(options);

  this->ReformatOptionDescriptions(options);

  Poco::Util::Option openOption("open", "o",
      "\n"
      "Opens a file with the given name. Several files can be opened by repeating this\n"
      "option. The data nodes will appear in the Data Manager in the same order as listed\n"
      "on the command line, i.e. the first data will be in the uppermost layer.\n");
  openOption.argument("<name>:<file>").repeatable(true);
  openOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(openOption);

  Poco::Util::Option derivesFromOption("derives-from", "d",
      "\n"
      "Makes the data nodes derive from the given source data. The data nodes will appear\n"
      "as children of the source data in the Data Manager, in the same order as listed\n"
      "on the command line, i.e. first data will be in the uppermost layer.\n");
  derivesFromOption.argument("<source data name>:<data name>[,<data name>]...").repeatable(true);
  derivesFromOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(derivesFromOption);

  Poco::Util::Option propertyOption("property", "p",
      "\n"
      "Sets properties of a data node to the given values.\n"
      "The type of the property is determined as follows:\n"
      "    - 'true', 'on' and 'yes' will become a bool property with 'true' value.\n"
      "    - 'false', 'off' and 'no' will become a bool property with 'false' value.\n"
      "    - Decimal numbers without fractional and exponential part will become an\n"
      "      int property.\n"
      "    - Other decimal numbers (with fractional and/or exponential part) will\n"
      "      become a float property. If you want an integer number to be interpreted\n"
      "      as a float property, specify it with a '.0' fractional part.\n"
      "    - Number ranges in the form '<number>-<number>' will become a level-window\n"
      "      property with the given lower and upper bound.\n"
      "    - Anything else will become a string property. String values can be enclosed\n"
      "      within single or double quotes. This is necessary if the value contains\n"
      "      white space or when it is in any of the previous form (boolean, decimal\n"
      "      number or number range) but you want it to be interpreted as a string.\n"
      "      The leading and trailing quote will be removed from the final string value.\n"
      "      You will likely need to protect the quotes and spaces by backslash because\n"
      "      of your shell.\n"
      "");
  propertyOption.argument("<data name>:<property>=<value>[,<property>=<value>]...").repeatable(true);
  propertyOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(propertyOption);

  Poco::Util::Option perspectiveOption("perspective", "P",
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


Poco::Util::HelpFormatter* BaseApplication::CreateHelpFormatter()
{
  Poco::Util::HelpFormatter* helpFormatter = new Poco::Util::HelpFormatter(this->options());
  helpFormatter->setCommand(this->commandName());
  helpFormatter->setUsage("[<option>]... [<image file>]... [<project file>]...");
  helpFormatter->setHeader(
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


  QString examples =
      "Examples:\n"
      "\n"
      "The following command will open 'image.nii.gz' as 'T1', 'left-hippocampus.nii.gz' as\n"
      "'l-hippo' and 'right-hippocampus.nii.gz' as 'r-hippo', and it makes the segmentations\n"
      "derived images ('children') of T1:\n"
      "\n"
      "    ${commandName} --open T1:/path/to/image.nii.gz --open l-hippo:/path/to/left-hippocampus.nii.gz \\\n"
      "        --open r-hippo:/path/to/right-hippocampus.nii.gz \\\n"
      "        --derives-from T1:l-hippo,r-hippo\n"
      "\n"
      "The next command is equivalent, with shorter notations:\n"
      "\n"
      "    ${commandName} -o T1:/path/to/image.nii.gz -o l-hippo:/path/to/left-hippocampus.nii.gz\\\n"
      "         -o r-hippo:/path/to/right-hippocampus.nii.gz -d T1:l-hippo,r-hippo\n"
      "\n"
      "The following command opens a reference image and a mask, sets the intensity range of\n"
      "the reference image to 100-3500, disables the feature of outlining the binary images\n"
      "so that the mask is rendered as a solid layer rather than a contour, and sets the opacity\n"
      "to make it transparent.\n"
      "\n"
      "    ${commandName} -o T1:image.nii.gz -o mask:segmentation.nii.gz -d T1:mask \\\n"
      "        -p T1:levelwindow=100-3500 -p mask:outline\\ binary=false,opacity=0.3\n"
      "\n"
      "The following command opens ${commandName} in 'Minimal' perspective, in which you have "
      "only the viewer and the Data Manager open.\n"
      "\n"
      "    ${commandName} --perspective Minimal\n"
      "";
  examples.replace("${commandName}", QString::fromStdString(this->commandName()));

  helpFormatter->setFooter(examples.toStdString());
  helpFormatter->setUnixStyle(true);
  helpFormatter->setIndent(4);
  helpFormatter->setWidth(160);

  return helpFormatter;
}


void BaseApplication::PrintHelp(const std::string& /*name*/, const std::string& /*value*/)
{
  Poco::Util::HelpFormatter* helpFormatter = this->CreateHelpFormatter();

  helpFormatter->format(std::cout);

  delete helpFormatter;

  std::exit(EXIT_OK);
}

}
