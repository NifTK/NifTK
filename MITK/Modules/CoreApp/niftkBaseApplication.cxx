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
#include <Poco/Util/OptionException.h>

#include <QString>
#include <QStringList>
#include <QVariant>

#include <NifTKConfigure.h>

namespace niftk
{

const QString BaseApplication::PROP_OPEN = "applicationArgs.open";
const QString BaseApplication::PROP_DERIVES_FROM = "applicationArgs.derives-from";
const QString BaseApplication::PROP_PROPERTY = "applicationArgs.property";
const QString BaseApplication::PROP_PERSPECTIVE = "applicationArgs.perspective";
const QString BaseApplication::PROP_RESET_PERSPECTIVE = "applicationArgs.reset-perspective";
const QString BaseApplication::PROP_VERSION = "applicationArgs.version";
const QString BaseApplication::PROP_PRODUCT_NAME = "applicationArgs.product-name";


//-----------------------------------------------------------------------------
BaseApplication::BaseApplication(int argc, char **argv)
  : mitk::BaseApplication(argc, argv)
{
  this->setOrganizationName("CMIC");
}


//-----------------------------------------------------------------------------
int BaseApplication::run()
{
  try {
    return mitk::BaseApplication::run();
  }
  catch (const Poco::Util::UnknownOptionException& exc)
  {
    std::cerr << "Unknown command line option: " << exc.message() << std::endl;
    std::cerr << "Exiting." << std::endl;
  }
  catch (const Poco::Exception& exc)
  {
    std::cerr << "Unknown error occurred." << std::endl;
    std::cerr << exc.what() << std::endl;
    std::cerr << "Exiting." << std::endl;
  }
  /// Note:
  /// If std::exception is caught here, the debugger gets stuck somewhere in Poco
  /// when starting the application. This happened on Mac with QtCreator 4.0.1 and
  /// clang of XCode 7.3.1.

  return 1;
}


//-----------------------------------------------------------------------------
void BaseApplication::defineOptions(Poco::Util::OptionSet& options)
{
  mitk::BaseApplication::defineOptions(options);

  this->ReformatOptionDescriptions(options);

  Poco::Util::Option versionOption("version", "V",
      "\n"
      "Prints the version number.\n");
  versionOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::PrintVersion));
  options.addOption(versionOption);

  Poco::Util::Option openOption("open", "o",
      "\n"
      "Opens a file with the given name(s).\n"
      "\n"
      "Several files can be opened by repeating this option. The data nodes will appear "
      "in the Data Manager in the same order as listed on the command line, i.e. the first "
      "data node will be in the uppermost layer.\n"
      "\n"
      "<data node names> is a comma separated list of data node names. It is typically one "
      "name only, but you can specify multiple names to open copies of the same data. "
      "It can be omitted (with the colon), then the name will be taken from the file name, "
      "by removing the extensions.\n");
  openOption.argument("<data node names>:<file>").repeatable(true);
  openOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(openOption);

  Poco::Util::Option derivesFromOption("derives-from", "d",
      "\n"
      "Makes the data nodes derive from the given source data node.\n"
      "\n"
      "The data nodes will appear as children of the source data node in the Data Manager, "
      "in the same order as listed on the command line, i.e. first data node will be in the "
      "uppermost layer.\n"
      "\n"
      "<source data node> is the name of the data node from which the other data nodes "
      "have to derive from. The name must be used with the '--open' option.\n"
      "\n"
      "It is also possible to specify several source data nodes, separated by commas, "
      "but it is better not to do so. "
      "Although the MITK data storage model allows data nodes to have multiple sources, "
      "the Data Manager renders the data nodes into a tree and displays only one of them.\n"
      "\n"
      "<data nodes> is a comma separated list of data node names. The data nodes "
      "will be derived from the source data node. The names must be used with "
      "the '--open' option.\n");
  derivesFromOption.argument("<source data node>:<data nodes>").repeatable(true);
  derivesFromOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(derivesFromOption);

  Poco::Util::Option propertyOption("property", "p",
      "\n"
      "Sets properties of data nodes to the given values.\n"
      "\n"
      "<data nodes> is a comma separated list of data node names. If you specify multiple "
      "names, the property values will be set for each data node. At least one data node "
      "name has to be given. The names must be used with the '--open' option.\n"
      "\n"
      "<property assignments> is a comma separated list of property assignments of "
      "the following form:\n"
      "\n"
      "    <property name>=<value>\n"
      "\n"
      "The type of the property is determined from the given value as follows:\n"
      "\n"
      "    - 'true', 'on' and 'yes' will become a bool property with 'true' value.\n"
      "\n"
      "    - 'false', 'off' and 'no' will become a bool property with 'false' value.\n"
      "\n"
      "    - colours in any of the following forms will become a colour property:\n"
      "\n"
      "      - '#RGB', '#RRGGBB', '#RRRGGGBBB' or '#RRRRGGGGBBBB' where 'R', 'G' and 'B' is a single hex digit.\n"
      "\n"
      "      - a name from the list of SVG color keyword names provided by the World Wide Web Consortium:\n"
      "        http://www.w3.org/TR/SVG/types.html#ColorKeywords\n"
      "\n"
      "    - Decimal numbers without fractional and exponential part will become an int property.\n"
      "\n"
      "    - Other decimal numbers (with fractional and/or exponential part) will become a float property.\n"
      "      If you want an integer number to be interpreted as a float property, specify it with a '.0' fractional part.\n"
      "\n"
      "    - Number ranges in the form '<number>-<number>' will become a level-window property with the given lower and upper bound.\n"
      "\n"
      "    - Anything else will become a string property. String values can be enclosed within single or double quotes.\n"
      "      This is necessary if the value contains white space or when it is in any of the previous form (boolean,\n"
      "      decimal number or number range) but you want it to be interpreted as a string.\n"
      "      The leading and trailing quote will be removed from the final string value.\n"
      "      You will likely need to protect the quotes and spaces by backslash because of your shell.\n"
      "");
  propertyOption.argument("<data nodes>:<property assignments>").repeatable(true);
  propertyOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(propertyOption);

  Poco::Util::Option perspectiveOption("perspective", "P",
      "\n"
      "The initial window perspective.\n");
  perspectiveOption.argument("<perspective>").binding(PROP_PERSPECTIVE.toStdString());
  options.addOption(perspectiveOption);

  Poco::Util::Option resetPerspectiveOption("reset-perspective", "",
      "\n"
      "Reverts the perspective to its original layout.\n");
  resetPerspectiveOption.binding(PROP_RESET_PERSPECTIVE.toStdString());
  options.addOption(resetPerspectiveOption);

  Poco::Util::Option productNameOption("product-name", "N",
      "\n"
      "The product name that appears in the title of the application window.\n");
  productNameOption.argument("<product name>").binding(PROP_PRODUCT_NAME.toStdString());
  options.addOption(productNameOption);
}


//-----------------------------------------------------------------------------
void BaseApplication::PrintVersion(const std::string& /*name*/, const std::string& /*value*/)
{
  // This stuff gets generated during CMake into NifTKConfigure.h
  std::string platformName(NIFTK_PLATFORM);
  std::string versionNumber(NIFTK_VERSION_STRING);
  std::string copyrightText(NIFTK_COPYRIGHT);
  std::string niftkVersion(NIFTK_VERSION);
  std::string niftkDateTime(NIFTK_DATE_TIME);

  std::cout << platformName << " " << versionNumber << " (" << niftkVersion << ")" << std::endl;
  std::cout << "Built at: " << niftkDateTime << std::endl;
  std::cout << copyrightText << std::endl;

  std::exit(EXIT_OK);
}


//-----------------------------------------------------------------------------
void BaseApplication::HandleRepeatableOption(const std::string& name, const std::string& value)
{
  QString propertyName = "applicationArgs.";
  propertyName.append(QString::fromStdString(name));

  QStringList valueList = this->getProperty(propertyName).toStringList();
  valueList.append(QString::fromStdString(value));
  this->setProperty(propertyName, QVariant::fromValue(valueList));
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
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
      "The following command opens 'image.nii.gz' as 'T1', 'left-hippocampus.nii.gz' as "
      "'l-hippo' and 'right-hippocampus.nii.gz' as 'r-hippo', and it makes the segmentations "
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
      "The following command opens a grey scale image with the 'MIDAS Spectrum' lookup table.\n"
      "\n"
      "    ${commandName} -o T1:image1.nii.gz --property T1:LookupTableName=\"MIDAS Spectrum\"\n"
      "\n"
      "The following command opens a reference image and the segmentation of the left and right "
      "hippocampi and sets the colour of the segmentations to midnight blue and dark violet, respectively.\n"
      "\n"
      "    ${commandName} -o T1:image.nii.gz -o LHippo:left-hc.nii.gz -o RHippo:right-hc.nii.gz \\\n"
      "        -p LHippo:color=midnightblue -p RHippo:color=darkviolet\n"
      "\n"
      "The following command opens a reference and a segmentation image and sets the colour of "
      "the segmentation to cornflower blue by its RGB components:\n"
      "\n"
      "    ${commandName} -o T1:image.nii.gz -o mask:segmentation.nii.gz -p mask:color=\\#6495ed\n"
      "\n"
      "Note that you need to protect the hashmark with a backslash when using a Unix shell.\n"
      "\n"
      "The following command opens a reference image and a mask, sets the intensity range of "
      "the reference image to 100-3500, disables the feature of outlining the binary images "
      "so that the mask is rendered as a solid layer rather than a contour, and sets the opacity "
      "to make it transparent.\n"
      "\n"
      "    ${commandName} -o T1:image.nii.gz -o mask:segmentation.nii.gz -d T1:mask \\\n"
      "        -p T1:levelwindow=100-3500 -p mask:outline\\ binary=false,opacity=0.3\n"
      "\n"
      "The following command opens two grey scale images and switches off the texture interpolation for both.\n"
      "\n"
      "    ${commandName} -o T1-1:image1.nii.gz -o T1-2:image2.nii.gz \\\n"
      "        -p T1-1,T1-2:texture\\ interpolation=off\n"
      "\n"
      "The following command opens ${commandName} in 'Minimal' perspective, in which you have "
      "only the viewer and the Data Manager open. It also sets the title of the application window.\n"
      "\n"
      "    ${commandName} --perspective Minimal --title \"Clinical Trial 1\"\n"
      "";
  examples.replace("${commandName}", QString::fromStdString(this->commandName()));

  helpFormatter->setFooter(examples.toStdString());
  helpFormatter->setUnixStyle(true);
  helpFormatter->setIndent(4);
  helpFormatter->setWidth(160);

  return helpFormatter;
}


//-----------------------------------------------------------------------------
void BaseApplication::PrintHelp(const std::string& /*name*/, const std::string& /*value*/)
{
  Poco::Util::HelpFormatter* helpFormatter = this->CreateHelpFormatter();

  helpFormatter->format(std::cout);

  delete helpFormatter;

  std::exit(EXIT_OK);
}

}
