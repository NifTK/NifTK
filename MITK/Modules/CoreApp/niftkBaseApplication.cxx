/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBaseApplication.h>

#include <QString>
#include <QStringList>
#include <QVariant>

namespace niftk
{

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

  Poco::Util::Option openOption("open", "o", "opens a file with the given name");
  openOption.argument("<name>:<file>").repeatable(true);
  openOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &BaseApplication::HandleRepeatableOption));
  options.addOption(openOption);

  Poco::Util::Option perspectiveOption("perspective", "", "the initial window perspective");
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


const QString BaseApplication::PROP_OPEN = "applicationArgs.open";
const QString BaseApplication::PROP_PERSPECTIVE = "applicationArgs.perspective";

}
