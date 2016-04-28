/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkHelpAboutDialog_h
#define niftkHelpAboutDialog_h

#include "niftkCoreGuiExports.h"
#include "ui_niftkHelpAboutDialog.h"
#include <QDialog>

namespace niftk
{

/**
* \class HelpAboutDialog
* \brief Prints out a useful About dialog with software version numbers.
*/
class NIFTKCOREGUI_EXPORT HelpAboutDialog : public QDialog, public Ui_HelpAboutDialog {

  Q_OBJECT

public:

  /**
  * \brief Constructor with additional name parameter.
  */
  HelpAboutDialog(QWidget *parent, QString applicationName);

  /**
  * \brief Destructor.
  */
  virtual ~HelpAboutDialog();

private:

  HelpAboutDialog(const HelpAboutDialog&);  // Purposefully not implemented.
  void operator=(const HelpAboutDialog&);  // Purposefully not implemented.

  // Generates the text, taking the application name, and internally lots of values configured from NifTKConfigure.h
  void GenerateHelpAboutText(QString applicationName);

  QString m_ApplicationName;
};

} // end namespace

#endif
