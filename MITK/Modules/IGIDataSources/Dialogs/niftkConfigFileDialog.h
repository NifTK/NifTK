/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkConfigFileDialog_h
#define niftkConfigFileDialog_h

#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkConfigFileDialog.h>

namespace niftk
{

class ConfigFileDialog : public IGIInitialisationDialog,
                         public Ui_niftkConfigFileDialog

{
  Q_OBJECT

public:

  ConfigFileDialog(QWidget *parent,
                   const QString& trackerName,
                   const QString& settingsName);
  ~ConfigFileDialog();

private slots:

  void OnOKClicked();

private:
  QString m_TrackerName;
  QString m_SettingsName;
};

} // end namespace

#endif
