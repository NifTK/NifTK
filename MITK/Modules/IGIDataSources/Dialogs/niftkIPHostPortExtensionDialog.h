/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIPHostPortExtensionDialog_h
#define niftkIPHostPortExtensionDialog_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkIPHostPortExtensionDialog.h>
#include <QStringList>

namespace niftk
{

class NIFTKIGIDATASOURCES_EXPORT IPHostPortExtensionDialog : public IGIInitialisationDialog,
                                                             public Ui_niftkIPHostPortExtensionDialog

{
  Q_OBJECT

public:

  IPHostPortExtensionDialog(QWidget *parent,
                            const QString& settingsName,
                            const int& defaultPortNumber,
                            const QStringList& extensionNames,
                            const QStringList& extensionsWithDots
                           );

  ~IPHostPortExtensionDialog();

  void SetHostVisible(const bool& isVisible);
  void SetPortVisible(const bool& isVisible);
  void SetExtensionVisible(const bool& isVisible);

private slots:

  void OnOKClicked();

private:

  QString m_SettingsName;
  int     m_DefaultPortNumber;
};

} // end namespace

#endif
