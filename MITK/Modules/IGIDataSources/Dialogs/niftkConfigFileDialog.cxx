/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkConfigFileDialog.h"
#include <QSettings>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
ConfigFileDialog::ConfigFileDialog(QWidget *parent,
                                   const QString& trackerName,
                                   const QString& settingsName
                                  )
: IGIInitialisationDialog(parent)
, m_TrackerName(trackerName)
, m_SettingsName(settingsName)
{
  setupUi(this);

  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);

  QSettings settings;
  settings.beginGroup(m_SettingsName);

  m_FileOpen->setCurrentPath(settings.value("file", "").toString());

  settings.endGroup();
}


//-----------------------------------------------------------------------------
ConfigFileDialog::~ConfigFileDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void ConfigFileDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("file", QVariant::fromValue(m_FileOpen->currentPath()));

  m_Properties = props;

  QSettings settings;
  settings.beginGroup(m_SettingsName);
  settings.setValue("file", m_FileOpen->currentPath());
  settings.endGroup();
}

} // end namespace
