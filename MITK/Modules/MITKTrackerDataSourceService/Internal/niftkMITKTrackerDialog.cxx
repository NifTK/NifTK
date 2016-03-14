/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDialog.h"
#include <QSettings>
#include <qextserialenumerator.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKTrackerDialog::MITKTrackerDialog(QWidget *parent, QString trackerName)
  : IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_TrackerName = trackerName;

  // Enumerate the available ports, so they have a natural name rather than "COM1", "COM2".
  QStringList ports = getAvailableSerialPorts();
  for (int i = 0; i < ports.count(); i++)
  {
    m_PortName->addItem(ports.at(i));
  }

  bool ok = false;
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);

  std::string id = "uk.ac.ucl.cmic.niftkMITKTrackerDataSourceService.MITKTrackerDialog";
  if (false && this->GetPeristenceService()) // Does not load first time, does not load on Windows - MITK bug?
  {
    std::string portName;
    std::string fileName;
    mitk::PropertyList::Pointer propList = this->GetPeristenceService()->GetPropertyList(id);
    if (propList.IsNull())
    {
      MITK_WARN << "Property list for (" << id << ") is not available!";
      return;
    }

    propList->Get("port", portName);
    propList->Get("file", fileName);

    int position = m_PortName->findText(QString::fromStdString(portName));
    if (position != -1)
    {
      m_PortName->setCurrentIndex(position);
    }
    m_FileOpen->setCurrentPath(QString::fromStdString(fileName));
  }
  else
  {
    QSettings settings;
    settings.beginGroup(QString::fromStdString(id));

    int position = m_PortName->findText(settings.value("port", "").toString());
    if (position != -1)
    {
      m_PortName->setCurrentIndex(position);
    }
    m_FileOpen->setCurrentPath(settings.value("file", "").toString());

    settings.endGroup();
  }
}


//-----------------------------------------------------------------------------
MITKTrackerDialog::~MITKTrackerDialog()
{
  bool ok = false;
  ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void MITKTrackerDialog::OnOKClicked()
{
  int currentSelection = m_PortName->currentIndex() + 1;

  IGIDataSourceProperties props;
  props.insert("port", QVariant::fromValue(currentSelection));
  props.insert("file", QVariant::fromValue(m_FileOpen->currentPath()));
  m_Properties = props;

  std::string id = "uk.ac.ucl.cmic.niftkMITKTrackerDataSourceService.MITKTrackerDialog";
  if (false && this->GetPeristenceService()) // Does not load first time, does not load on Windows - MITK bug?
  {
    mitk::PropertyList::Pointer propList = this->GetPeristenceService()->GetPropertyList(id);
    propList->Set("port", m_PortName->currentText().toStdString());
    propList->Set("file", m_FileOpen->currentPath().toStdString());
  }
  else
  {
    QSettings settings;
    settings.beginGroup(QString::fromStdString(id));
    settings.setValue("port", m_PortName->currentText());
    settings.setValue("file", m_FileOpen->currentPath());
    settings.endGroup();
  }
}

} // end namespace

