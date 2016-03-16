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
#include <qdebug>

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
  QStringList portPaths = getAvailableSerialPortPaths();
  for (int i = 0; i < ports.count(); i++)
  {
#ifdef _WIN32
    m_PortName->addItem(ports.at(i), QVariant::fromValue(ports.at(i).right(1)));
#else
    m_PortName->addItem(ports.at(i), QVariant::fromValue(portPaths.at(i)));
#endif
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

    int position = m_PortName->findData(QVariant(QString::fromStdString(portName)));
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

    int position = m_PortName->findData(QVariant(settings.value("port", "")));
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
  IGIDataSourceProperties props;
#if QT_VERSION >= 0x050200 // Because currentData() was introduced in Qt 5.2
  props.insert("port", QVariant::fromValue(m_PortName->currentData().toString()));
#else
  props.insert("port", QVariant::fromValue(m_PortName->itemData(m_PortName->currentIndex())));
#endif
  props.insert("file", QVariant::fromValue(m_FileOpen->currentPath()));
  m_Properties = props;

  std::string id = "uk.ac.ucl.cmic.niftkMITKTrackerDataSourceService.MITKTrackerDialog";
  if (false && this->GetPeristenceService()) // Does not load first time, does not load on Windows - MITK bug?
  {
    mitk::PropertyList::Pointer propList = this->GetPeristenceService()->GetPropertyList(id);
#if QT_VERSION >= 0x050200 // Because currentData() was introduced in Qt 5.2
    propList->Set("port", m_PortName->currentData().toString().toStdString());
#else
    propList->Set("port", m_PortName->currentText().toStdString());
#endif
    propList->Set("file", m_FileOpen->currentPath().toStdString());
  }
  else
  {
    QSettings settings;
    settings.beginGroup(QString::fromStdString(id));
#if QT_VERSION >= 0x050200 // Because currentData() was introduced in Qt 5.2
    settings.setValue("port", m_PortName->currentData().toString());
#else
    settings.setValue("port", m_PortName->itemData(m_PortName->currentIndex()));
#endif
    settings.setValue("file", m_FileOpen->currentPath());
    settings.endGroup();
  }
}

} // end namespace

