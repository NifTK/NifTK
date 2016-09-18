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
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKTrackerDialog::MITKTrackerDialog(QWidget *parent,
                                     QString trackerName,
                                     int defaultBaudRate)
  : IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_TrackerName = trackerName;

  m_BaudRate->addItem(QString::number(9600), QVariant::fromValue(9600));
  m_BaudRate->addItem(QString::number(14400), QVariant::fromValue(14400));
  m_BaudRate->addItem(QString::number(19200), QVariant::fromValue(19200));
  m_BaudRate->addItem(QString::number(38400), QVariant::fromValue(38400));
  m_BaudRate->addItem(QString::number(57600), QVariant::fromValue(57600));
  m_BaudRate->addItem(QString::number(115200), QVariant::fromValue(115200));
  m_BaudRate->addItem(QString::number(921600), QVariant::fromValue(921600));
  m_BaudRate->addItem(QString::number(1228739), QVariant::fromValue(1228739));

  int position = m_BaudRate->findData(QVariant(defaultBaudRate));
  if (position != -1)
  {
    m_BaudRate->setCurrentIndex(position);
  }
  else
  {
    mitkThrow() << "Invalid baud rate specified:" << defaultBaudRate;
  }

  // Enumerate the available ports, so they have a natural name rather than "COM1", "COM2".
  QStringList ports = getAvailableSerialPorts();
#if !defined(_WIN32) && !defined(__APPLE__)
  QStringList portPaths = getAvailableSerialPortPaths();
#endif
  for (int i = 0; i < ports.count(); i++)
  {
#ifdef _WIN32
    // On windows, we want to see "COM1", but we need the number at the end to connect via port number.
    QString comString = ports.at(i);
    comString.remove("com",Qt::CaseInsensitive);
#elif !defined(__APPLE__)
    QString comString = portPaths.at(i);
#endif
    m_PortName->addItem(ports.at(i), QVariant::fromValue(comString));
  }

  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);

  std::string id = "uk.ac.ucl.cmic.niftkMITKTrackerDataSourceService.MITKTrackerDialog";
  if (false && this->GetPeristenceService()) // Does not load first time, does not load on Windows - MITK bug?
  {
    std::string fileName;
    std::string portName;
    std::string baudRate;

    mitk::PropertyList::Pointer propList = this->GetPeristenceService()->GetPropertyList(id);
    if (propList.IsNull())
    {
      MITK_WARN << "Property list for (" << id << ") is not available!";
      return;
    }

    propList->Get("file", fileName);
    propList->Get("port", portName);
    propList->Get("baudRate", baudRate);

    m_FileOpen->setCurrentPath(QString::fromStdString(fileName));

    int position = m_PortName->findData(QVariant(QString::fromStdString(portName)));
    if (position != -1)
    {
      m_PortName->setCurrentIndex(position);
    }

    position = m_BaudRate->findData(QVariant(QString::fromStdString(baudRate)));
    if (position != -1)
    {
      m_BaudRate->setCurrentIndex(position);
    }
  }
  else
  {
    QSettings settings;
    settings.beginGroup(QString::fromStdString(id));

    m_FileOpen->setCurrentPath(settings.value("file", "").toString());

    int position = m_PortName->findData(QVariant(settings.value("port", "")));
    if (position != -1)
    {
      m_PortName->setCurrentIndex(position);
    }

    position = m_BaudRate->findData(QVariant(settings.value("baudRate", "")));
    if (position != -1)
    {
      m_BaudRate->setCurrentIndex(position);
    }

    settings.endGroup();
  }
}


//-----------------------------------------------------------------------------
MITKTrackerDialog::~MITKTrackerDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void MITKTrackerDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("file", QVariant::fromValue(m_FileOpen->currentPath()));
  props.insert("port", QVariant::fromValue(m_PortName->itemData(m_PortName->currentIndex())));
  props.insert("baudRate", QVariant::fromValue(m_BaudRate->itemData(m_BaudRate->currentIndex())));

  m_Properties = props;

  std::string id = "uk.ac.ucl.cmic.niftkMITKTrackerDataSourceService.MITKTrackerDialog";
  if (false && this->GetPeristenceService()) // Does not load first time, does not load on Windows - MITK bug?
  {
    mitk::PropertyList::Pointer propList = this->GetPeristenceService()->GetPropertyList(id);
    propList->Set("file", m_FileOpen->currentPath().toStdString());
    propList->Set("port", m_PortName->itemData(m_PortName->currentIndex()).toString().toStdString());
    propList->Set("baudRate", m_BaudRate->itemData(m_BaudRate->currentIndex()).toString().toStdString());
  }
  else
  {
    QSettings settings;
    settings.beginGroup(QString::fromStdString(id));
    settings.setValue("file", m_FileOpen->currentPath());
    settings.setValue("port", m_PortName->itemData(m_PortName->currentIndex()).toString());
    settings.setValue("baudRate", m_BaudRate->itemData(m_BaudRate->currentIndex()).toString());
    settings.endGroup();
  }
}

} // end namespace
