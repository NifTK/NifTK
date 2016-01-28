/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDialog.h"
#include <mitkIPersistenceService.h>

namespace niftk
{

struct MITKTrackerDialogPersistenceClass
{
  MITKTrackerDialogPersistenceClass(QString name)
    : m_Id(name.toStdString())
    , m_Filename("")
    , m_PortName("")
  {}

  std::string m_Id;
  std::string m_Filename;
  std::string m_PortName;
  PERSISTENCE_CREATE2(MITKTrackerDialogPersistenceClass, m_Id, m_Filename, m_PortName)
};

//-----------------------------------------------------------------------------
MITKTrackerDialog::MITKTrackerDialog(QWidget *parent, QString trackerName)
  : IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_TrackerName = trackerName;
  m_PortName->addItem("COM1");
  m_PortName->addItem("COM2");
  m_PortName->addItem("COM3");
  m_PortName->addItem("COM4");
  m_PortName->addItem("COM5");
  m_PortName->addItem("COM6");
  m_PortName->addItem("COM7");
  m_PortName->addItem("COM8");
  m_PortName->addItem("COM9");
  m_PortName->addItem("COM10");
  m_PortName->addItem("COM11");
  m_PortName->addItem("COM12");
  m_PortName->addItem("COM13");

  MITKTrackerDialogPersistenceClass previouslySaved(m_TrackerName);
  previouslySaved.Load();

  int position = m_PortName->findText(QString::fromStdString(previouslySaved.m_PortName));
  if (position != -1)
  {
    m_PortName->setCurrentIndex(position);
  }
  m_FileOpen->setCurrentPath(QString::fromStdString(previouslySaved.m_Filename));

  bool ok = false;
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
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

  MITKTrackerDialogPersistenceClass stuffToSave(m_TrackerName);
  stuffToSave.m_Filename = m_FileOpen->currentPath().toStdString();
  stuffToSave.m_PortName = m_PortName->currentText().toStdString();
  stuffToSave.Save();
}

} // end namespace
