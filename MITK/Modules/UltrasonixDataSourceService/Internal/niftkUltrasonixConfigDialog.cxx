/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasonixConfigDialog.h"

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
UltrasonixConfigDialog::UltrasonixConfigDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer service)
:IGIConfigurationDialog(parent, service)
{
  setupUi(this);
  m_DataTypeComboBox->clear();
  m_DataTypeComboBox->addItem(QString("udtScreen"), QVariant(0x00000001));
  m_DataTypeComboBox->addItem(QString("udtBPre"), QVariant(0x00000002));
  m_DataTypeComboBox->addItem(QString("udtBPost (8 bit)"), QVariant(0x00000004));
  m_DataTypeComboBox->addItem(QString("udtBPost32 (32 bit)"), QVariant(0x00000008));
  m_DataTypeComboBox->addItem(QString("udtMPre"), QVariant(0x00000020));
  m_DataTypeComboBox->addItem(QString("udtMPost"), QVariant(0x00000040));
  m_DataTypeComboBox->addItem(QString("udtPWRF"), QVariant(0x00000080));
  m_DataTypeComboBox->addItem(QString("udtPWSpectrum"), QVariant(0x00000100));
  m_DataTypeComboBox->addItem(QString("udtColorRF"), QVariant(0x00000200));
  m_DataTypeComboBox->addItem(QString("udtColorCombined"), QVariant(0x00000400));
  m_DataTypeComboBox->addItem(QString("udtColorVelocityVariance"), QVariant(0x00000800));
  m_DataTypeComboBox->addItem(QString("udtContrast"), QVariant(0x00001000));
  m_DataTypeComboBox->addItem(QString("udtElastoCombined (32 bit)"), QVariant(0x00002000));
  m_DataTypeComboBox->addItem(QString("udtElastoOverlay (8 bit)"), QVariant(0x00004000));
  m_DataTypeComboBox->addItem(QString("udtElastoPre (8 bit)"), QVariant(0x00008000));
  m_DataTypeComboBox->addItem(QString("udtECG"), QVariant(0x00010000));
  m_DataTypeComboBox->setCurrentIndex(2);

  IGIDataSourceProperties props = m_Service->GetProperties();
  if (props.contains("lag"))
  {
    m_LagSpinBox->setValue(props.value("lag").toInt());
  }
  if (props.contains("uData"))
  {
    m_DataTypeComboBox->setCurrentIndex(m_DataTypeComboBox->findData(props.value("uData")));
  }
  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
UltrasonixConfigDialog::~UltrasonixConfigDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void UltrasonixConfigDialog::OnOKClicked()
{
  QMap<QString, QVariant> props;
  props.insert("lag", QVariant::fromValue(m_LagSpinBox->value()));
  props.insert("uData", QVariant::fromValue(m_DataTypeComboBox->itemData(m_DataTypeComboBox->currentIndex())));
  m_Service->SetProperties(props);
}

} // end namespace
