/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtAudioDataDialog.h"

#include <QAudioDeviceInfo>
#include <QAudioInput>

#include <cassert>

Q_DECLARE_METATYPE(QAudioFormat);

namespace niftk
{

//-----------------------------------------------------------------------------
QtAudioDataDialog::QtAudioDataDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);

  m_DeviceComboBox->clear();
  m_FormatComboBox->clear();

  QList<QAudioDeviceInfo> allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
  foreach(QAudioDeviceInfo d, allDevices)
  {
    m_DeviceComboBox->addItem(d.deviceName());
  }
  m_DeviceComboBox->setCurrentIndex(0);
  this->Update();

  bool ok = false;
  ok = QObject::connect(m_DeviceComboBox, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(OnCurrentDeviceIndexChanged()));
  assert(ok);
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
QtAudioDataDialog::~QtAudioDataDialog()
{
  bool ok = false;
  ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
  ok = QObject::disconnect(m_DeviceComboBox, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(OnCurrentDeviceIndexChanged()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void QtAudioDataDialog::OnCurrentDeviceIndexChanged()
{
  this->Update();
}


//-----------------------------------------------------------------------------
void QtAudioDataDialog::Update()
{
  m_FormatComboBox->blockSignals(true);
  m_FormatComboBox->clear();

  QList<QAudioDeviceInfo> allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
  foreach(QAudioDeviceInfo d, allDevices)
  {
    if (d.deviceName() == m_DeviceComboBox->currentText())
    {
      QList<int> channelCounts = d.supportedChannelCounts();
      foreach(int c, channelCounts)
      {
        QList<int> sampleRates = d.supportedSampleRates();
        foreach(int r, sampleRates)
        {
          QList<int> sampleSizes = d.supportedSampleSizes();
          qSort(sampleSizes);
          for (int s = 0; s < sampleSizes.size(); ++s)
          {
            // don't bother with 8 bit, it sounds like trash.
            if ((s <= 8) && (s < sampleSizes.size() - 1))
              // this breaks if for example 8 appears multiple times.
              continue;

            // FIXME: we should probably restrict the codec to pcm, in general.
            QStringList codecs = d.supportedCodecs();
            foreach(QString m, codecs)
            {
              QAudioFormat f;
              f.setChannels(c);
              f.setSampleRate(r);
              f.setCodec(m);
              f.setSampleSize(sampleSizes[s]);
              f.setSampleType(d.preferredFormat().sampleType());

              if (d.isFormatSupported(f))
              {
                QString text = QString("%1 channels @ %2 Hz, %3 bit, %4").arg(f.channels()).arg(f.sampleRate()).arg(f.sampleSize()).arg(f.codec());
                m_FormatComboBox->addItem(text, QVariant::fromValue(f));
              }
            }
          }
        }
      }
    }
  }

  m_FormatComboBox->blockSignals(false);
}


//-----------------------------------------------------------------------------
void QtAudioDataDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("name", QVariant::fromValue(m_DeviceComboBox->currentText()));
  props.insert("format", QVariant::fromValue(m_FormatComboBox->itemData(m_FormatComboBox->currentIndex())));
  m_Properties = props;
}

} // end namespace
