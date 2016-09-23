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

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
Q_DECLARE_METATYPE(QAudioFormat);
#endif

namespace niftk
{

//-----------------------------------------------------------------------------
QtAudioDataDialog::QtAudioDataDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);

  m_DeviceComboBox->clear();
  m_FormatComboBox->clear();

  QAudioDeviceInfo  defaultDevice = QAudioDeviceInfo::defaultInputDevice();

  QList<QAudioDeviceInfo> allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
  foreach(QAudioDeviceInfo d, allDevices)
  {
    m_DeviceComboBox->addItem(d.deviceName());
  }
  m_DeviceComboBox->setCurrentIndex(m_DeviceComboBox->findText(defaultDevice.deviceName()));
  this->Update();

  bool ok = QObject::connect(m_DeviceComboBox, SIGNAL(currentIndexChanged(const QString&)),
                             this, SLOT(OnCurrentDeviceIndexChanged()));
  assert(ok);
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
QtAudioDataDialog::~QtAudioDataDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()),
                                this, SLOT(OnOKClicked()));
  assert(ok);
  ok = QObject::disconnect(m_DeviceComboBox, SIGNAL(currentIndexChanged(const QString&)),
                           this, SLOT(OnCurrentDeviceIndexChanged()));
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

  QAudioDeviceInfo selectedDevice;
  QList<QAudioDeviceInfo> allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
  foreach(QAudioDeviceInfo d, allDevices)
  {
    if (d.deviceName() == m_DeviceComboBox->currentText())
    {
      selectedDevice = d;
      QList<int> channelCounts = d.supportedChannelCounts();
      foreach(int c, channelCounts)
      {
        QList<int> sampleRates = d.supportedSampleRates();
        qSort(sampleRates);
        foreach(int r, sampleRates)
        {
          QList<int> sampleSizes = d.supportedSampleSizes();
          qSort(sampleSizes);
          for (int s = 0; s < sampleSizes.size(); ++s)
          {
            // don't bother with 8 bit, it sounds like trash.
            if (sampleSizes[s] <= 8)
              continue;

            // FIXME: we should probably restrict the codec to pcm, in general.
            QStringList codecs = d.supportedCodecs();
            foreach(QString m, codecs)
            {
              QAudioFormat f;
              f.setChannelCount(c);
              f.setSampleRate(r);
              f.setCodec(m);
              f.setSampleSize(sampleSizes[s]);
              f.setSampleType(d.preferredFormat().sampleType());

              if (d.isFormatSupported(f))
              {
                QString text = QString("%1 channels @ %2 Hz, %3 bit, %4")
                    .arg(f.channelCount())
                    .arg(f.sampleRate())
                    .arg(f.sampleSize())
                    .arg(f.codec());

                m_FormatComboBox->addItem(text, QVariant::fromValue(f));
              }
            }
          }
        }
      }
    }
  }
  QAudioFormat defaultFormat = selectedDevice.preferredFormat();
  QString defaultFormatText = QString("%1 channels @ %2 Hz, %3 bit, %4")
      .arg(defaultFormat.channelCount())
      .arg(defaultFormat.sampleRate())
      .arg(defaultFormat.sampleSize())
      .arg(defaultFormat.codec());
  m_FormatComboBox->setCurrentIndex(m_FormatComboBox->findText(defaultFormatText));
  m_FormatComboBox->blockSignals(false);
}


//-----------------------------------------------------------------------------
void QtAudioDataDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("name", QVariant::fromValue(m_DeviceComboBox->currentText()));
  props.insert("format", QVariant::fromValue(m_FormatComboBox->itemData(m_FormatComboBox->currentIndex())));
  props.insert("formatString", QVariant::fromValue(m_FormatComboBox->currentText()));
  m_Properties = props;
}

} // end namespace
