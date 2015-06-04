/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "AudioDataSourceGui.h"
#include "AudioDataSource.h"
#include "QmitkIGIDataSourceMacro.h"
#include <QList>
#include <QAudioDeviceInfo>
#include <QAudioInput>


Q_DECLARE_METATYPE(QAudioFormat);

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIDATASOURCES_EXPORT, AudioDataSourceGui, "IGI Audio Source Gui")


//-----------------------------------------------------------------------------
AudioDataSourceGui::AudioDataSourceGui()
{
}


//-----------------------------------------------------------------------------
AudioDataSourceGui::~AudioDataSourceGui()
{
}


//-----------------------------------------------------------------------------
void AudioDataSourceGui::Update()
{
  AudioDataSource::Pointer  src = dynamic_cast<AudioDataSource*>(GetSource());
  if (src.IsNotNull())
  {
    const QAudioDeviceInfo*   device  = src->GetDeviceInfo();
    const QAudioFormat*       format  = src->GetFormat();

    // find entry that matches our device.
    {
      int  foundDeviceEntry = -1;
      for (int i = 0; i < m_DeviceComboBox->count(); ++i)
      {
        if (m_DeviceComboBox->itemText(i) == device->deviceName())
        {
          foundDeviceEntry = i;
          break;
        }
      }
      if (foundDeviceEntry >= 0)
      {
        if (m_DeviceComboBox->currentIndex() != foundDeviceEntry)
        {
          m_DeviceComboBox->setCurrentIndex(foundDeviceEntry);
          m_FormatComboBox->clear();
        }
      }
      else
      {
        m_DeviceComboBox->addItem(device->deviceName());
        m_FormatComboBox->clear();
      }
    }

    // sort out format.
    {
      if (m_FormatComboBox->count() == 0)
      {
        m_FormatComboBox->blockSignals(true);

        // this kinda sucks...
        QList<int>  channelCounts = device->supportedChannelCounts();
        foreach(int c, channelCounts)
        {
          QList<int>  sampleRates = device->supportedSampleRates();
          foreach(int r, sampleRates)
          {
            QList<int>  sampleSizes = device->supportedSampleSizes();
            qSort(sampleSizes);
            for (int s = 0; s < sampleSizes.size(); ++s)
            {
              // don't bother with 8 bit, it sounds like trash.
              if ((s <= 8) && (s < sampleSizes.size() - 1))
                // this breaks if for example 8 appears multiple times.
                continue;

              // FIXME: we should probably restrict the codec to pcm, in general.
              QStringList   codecs = device->supportedCodecs();
              foreach(QString m, codecs)
              {
                QAudioFormat    f;
                f.setChannels(c);
                f.setSampleRate(r);
                f.setCodec(m);
                f.setSampleSize(sampleSizes[s]);
                f.setSampleType(device->preferredFormat().sampleType());

                if (device->isFormatSupported(f))
                {
                  QString   text = AudioDataSource::formatToString(&f);
                  m_FormatComboBox->addItem(text, QVariant::fromValue(f));
                }
              }
            }
          }
        }

        m_FormatComboBox->blockSignals(false);
      }
      QString currentFormatText = AudioDataSource::formatToString(format);
      int   foundFormatEntry = -1;
      for (int i = 0; i < m_FormatComboBox->count(); ++i)
      {
        if (m_FormatComboBox->itemText(i) == currentFormatText)
        {
          foundFormatEntry = i;
          break;
        }
      }
      if (foundFormatEntry >= 0)
      {
        if (m_FormatComboBox->currentIndex() != foundFormatEntry)
        {
          m_FormatComboBox->setCurrentIndex(foundFormatEntry);
        }
      }
      else
      {
        // this else-branch can happen if the default format has not been added in the above loop.
        // for example, we skip 8 bit, but the default-format-selection happens to pick an 8 bit format.
        QString   text = AudioDataSource::formatToString(format);
        m_FormatComboBox->addItem(text, QVariant::fromValue(*format));
      }
    }

    m_DeviceComboBox->setEnabled(!src->IsRecording());
    m_FormatComboBox->setEnabled(!src->IsRecording());
  }
}


//-----------------------------------------------------------------------------
void AudioDataSourceGui::Initialize(QWidget* parent)
{
  // BEWARE: parent is always null!

  setupUi(this);

  m_DeviceComboBox->clear();
  m_FormatComboBox->clear();

  QList<QAudioDeviceInfo>   allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
  foreach(QAudioDeviceInfo d, allDevices)
  {
    m_DeviceComboBox->addItem(d.deviceName());
  }

  // format combobox depends on the selected device.
  Update();

  // signals are connected after init, avoids unnecessary roundtrips.
  bool    ok = false;
  ok = QObject::connect(m_DeviceComboBox, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(OnCurrentDeviceIndexChanged(const QString&)));
  assert(ok);
  ok = QObject::connect(m_FormatComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnCurrentFormatIndexChanged(int)));
  assert(ok);
}


//-----------------------------------------------------------------------------
void AudioDataSourceGui::OnCurrentDeviceIndexChanged(const QString& text)
{
  QList<QAudioDeviceInfo>   allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
  foreach(QAudioDeviceInfo d, allDevices)
  {
    if (d.deviceName() == text)
    {
      // format combobox needs re-populating.
      // that will be done in due time via Update().
      m_FormatComboBox->clear();

      AudioDataSource::Pointer  src = dynamic_cast<AudioDataSource*>(GetSource());
      if (src.IsNotNull())
      {
        QAudioFormat format = d.preferredFormat();
        src->SetAudioDevice(&d, &format);
      }

      break;
    }
  }
}


//-----------------------------------------------------------------------------
void AudioDataSourceGui::OnCurrentFormatIndexChanged(int index)
{
  QVariant v = m_FormatComboBox->itemData(index);
  if (v.isValid())
  {
    QAudioFormat    format(v.value<QAudioFormat>());
    assert(m_FormatComboBox->itemText(index) == AudioDataSource::formatToString(&format));

    AudioDataSource::Pointer  src = dynamic_cast<AudioDataSource*>(GetSource());
    if (src.IsNotNull())
    {
      QList<QAudioDeviceInfo>   allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
      foreach(QAudioDeviceInfo d, allDevices)
      {
        if (d.deviceName() == m_DeviceComboBox->currentText())
        {
          src->SetAudioDevice(&d, &format);
          break;
        }
      }
    }
  }
}
