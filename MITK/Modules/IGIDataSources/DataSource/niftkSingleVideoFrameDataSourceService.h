/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkSingleVideoFrameDataSourceService_h
#define niftkSingleVideoFrameDataSourceService_h

#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>
#include <niftkIGIDataSourceBuffer.h>
#include <niftkIGILocalDataSourceI.h>
#include <niftkIGICleanableDataSourceI.h>
#include <niftkIGIDataSourceBackgroundDeleteThread.h>
#include <niftkIGIBufferedSaveableDataSourceI.h>
#include <mitkImage.h>

#include <QObject>
#include <QMutex>
#include <QString>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceService
* \brief Base class for simple (really meaning "test purposes only") video data sources.
* For example, we save each image frame as .jpg/.png rather than some video format like .h264.
* \see OpenCVVideoDataSourceService
* \see QtVideoDataSourceService
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class SingleVideoFrameDataSourceService
    : public QObject
    , public IGIDataSource
    , public IGILocalDataSourceI
    , public IGICleanableDataSourceI
    , public IGIBufferedSaveableDataSourceI
{

public:

  mitkClassMacroItkParent(SingleVideoFrameDataSourceService, IGIDataSource)

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataType::IGITimeType* lastTimeStampInStore) override;

  /**
  * \see  IGIDataSourceI::StartPlayback()
  */
  virtual void StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                             niftk::IGIDataType::IGITimeType lastTimeStamp) override;

  /**
  * \see IGIDataSourceI::StopPlayback()
  */
  virtual void StopPlayback() override;

  /**
  * \see IGIDataSourceI::PlaybackData()
  */
  void PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp) override;

  /**
  * \brief IGIDataSourceI::SetProperties()
  */
  virtual void SetProperties(const IGIDataSourceProperties& properties) override;

  /**
  * \brief IGIDataSourceI::GetProperties()
  */
  virtual IGIDataSourceProperties GetProperties() const override;

  /**
  * \see IGIDataSourceI::Update()
  */
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataType::IGITimeType& time) override;

  /**
  * \see niftk::IGICleanableDataSourceI::CleanBuffer()
  */
  virtual void CleanBuffer() override;

  /**
  * \see niftk::IGILocalDataSourceI::GrabData()
  */
  virtual void GrabData() override;

protected:

  SingleVideoFrameDataSourceService(QString deviceName,
                                    QString factoryName,
                                    const IGIDataSourceProperties& properties,
                                    mitk::DataStorage::Pointer dataStorage
                                   );
  virtual ~SingleVideoFrameDataSourceService();

  /**
   * \brief Derived classes implement this to grab a new image.
   */
  virtual niftk::IGIDataType::Pointer GrabImage() = 0;

  /**
   * \brief Derived classes must save the item to the given filename.
   */
  virtual void SaveImage(const std::string& filename, niftk::IGIDataType::Pointer item) = 0;

  /**
   * \brief Derived classes must load the image at the given filename.
   */
  virtual niftk::IGIDataType::Pointer LoadImage(const std::string& filename) = 0;

  /**
   * \brief Derived classes must implement this to convert the IGIDataType to an mitk::Image.
   */
  virtual mitk::Image::Pointer ConvertImage(niftk::IGIDataType::Pointer inputImage,
                                            unsigned int& outputNumberOfBytes) = 0;

  static niftk::IGIDataSourceLocker                s_Lock;
  int GetChannelNumber() const                     { return m_ChannelNumber;}
  int GetApproximateIntervalInMilliseconds() const { return m_ApproxIntervalInMilliseconds; }

private:

  SingleVideoFrameDataSourceService(const SingleVideoFrameDataSourceService&); // deliberately not implemented
  SingleVideoFrameDataSourceService& operator=(const SingleVideoFrameDataSourceService&); // deliberately not implemented

  void SaveItem(niftk::IGIDataType::Pointer item) override;

  QMutex                                          m_Lock;
  int                                             m_ChannelNumber;
  niftk::IGIDataType::IGIIndexType                m_FrameId;
  niftk::IGIDataSourceBuffer::Pointer             m_Buffer;
  niftk::IGIDataSourceBackgroundDeleteThread*     m_BackgroundDeleteThread;
  std::set<niftk::IGIDataType::IGITimeType>       m_PlaybackIndex;
  int                                             m_ApproxIntervalInMilliseconds;
  QString                                         m_FileExtension;

}; // end class

} // end namespace

#endif
