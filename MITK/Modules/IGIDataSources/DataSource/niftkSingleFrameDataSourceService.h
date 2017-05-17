/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkSingleFrameDataSourceService_h
#define niftkSingleFrameDataSourceService_h

#include <niftkIGIDataSourcesExports.h>
#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>
#include <niftkIGILocalDataSourceI.h>
#include <niftkIGIDataSourceRingBuffer.h>
#include <mitkImage.h>

#include <QObject>
#include <QMutex>
#include <QString>
#include <memory>

namespace niftk
{

/**
* \class SingleFrameDataSourceService
* \brief Base class for simple data sources, that save frame by frame.
* For example, we save each image frame as .jpg/.png rather than some video format like .h264.
* \see OpenCVVideoDataSourceService
* \see QtCameraVideoDataSourceService
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT SingleFrameDataSourceService
    : public QObject
    , public IGIDataSource
    , public IGILocalDataSourceI
{

public:

  mitkClassMacroItkParent(SingleFrameDataSourceService, IGIDataSource)

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore) override;

  /**
  * \see  IGIDataSourceI::StartPlayback()
  */
  virtual void StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                             niftk::IGIDataSourceI::IGITimeType lastTimeStamp) override;

  /**
  * \see IGIDataSourceI::StopPlayback()
  */
  virtual void StopPlayback() override;

  /**
  * \see IGIDataSourceI::PlaybackData()
  */
  void PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp) override;

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
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataSourceI::IGITimeType& time) override;

  /**
  * \see niftk::IGILocalDataSourceI::GrabData()
  */
  virtual void GrabData() override;

protected:

  SingleFrameDataSourceService(QString deviceName,
                               QString factoryName,
                               unsigned int framesPerSecond,
                               unsigned int bufferSize,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                              );
  virtual ~SingleFrameDataSourceService();

  /**
   * \brief Derived classes implement this to grab a new image.
   *
   * The GrabImage method functions like a factory, returning a new,
   * heap allocated data type.
   */
  virtual std::unique_ptr<niftk::IGIDataType> GrabImage() = 0;

  /**
   * \brief Derived classes must implement this to convert the IGIDataType to an mitk::Image.
   */
  virtual mitk::Image::Pointer RetrieveImage(const niftk::IGIDataSourceI::IGITimeType& requested,
                                             niftk::IGIDataSourceI::IGITimeType& actualTime,
                                             unsigned int& outputNumberOfBytes) = 0;

  /**
   * \brief Derived classes must save the item to the given filename.
   */
  virtual void SaveImage(const std::string& filename, niftk::IGIDataType& item) = 0;

  /**
   * \brief Derived classes must load the image at the given filename.
   *
   * The GrabImage method functions like a factory, returning a new,
   * heap allocated data type.
   */
  virtual std::unique_ptr<niftk::IGIDataType> LoadImage(const std::string& filename) = 0;

  int GetChannelNumber() const                              { return m_ChannelNumber;}
  int GetApproximateIntervalInMilliseconds() const          { return m_ApproxIntervalInMilliseconds; }
  void SetApproximateIntervalInMilliseconds(const int& ms);

  static niftk::IGIDataSourceLocker                         s_Lock;
  niftk::IGIDataSourceRingBuffer                            m_Buffer;

private:

  SingleFrameDataSourceService(const SingleFrameDataSourceService&); // deliberately not implemented
  SingleFrameDataSourceService& operator=(const SingleFrameDataSourceService&); // deliberately not impl'd.

  void SaveItem(niftk::IGIDataType& item);

  int                                          m_ChannelNumber;
  niftk::IGIDataSourceI::IGIIndexType          m_FrameId;
  std::set<niftk::IGIDataSourceI::IGITimeType> m_PlaybackIndex;
  int                                          m_ApproxIntervalInMilliseconds;
  QString                                      m_FileExtension;

}; // end class

} // end namespace

#endif
