/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceI_h
#define niftkIGIDataSourceI_h

#include "niftkIGIDataSourcesExports.h"
#include <niftkIGIDataType.h>

#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace niftk
{

/**
* \class IGIDataItemInfo
* \brief Info class to describe current state, so that GUI can display status.
*
* This is per item. One Source (e.g. tracker), may return data from many items (eg. tools).
* So, each tool is considered an item. So the data source should return one
* of these IGIDataSourceInfo for each tool. Other sources such as a video
* source or framegrabber will probably only return one of these per frame. But in
* principle it could be any number from each source.
*
* Note: Deliberately not using Qt datatypes, so that an implementing class does not have to.
*
* The first implementing class was niftk::OpenCVVideoDataSourceService().
*/
struct NIFTKIGIDATASOURCES_EXPORT IGIDataItemInfo
{
  IGIDataItemInfo()
  {
    m_Name = "Unknown";
    m_Status = "Unknown";
    m_ShouldUpdate = false;
    m_IsLate = false;
    m_LagInMilliseconds = 0;
    m_FramesPerSecond = 0;
    m_Description = "Unknown";
  }

  std::string  m_Name;
  std::string  m_Status;
  bool         m_ShouldUpdate;
  bool         m_IsLate;
  unsigned int m_LagInMilliseconds;
  float        m_FramesPerSecond;
  std::string  m_Description;
};


/**
* \class IGIDataSourceI
* \brief Interface for an IGI Data Source (e.g. video feed, ultrasound feed, tracker feed).
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*
* Note: Deliberately not using Qt datatypes, so that an implementing class does not have to.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceI : public itk::Object
{

public:

  mitkClassMacroItkParent(IGIDataSourceI, itk::Object);

  /**
  * \brief Returns the unique name of the data source, e.g. OpenCV-0.
  *
  * Implementing classes should have this value set at construction,
  * and it should be immutable throughout the lifetime of the object.
  */
  virtual std::string GetName() const = 0;

  /**
  * \brief Returns the name of the factory that created it. The name
  * should be the string that appears in the GUI combo-box. So,
  * implementing classes should have this value set at construction,
  * and it should be immutable throughout the lifetime of the object.
  */
  virtual std::string GetFactoryName() const = 0;

  /**
  * \brief Returns a status string.
  *
  * At this stage, each data-source can return whatever it likes
  * that best describes its current status. This is currently not
  * something specific like an enum, but may have to change in future.
  */
  virtual std::string GetStatus() const = 0;

  /**
  * \brief Starts the capture/grabbing process.
  *
  * It is envisaged that all implementing classes implement RAII,
  * so that all implementors start capturing data immediately once
  * they are constructed. So, a constructor would typically call this.
  *
  * However, in this interface we provide StartCapturing() and
  * StopCapturing() to implement a pause functionality.
  * Derived classes however could chose to ignore this if it was
  * more appropriate.
  */
  virtual void StartCapturing() = 0;

  /**
  * \brief Stops the capture/grabbing process.
  * \see StartCapturing()
  */
  virtual void StopCapturing() = 0;

  /**
  * \brief Starts the playback process, implementors are meant to do
  * all initialisation in this method, so that after the successful return
  * of this method, the data source is fully ready to play back.
  * \param firstTimeStamp specifies the minimum time of the recording session.
  * \param lastTimeStamp specifies the maximum time of the recording session.
  */
  virtual void StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp, niftk::IGIDataType::IGITimeType lastTimeStamp) = 0;

  /**
  * \brief Request that the data source loads data corresponding to the given timestamp.
  *
  * This is expected to just load data into internal buffers. It should then be
  * up to the Update() method to update data storage.
  */
  virtual void PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp) = 0;

  /**
  * \brief Stops the playback process, so after this call, implementing data sources
  * should be grabbing live data again.
  */
  virtual void StopPlayback() = 0;

  /**
  * \brief Set the root directory of the recording session.
  */
  virtual void SetRecordingLocation(const std::string& pathName) = 0;

  /**
  * \brief Retrieves the full directory where this source is saving to.
  */
  virtual std::string GetRecordingDirectoryName() = 0;

  /**
  * \brief Start recording.
  *
  * Given a pathName (root directory) for a recording session, specified
  * by calling SetRecordingLocation(), a data source can decide to record
  * to a sub-folder. This method should initialise the sub-folder and
  * make sure the data-source is ready to record all data. The data source
  * is assumed to be grabbing live while switching into recording mode,
  * so make sure all your code is thread safe.
  */
  virtual void StartRecording() = 0;

  /**
  * \brief Stops the recording process, switching the data source back
  * to the live grabbing mode.
  */
  virtual void StopRecording() = 0;

  /**
  * \brief Set the amount of lag in milliseconds.
  *
  * For example, if a video data source, takes 200ms to output the data
  * and you immediately timestamp it then the timestamp does not correspond
  * to the actual real life timing. So, if we set an amount of lag, then
  * when we retrieve data, we can offset the timestamp to compensate for this.
  */
  virtual void SetLagInMilliseconds(const niftk::IGIDataType::IGITimeType& time) = 0;

  /**
  * \brief Sets a flag to say that this data source should/should not
  * update the data storage. This enables us to freeze the views while
  * the data source itself can continue grabbing and hence continue recording.
  */
  virtual void SetShouldUpdate(bool shouldUpdate) = 0;

  /**
  * \brief Returns the value of the update flag, \see SetShouldUpdate().
  */
  virtual bool GetShouldUpdate() const = 0;

  /**
  * \brief Requests that this data source updates data storage to reflect the given point in time.
  * \return When successful, we return a vector of information about each item that we report on.
  *
  * e.g. a tracker may have many tools, so each item in the return vector describes the
  * status of a specific tool. This means that as the number of visible items changes,
  * this array may change size.
  */
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataType::IGITimeType& time) = 0;

  /**
   * Checks whether the previously recorded data is readable, and returns the time-range for it.
   *
   * \param pathName points to the data source specific directory, e.g. "/blabla/2014-01-28-11-51-04-909/Polaris Spectra_4/"
   * \param firstTimeStampInStore earliest suitable data item. Not optional!
   * \param lastTimeStampInStore last suitable data item. Not optional!
   * \return true if there is suitable data to playback in path.
   * \throw should not throw! Return false instead.
   */
  virtual bool ProbeRecordedData(const std::string& pathName,
                                 niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                 niftk::IGIDataType::IGITimeType* lastTimeStampInStore) = 0;

protected:

  IGIDataSourceI();
  virtual ~IGIDataSourceI();

private:

  IGIDataSourceI(const IGIDataSourceI&); // deliberately not implemented
  IGIDataSourceI& operator=(const IGIDataSourceI&); // deliberately not implemented
};

} // end namespace

#endif
