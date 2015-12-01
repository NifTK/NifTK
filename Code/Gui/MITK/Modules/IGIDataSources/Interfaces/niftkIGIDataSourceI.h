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
* This is per item. One Source (e.g. tracker), may return data from many tools (items).
* So, each tool is considered an item. So the data source should return one
* of these IGIDataSourceInfo for each tool. Other sources such as a video
* source or framegrabber will probably only return one of these. But in
* principle it could be any number from each source.
*
* Note: Deliberately not using Qt datatypes, so that an implementing class does not have to.
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
* Note: Implementors of this interface must be thread-safe.
*
* Note: Deliberately not using Qt datatypes, so that an implementing class does not have to.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceI : public itk::Object
{

public:

  mitkClassMacroItkParent(IGIDataSourceI, itk::Object);

  virtual std::string GetName() const = 0;
  virtual std::string GetFactoryName() const = 0;
  virtual std::string GetStatus() const = 0;
  virtual void StartCapturing() = 0;
  virtual void StopCapturing() = 0;
  virtual void StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp, niftk::IGIDataType::IGITimeType lastTimeStamp) = 0;
  virtual void PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp) = 0;
  virtual void StopPlayback() = 0;
  virtual void StartRecording() = 0;
  virtual void StopRecording() = 0;
  virtual void SetLagInMilliseconds(const niftk::IGIDataType::IGITimeType& time) = 0;
  virtual void SetRecordingLocation(const std::string& pathName) = 0;
  virtual std::string GetRecordingDirectoryName() = 0;
  virtual void SetShouldUpdate(bool shouldUpdate) = 0;
  virtual bool GetShouldUpdate() const = 0;
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataType::IGITimeType& time) = 0;

  /**
   * Checks whether the previously recorded data is readable, and returns the time-range for it.
   * Default implementation returns false, i.e. is not capable of playback.
   *
   * @param path points to the data source specific directory, e.g. "/blabla/2014-01-28-11-51-04-909/Polaris Spectra_4/"
   * @param firstTimeStampInStore earliest suitable data item. Not optional!
   * @param lastTimeStampInStore last suitable data item. Not optional!
   * @return true if there is suitable data to playback in path.
   *
   * @throw should not throw! Return false instead.
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
