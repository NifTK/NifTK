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
*/
struct NIFTKIGIDATASOURCES_EXPORT IGIDataItemInfo
{
  IGIDataItemInfo()
  {
    m_Name = "Unknown";
    m_Status = "Unknown";
    m_IsLate = false;
    m_LagInMilliseconds = 0;
    m_FramesPerSecond = 0;
    m_Description = "Unknown";
  }

  std::string  m_Name;
  std::string  m_Status;
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
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceI : public itk::Object
{

public:

  mitkClassMacroItkParent(IGIDataSourceI, itk::Object);

  virtual void StartCapturing() = 0;
  virtual void StopCapturing() = 0;
  virtual void StartRecording() = 0;
  virtual void StopRecording() = 0;
  virtual void SetLagInMilliseconds(const niftk::IGIDataType::IGITimeType& time) = 0;
  virtual void SetRecordingLocation(const std::string& pathName) = 0;
  virtual std::string GetName() const = 0;
  virtual std::string GetStatus() const = 0;
  virtual std::string GetSaveDirectoryName() = 0;
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataType::IGITimeType& time) = 0;

protected:

  IGIDataSourceI();
  virtual ~IGIDataSourceI();

private:

  IGIDataSourceI(const IGIDataSourceI&); // deliberately not implemented
  IGIDataSourceI& operator=(const IGIDataSourceI&); // deliberately not implemented
};

} // end namespace

#endif
