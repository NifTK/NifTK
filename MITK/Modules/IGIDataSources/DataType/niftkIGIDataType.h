/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataType_h
#define niftkIGIDataType_h

#include "niftkIGIDataSourcesExports.h"
#include <niftkIGIDataSourceI.h>

namespace niftk
{

/**
* \class IGIDataType
* \brief Abstract base class for IGI Data, such as objects
* containing tracking data, video frames or ultrasound frames.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*
* Note: This class, and hence derived classes, must implement a full set of
* copy and move constructors and assignment operators.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataType
{
public:

  IGIDataType();
  virtual ~IGIDataType();
  IGIDataType(const IGIDataType&);             // Copy constructor
  IGIDataType& operator=(const IGIDataType&);  // Copy assignment
  IGIDataType(IGIDataType&&);                  // Move constructor
  IGIDataType& operator=(IGIDataType&&);       // Move assignment

  niftk::IGIDataSourceI::IGITimeType GetTimeStampInNanoSeconds() const { return m_TimeStamp; }
  void SetTimeStampInNanoSeconds(const niftk::IGIDataSourceI::IGITimeType& time) { m_TimeStamp = time; }

  niftk::IGIDataSourceI::IGITimeType GetDuration() const { return m_Duration; }
  void SetDuration(const niftk::IGIDataSourceI::IGITimeType& duration) { m_Duration = duration; }

  niftk::IGIDataSourceI::IGIIndexType GetFrameId() const { return m_FrameId; }
  void SetFrameId(const niftk::IGIDataSourceI::IGIIndexType& id) { m_FrameId = id; }

  bool GetIsSaved() const { return m_IsSaved; }
  void SetIsSaved(bool isSaved) { m_IsSaved = isSaved; }

  bool GetShouldBeSaved() const { return m_ShouldBeSaved; }
  void SetShouldBeSaved(bool shouldBe) { m_ShouldBeSaved = shouldBe; }

  std::string GetFileName() const { return m_FileName; }
  void SetFileName(const std::string& fileName) { m_FileName = fileName; }

  virtual void Clone(const IGIDataType&);

protected:

private:

  niftk::IGIDataSourceI::IGITimeType  m_TimeStamp;
  niftk::IGIDataSourceI::IGITimeType  m_Duration;
  niftk::IGIDataSourceI::IGIIndexType m_FrameId;
  bool                                m_IsSaved;
  bool                                m_ShouldBeSaved;
  std::string                         m_FileName;
};

} // end namespace

#endif
