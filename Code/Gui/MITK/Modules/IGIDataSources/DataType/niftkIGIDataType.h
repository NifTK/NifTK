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

#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace niftk
{

/**
* \class IGIDataType
* \brief Abstract base class for IGI Data, such as objects
* containing tracking data, video frames or ultrasound frames.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataType : public itk::Object
{
public:

  typedef unsigned long long IGITimeType;
  typedef unsigned long int  IGIIndexType;

  mitkClassMacroItkParent(IGIDataType, itk::Object);
  itkNewMacro(IGIDataType);

  IGITimeType GetTimeStampInNanoSeconds() const;
  void SetTimeStampInNanoSeconds(const IGITimeType& time);

  itkSetMacro(Duration, IGITimeType);
  itkGetMacro(Duration, IGITimeType);

  itkSetMacro(FrameId, IGIIndexType);
  itkGetMacro(FrameId, IGIIndexType);

  /**
  * \brief This object can contain any data, and derived classes should override this.
  */
  virtual void* GetData() const { return NULL; }

protected:

  IGIDataType(); // Purposefully hidden.
  virtual ~IGIDataType(); // Purposefully hidden.

  IGIDataType(const IGIDataType&); // Purposefully not implemented.
  IGIDataType& operator=(const IGIDataType&); // Purposefully not implemented.

private:

  IGITimeType  m_TimeStamp;
  IGITimeType  m_Duration;
  IGIIndexType m_FrameId;
};

} // end namespace

#endif
