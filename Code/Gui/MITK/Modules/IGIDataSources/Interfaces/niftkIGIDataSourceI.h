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
  virtual std::string GetSaveDirectoryName() = 0;
  virtual void Update(const niftk::IGIDataType::IGITimeType& time) = 0;

protected:

  IGIDataSourceI();
  virtual ~IGIDataSourceI();

private:

  IGIDataSourceI(const IGIDataSourceI&); // deliberately not implemented
  IGIDataSourceI& operator=(const IGIDataSourceI&); // deliberately not implemented
};

} // end namespace

#endif
