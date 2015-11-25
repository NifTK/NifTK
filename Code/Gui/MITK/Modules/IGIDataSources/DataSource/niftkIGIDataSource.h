/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSource_h
#define niftkIGIDataSource_h

#include "niftkIGIDataSourcesExports.h"
#include "niftkIGIDataType.h"
#include <niftkIGIDataSourceServiceI.h>

#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

#include <mitkDataStorage.h>
#include <mitkServiceInterface.h>
#include <usServiceRegistration.h>

namespace niftk
{

/**
* \class IGIDataSource
* \brief Abstract base class for IGI DataSources, such as objects
* that produce tracking data, video frames or ultrasound frames.
*
* Each source registers as a service when it is instantiated. You
* MUST allow for multiple instances of each service. Each service
* should have a different name, and random Id.
*
* Uses RAII pattern to register/de-register as MITK Micro-Service.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSource : public itk::Object, public niftk::IGIDataSourceServiceI
{
public:

  mitkClassMacroItkParent(IGIDataSource, itk::Object);

  /**
  * \brief An IGIDataSource should be able to save an instance of its own data.
  */
  virtual void SaveItem(niftk::IGIDataType::Pointer item) = 0;

  /**
  * \brief An IGIDataSource can manage its own buffers internally.
  */
  virtual void ClearBuffer() = 0;

  /**
  * \brief Gets the device name, set during construction, and never changed.
  */
  std::string GetDeviceName();

  /**
  * \see niftk::IGIDataSourceServiceI::SetRecordingLocation()
  */
  virtual void SetRecordingLocation(const std::string& pathName) override;

protected:

  IGIDataSource(const std::string& microServiceDeviceName, mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
  virtual ~IGIDataSource(); // Purposefully hidden.

  IGIDataSource(const IGIDataSource&); // Purposefully not implemented.
  IGIDataSource& operator=(const IGIDataSource&); // Purposefully not implemented.

private:

  mitk::DataStorage::Pointer    m_DataStorage;
  std::string                   m_MicroServiceDeviceName;
  us::ServiceRegistration<Self> m_MicroServiceRegistration;
  std::string                   m_RecordingLocation;
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSource, "uk.ac.ucl.cmic.IGIDataSource")

#endif
