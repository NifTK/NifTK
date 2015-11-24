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
* must allow for multiple instances of each service.
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
  * \brief A DataSource should be able to save its own data.
  */
  virtual void SaveItem(niftk::IGIDataType::Pointer item) = 0;

protected:

  IGIDataSource(const std::string& microServiceDeviceName); // Purposefully hidden.
  virtual ~IGIDataSource(); // Purposefully hidden.

  IGIDataSource(const IGIDataSource&); // Purposefully not implemented.
  IGIDataSource& operator=(const IGIDataSource&); // Purposefully not implemented.

private:
  std::string                   m_MicroServiceDeviceName;
  us::ServiceRegistration<Self> m_MicroServiceRegistration;
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSource, "uk.ac.ucl.cmic.IGIDataSource")

#endif
