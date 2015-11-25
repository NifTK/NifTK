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
#include <niftkIGIDataType.h>
#include <niftkIGIDataSourceServiceI.h>

#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <igtlTimeStamp.h>

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

  itkGetStringMacro(MicroServiceDeviceName);

  itkSetStringMacro(Status);
  itkGetStringMacro(Status);

  itkSetStringMacro(RecordingLocation);
  itkGetStringMacro(RecordingLocation);

  itkSetMacro(TimeStampTolerance, niftk::IGIDataType::IGITimeType);
  itkGetMacro(TimeStampTolerance, niftk::IGIDataType::IGITimeType);

protected:

  IGIDataSource(const std::string& microServiceDeviceName, mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
  virtual ~IGIDataSource(); // Purposefully hidden.

  IGIDataSource(const IGIDataSource&); // Purposefully not implemented.
  IGIDataSource& operator=(const IGIDataSource&); // Purposefully not implemented.

  /**
   * \brief Derived classes request a node for a given name. If the node does
   * not exist, it will be created with some default properties.
   * \param name if supplied the node will be assigned that name,
   * and if empty, the node will be given the name this->GetMicroServiceDeviceName().
   * \param addToDataStorage if true, will be added to data storage,
   * if false, the caller can determine when to do it.
   */
  mitk::DataNode::Pointer GetDataNode(const std::string& name=std::string(), const bool& addToDataStorage=true);
  mitk::DataStorage::Pointer GetDataStorage() const;

  igtl::TimeStamp::Pointer          m_TimeCreated; // Expensive to recreate, so available to sub-classes.

private:

  mitk::DataStorage::Pointer        m_DataStorage;
  std::set<mitk::DataNode::Pointer> m_DataNodes;
  std::string                       m_MicroServiceDeviceName;
  us::ServiceRegistration<Self>     m_MicroServiceRegistration;
  std::string                       m_RecordingLocation;
  std::string                       m_Status;
  niftk::IGIDataType::IGITimeType   m_TimeStampTolerance;

};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSource, "uk.ac.ucl.cmic.IGIDataSource")

#endif
