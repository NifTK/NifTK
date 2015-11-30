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
#include <niftkIGIDataSourceI.h>

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
class NIFTKIGIDATASOURCES_EXPORT IGIDataSource : public niftk::IGIDataSourceI
{
public:

  mitkClassMacroItkParent(IGIDataSource, niftk::IGIDataSourceI);

  /**
  * \brief An IGIDataSource should be able to save an instance of its own data.
  */
  virtual void SaveItem(niftk::IGIDataType::Pointer item) = 0;

  /**
  * \brief An IGIDataSource can manage its own buffers internally.
  */
  virtual void CleanBuffer() = 0;

  /**
  * \see IGIDataSourceI::GetName()
  */
  virtual std::string GetName() const override;

  /**
  * \see IGIDataSourceI::GetStatus()
   */
  virtual std::string GetStatus() const override;

  /**
  * \see IGIDataSourceI::GetShouldUpdate()
  */
  virtual bool GetShouldUpdate() const override;

  itkGetStringMacro(MicroServiceDeviceName);

  itkSetStringMacro(Status);
  itkSetMacro(ShouldUpdate, bool);

  itkSetStringMacro(RecordingLocation);
  itkGetStringMacro(RecordingLocation);

  itkSetMacro(TimeStampTolerance, niftk::IGIDataType::IGITimeType);
  itkGetConstMacro(TimeStampTolerance, niftk::IGIDataType::IGITimeType);

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
  std::string GetPreferredSlash() const;

  /**
  * \brief Returns true if the delay between requested and actual is
  * greater than the TimeStampTolerance, and false otherwise.
  */
  bool IsLate(const niftk::IGIDataType::IGITimeType& requested,
              const niftk::IGIDataType::IGITimeType& actual
              ) const;

  /**
  * \brief Simply checks the difference in time, and converts to milliseconds.
  */
  unsigned int GetLagInMilliseconds(const niftk::IGIDataType::IGITimeType& requested,
                                    const niftk::IGIDataType::IGITimeType& actual
                                   ) const;

  igtl::TimeStamp::Pointer          m_TimeCreated; // Expensive to recreate, so available to sub-classes.

private:

  mitk::DataStorage::Pointer        m_DataStorage;
  std::set<mitk::DataNode::Pointer> m_DataNodes;
  std::string                       m_MicroServiceDeviceName;
  us::ServiceRegistration<Self>     m_MicroServiceRegistration;
  std::string                       m_RecordingLocation;
  std::string                       m_Status;
  bool                              m_ShouldUpdate;
  niftk::IGIDataType::IGITimeType   m_TimeStampTolerance; // nanoseconds.
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSource, "uk.ac.ucl.cmic.IGIDataSource")

#endif
