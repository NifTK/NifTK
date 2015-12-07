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

#include <niftkIGIDataSourcesExports.h>
#include <niftkIGIDataType.h>
#include <niftkSystemTimeServiceRAII.h>
#include <niftkIGIDataSourceI.h>
#include <niftkSystemTimeServiceI.h>

#include <mitkDataStorage.h>
#include <mitkServiceInterface.h>
#include <usServiceRegistration.h>

#include <QDir>
#include <QString>

namespace niftk
{

/**
* \class IGIDataSource
* \brief Abstract base class for IGI DataSources, such as objects
* that produce tracking data, video frames or ultrasound frames.
*
* Each source registers as a service when it is instantiated. You
* MUST allow for multiple instances of each service. Each service
* should have a different name, and unique Id.
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
  virtual QString GetName() const override;

  /**
  * \see IGIDataSourceI::GetFactoryName()
  */
  virtual QString GetFactoryName() const override;

  /**
  * \see IGIDataSourceI::GetStatus()
   */
  virtual QString GetStatus() const override;

  /**
  * \see IGIDataSourceI::GetShouldUpdate()
  */
  virtual bool GetShouldUpdate() const override;

  /**
  * \see IGIDataSourceI::SetShouldUpdate()
  */
  virtual void SetShouldUpdate(bool shouldUpdate) override;

  /**
  * \see IGIDataSourceI::StartPlayback()
  */
  virtual void StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                             niftk::IGIDataType::IGITimeType lastTimeStamp) override;

  /**
  * \see IGIDataSourceI::StopPlayback()
  */
  virtual void StopPlayback() override;

  /**
  * \see IGIDataSourceI::StartRecording()
  */
  virtual void StartRecording() override;

  /**
  * \see IGIDataSourceI::StopRecording()
  */
  virtual void StopRecording() override;

  QString GetRecordingLocation() const;
  virtual void SetRecordingLocation(const QString& pathName) override;

  itkGetConstMacro(IsRecording, bool);
  itkGetConstMacro(IsPlayingBack, bool);

  itkSetMacro(TimeStampTolerance, niftk::IGIDataType::IGITimeType);
  itkGetConstMacro(TimeStampTolerance, niftk::IGIDataType::IGITimeType);

  /**
   * \brief Scans the directory for individual files that match a timestamp pattern.
   * \param suffix for example ".jpg" or "-ultrasoundImage.nii".
   */
  static std::set<niftk::IGIDataType::IGITimeType> ProbeTimeStampFiles(QDir path, const QString& suffix);

  /**
  * \brief Returns the platform specific directory separator.
  */
  static QString GetPreferredSlash();

protected:

  IGIDataSource(const std::string& name,
                const std::string& factoryName,
                mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
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
  mitk::DataNode::Pointer GetDataNode(const QString& name=QString(), const bool& addToDataStorage=true);

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

  /**
  * \brief Returns the pointer to the internal data storage.
  */
  mitk::DataStorage::Pointer GetDataStorage() const;

  /**
  * \brief Queries the internal igtl::TimeStamp to get an up-to-date timestamp.
  */
  niftk::IGIDataType::IGITimeType GetTimeStampInNanoseconds();

  itkSetMacro(IsRecording, bool);
  itkSetMacro(IsPlayingBack, bool);

  itkSetStringMacro(Status);

private:

  niftk::SystemTimeServiceRAII     *m_SystemTimeService;
  mitk::DataStorage::Pointer        m_DataStorage;
  std::set<mitk::DataNode::Pointer> m_DataNodes;
  us::ServiceRegistration<Self>     m_MicroServiceRegistration;
  QString                           m_Name;
  QString                           m_FactoryName;
  QString                           m_Status;
  QString                           m_RecordingLocation;
  niftk::IGIDataType::IGITimeType   m_TimeStampTolerance; // nanoseconds.
  bool                              m_ShouldUpdate;
  bool                              m_IsRecording;
  bool                              m_IsPlayingBack;
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSource, "uk.ac.ucl.cmic.IGIDataSource")

#endif
