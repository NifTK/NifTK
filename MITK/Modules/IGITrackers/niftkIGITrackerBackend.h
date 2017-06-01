/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGITrackerBackend_h
#define niftkIGITrackerBackend_h

#include <niftkIGITrackersExports.h>
#include <niftkIGIDataSourceI.h>
#include <niftkIGITrackerDataType.h>
#include <niftkIGIDataSourceRingBuffer.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkDataStorage.h>
#include <QString>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
 * \class IGITrackerBackend
 * \brief Abstract interface for tracker back-ends, basically,
 * something to offload the bits that are not grabbing data
 * (e.g. saving, loading from file store).
 */
class NIFTKIGITRACKERS_EXPORT IGITrackerBackend : public itk::Object
{
public:

  mitkClassMacroItkParent(IGITrackerBackend, itk::Object)
  itkGetMacro(Lag, int);
  itkSetMacro(Lag, int);
  itkGetMacro(ExpectedFramesPerSecond, unsigned int);
  itkSetMacro(ExpectedFramesPerSecond, unsigned int);

  /**
  * \brief Add's one frame of data into the buffers, saving to directoryName if isRecording is true.
  */
  virtual void AddData(const QString& directoryName,
                       const bool& isRecording,
                       const niftk::IGIDataSourceI::IGITimeType& duration,
                       const niftk::IGIDataSourceI::IGITimeType& timeStamp,
                       const std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >&) = 0;

  /**
  * \see  IGIDataSourceI::StartPlayback()
  */
  virtual void StartPlayback(const QString& directoryName,
                             const niftk::IGIDataSourceI::IGITimeType& firstTimeStamp,
                             const niftk::IGIDataSourceI::IGITimeType& lastTimeStamp) = 0;

  /**
  * \see IGIDataSourceI::PlaybackData()
  */
  virtual void PlaybackData(const QString& directoryName,
                            const niftk::IGIDataSourceI::IGITimeType& duration,
                            const niftk::IGIDataSourceI::IGITimeType& requestedTimeStamp) = 0;

  /**
  * \see IGIDataSourceI::StopPlayback()
  */
  virtual void StopPlayback() = 0;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  virtual bool ProbeRecordedData(const QString& directoryName,
                                 niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                 niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore) = 0;

  /**
  * \see IGIDataSourceI::StopRecording()
  */
  virtual void StopRecording() {} // deliberately empty.

  /**
  * \brief IGIDataSourceI::SetProperties()
  */
  virtual void SetProperties(const IGIDataSourceProperties& properties);

  /**
  * \brief IGIDataSourceI::GetProperties()
  */
  virtual IGIDataSourceProperties GetProperties() const;

  /**
  * \brief Loads one frame of data into mitk::DataStorage corresponding to the given time.
  */
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataSourceI::IGITimeType& time);

protected:

  IGITrackerBackend(QString name,
                    mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
  virtual ~IGITrackerBackend(); // Purposefully hidden.

  IGITrackerBackend(const IGITrackerBackend&); // Purposefully not implemented.
  IGITrackerBackend& operator=(const IGITrackerBackend&); // Purposefully not implemented.

  void WriteToDataStorage(const std::string& name,
                          const niftk::IGITrackerDataType& transform);

  QString                            m_Name;
  mitk::DataStorage::Pointer         m_DataStorage;
  int                                m_FrameId;
  int                                m_Lag;
  unsigned int                       m_ExpectedFramesPerSecond;
  std::set<mitk::DataNode::Pointer>  m_DataNodes;
  vtkSmartPointer<vtkMatrix4x4>      m_CachedTransform;
  niftk::IGITrackerDataType          m_CachedDataType;
  std::map<std::string,
           std::unique_ptr<
             niftk::IGIDataSourceRingBuffer>
          >                          m_Buffers;
};

} // end namespace

#endif
