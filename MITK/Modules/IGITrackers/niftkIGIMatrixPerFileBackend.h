/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIMatrixPerFileBackend_h
#define niftkIGIMatrixPerFileBackend_h

#include <niftkIGITrackersExports.h>
#include "niftkIGITrackerBackend.h"
#include <QSet>

namespace niftk
{
/**
 * \class IGIMatrixPerFileBackend
 * \brief Tracker backend that saves each transformation as a 4x4 matrix, each in a separate file.
 */
class NIFTKIGITRACKERS_EXPORT IGIMatrixPerFileBackend : public niftk::IGITrackerBackend
{
public:

  mitkClassMacroItkParent(IGIMatrixPerFileBackend, niftk::IGITrackerBackend)
  mitkNewMacro2Param(IGIMatrixPerFileBackend, QString, mitk::DataStorage::Pointer)

  /**
  * \brief Add's one frame of data into the buffers, saving to directory if needed.
  */
  void AddData(const QString& directoryName,
               const bool& isRecording,
               const niftk::IGIDataSourceI::IGITimeType& duration,
               const niftk::IGIDataSourceI::IGITimeType& timeStamp,
               const std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >& data);

  /**
  * \see  IGIDataSourceI::StartPlayback()
  * \brief Pre-loads all timestamps and filenames.
  */
  virtual void StartPlayback(const QString& directoryName,
                             const niftk::IGIDataSourceI::IGITimeType& firstTimeStamp,
                             const niftk::IGIDataSourceI::IGITimeType& lastTimeStamp) override;

  /**
  * \see IGIDataSourceI::PlaybackData()
  * \brief Basically, loads the closest (in time) data into the buffers.
  */
  virtual void PlaybackData(const niftk::IGIDataSourceI::IGITimeType& duration,
                            const niftk::IGIDataSourceI::IGITimeType& requestedTimeStamp) override;

  /**
  * \see IGIDataSourceI::StopPlayback()
  * \brief Basically, clears buffers.
  */
  virtual void StopPlayback() override;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  * \brief Scans directoryName to determine the min and max timestamp.
  */
  virtual bool ProbeRecordedData(const QString& directoryName,
                                 niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                 niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore) override;

protected:

  IGIMatrixPerFileBackend(QString name, mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
  virtual ~IGIMatrixPerFileBackend(); // Purposefully hidden.

  IGIMatrixPerFileBackend(const IGIMatrixPerFileBackend&); // Purposefully not implemented.
  IGIMatrixPerFileBackend& operator=(const IGIMatrixPerFileBackend&); // Purposefully not implemented.

private:

  // This loads all the timestamps and filenames into memory!
  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> > GetPlaybackIndex(const QString& directory);

  void SaveItem(const QString& directoryName,
                const std::unique_ptr<niftk::IGIDataType>& item);

  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> > m_PlaybackIndex;

private:

  vtkSmartPointer<vtkMatrix4x4>      m_CachedTransformForSaving;
};

} // end namespace

#endif
