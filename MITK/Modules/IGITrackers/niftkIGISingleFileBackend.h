/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGISingleFileBackend_h
#define niftkIGISingleFileBackend_h

#include <niftkIGITrackersExports.h>
#include "niftkIGITrackerBackend.h"
#include <niftkIGIDataSourceRingBuffer.h>
#include <iostream>

namespace niftk
{
/**
 * \class IGISingleFileBackend
 * \brief Tracker backend that saves all transforms for a single tool
 * inside the same file, using the format:
 * <verbatim>
 * timestamp q1 q2 q3 q4 t1 t2 t3 t4
 * </verbatim>
 * and each tool goes in a separate folder, just like in niftk::IGIMatrixPerFileBackend.
 */
class NIFTKIGITRACKERS_EXPORT IGISingleFileBackend : public niftk::IGITrackerBackend
{
public:

  mitkClassMacroItkParent(IGISingleFileBackend, niftk::IGITrackerBackend)
  mitkNewMacro2Param(IGISingleFileBackend, QString, mitk::DataStorage::Pointer)

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

  virtual void StopRecording();

protected:

  IGISingleFileBackend(QString name, mitk::DataStorage::Pointer dataStorage); // Purposefully hidden.
  virtual ~IGISingleFileBackend(); // Purposefully hidden.

  IGISingleFileBackend(const IGISingleFileBackend&); // Purposefully not implemented.
  IGISingleFileBackend& operator=(const IGISingleFileBackend&); // Purposefully not implemented.

private:

  typedef std::map<niftk::IGIDataSourceI::IGITimeType,
                   std::pair<mitk::Point4D, mitk::Vector3D> // quaternion, translation
                  > PlaybackTransformType;
  typedef std::map<std::string, PlaybackTransformType> PlaybackIndexType;

  // This loads all the timestamps and transformations into memory!
  PlaybackIndexType GetPlaybackIndex(const QString& directory);
  PlaybackTransformType ParseFile(const QString& fileName);

  void SaveItem(const QString& directoryName,
                const std::unique_ptr<niftk::IGIDataType>& item);

  PlaybackIndexType                                  m_PlaybackIndex;
  std::map<std::string, std::unique_ptr<ofstream> >  m_OpenFiles;

  int                                                m_FileHeaderSize; //a fixed length header
  std::string                                        m_FileHeader; //some text for the header

  // throws an exception if the file header does not match the expected file type
  void CheckFileHeader (std::ifstream& ifs);

};

} // end namespace

#endif
