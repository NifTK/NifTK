/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceManager_h
#define niftkIGIDataSourceManager_h

#include "niftkIGIDataSourcesManagerExports.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataType.h>

#include <usServiceReference.h>
#include <usModuleContext.h>

#include <mitkDataStorage.h>
#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <igtlTimeStamp.h>

#include <QMap>
#include <QTimer>
#include <QList>
#include <QString>
#include <QObject>

namespace niftk
{

/**
 * \class IGIDataSourceManager
 * \brief Class to manage a list of IGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * This class should not contain Widget related stuff, so we can instantiate it directly
 * in any class, or a command line app or something without a GUI. It can still
 * derive from QObject, so that we have the benefit of signals and slots.
 *
 * Note: All errors should be thrown as mitk::Exception or sub-class thereof.
 */
class NIFTKIGIDATASOURCESMANAGER_EXPORT IGIDataSourceManager :
    public QObject,
    public itk::Object
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(IGIDataSourceManager, itk::Object);
  mitkNewMacro1Param(IGIDataSourceManager, mitk::DataStorage::Pointer);

  static const int    DEFAULT_FRAME_RATE;
  static const char*  DEFAULT_RECORDINGDESTINATION_ENVIRONMENTVARIABLE;

  bool IsUpdateTimerOn() const;
  bool IsPlayingBack() const;
  void StopUpdateTimer();
  void StartUpdateTimer();

  /**
  * \brief Returns a default path, to somewhere writable, like the desktop.
  */
  static QString GetDefaultPath();

  /**
   * \brief Gets a suitable directory name from a prefix determined by preferences, and a date-time stamp.
   */
  QString GetDirectoryName();

  /**
  * \brief Sets the base directory into which all recording sessions will be saved.
  *
  * This is normally set via a GUI preference, so remains unchanged as each
  * recording session is recorded into a new sub-directory within this directory.
  */
  void SetDirectoryPrefix(const QString& directoryPrefix);

  /**
  * \brief Sets the update rate, effectively the number of times
  * per second the internal timer ticks, and the number of times
  * the mitk::RenderingManager is asked to update.
  */
  void SetFramesPerSecond(const int& framesPerSecond);
  int GetFramesPerSecond() const;

  /**
  * \brief When creating sources, some will need configuring (e.g. port number).
  * So, given the display name of a data source (string in combo-box in GUI),
  * will return true if the manager can create a GUI for you to configure the service.
  */
  bool NeedsStartupGui(QString name);

  /**
  * \brief Writes the descriptor file for a recording session.
  *
  * This descriptor is then used to reconstruct the right number
  * of data sources when you playback.
  */
  void WriteDescriptorFile(QString absolutePath);

  /**
  * \brief Retrieves the name of all the available data source factory names.
  *
  * The returned list is the display name, as shown in the GUI,
  * e.g. "OpenCV Frame Grabber", and these strings are
  * created in each data sources factory class.
  */
  QList<QString> GetAllFactoryNames() const;

  /**
  * \brief Adds a source, using the display name of a factory,
  * and configures it with the provided properties.
  */
  void AddSource(QString name, QList<QMap<QString, QVariant> >& properties);

  /**
  * \brief Removes a source at a given rowIndex.
  */
  void RemoveSource(int rowIndex);

  /**
  * \brief Removes all sources.
  */
  void RemoveAllSources();

  /**
  * \brief Starts a new recording session, writing to the folder given by the absolutePath.
  */
  void StartRecording(QString absolutePath);

  /**
  * \brief Stops the recording process.
  */
  void StopRecording();

  /**
  * \brief Freezes the data sources (i.e. does not do update).
  *
  * Does not affect the saving of data. The data source can
  * continue to grab data, and save it, as it feels like.
  */
  void FreezeAllDataSources(bool isFrozen);

  /**
  * \brief Freezes individual data sources (i.e. does not do update).
  *
  * Does not affect the saving of data. The data source can
  * continue to grab data, and save it, as it feels like.
  */
  void FreezeDataSource(unsigned int i, bool isFrozen);

  /**
  * \brief Sets the manager ready for playback.
  *
  * \param directoryPrefix path to the root folder of the recording session
  * \param descriptorPath path to a descriptor to parse.
  * \param startTime returns the minimum of start times of all available data sources.
  * \param endTime returns the maximum of end times of all available data sources.
  *
  * Here, the user must have created the right data sources. The reason is
  * that each data-source might need configuring each time you set up
  * the system.  e.g. An Aurora tracker might be connected to a specific COM port.
  * These configurations might be different between record and playback.
  */
  void StartPlayback(const QString& directoryPrefix,
                     const QString& descriptorPath,
                     IGIDataType::IGITimeType& startTime,
                     IGIDataType::IGITimeType& endTime);

  /**
  * \brief Stops all sources playing back.
  */
  void StopPlayback();

  /**
  * \brief Sets the current time of the manager to time,
  * and the next available update pulse will trigger a refresh.
  */
  void SetPlaybackTime(const IGIDataType::IGITimeType& time);

signals:

  /**
  * \brief Emmitted when this manager has asked each data source to update, and they have all updated.
  */
  void UpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >);

  /**
  * \brief Emmitted when this manager has called for rendering to be updated, and that call has completed.
  *
  * (This doesn't mean that the rendering has actually happened. That depends on the mitk::RenderingManager).
  */
  void UpdateFinishedRendering();

protected:

  IGIDataSourceManager(mitk::DataStorage::Pointer dataStorage);
  virtual ~IGIDataSourceManager();

  IGIDataSourceManager(const IGIDataSourceManager&); // Purposefully not implemented.
  IGIDataSourceManager& operator=(const IGIDataSourceManager&); // Purposefully not implemented.

private slots:

  /**
  * \brief Updates the whole rendered scene, based on the available sources.
  */
  void OnUpdateGui();

private:

  /**
  * \brief Inspects the module registry to retrieve the list of all data source factories.
  */
  void RetrieveAllDataSourceFactories();

  /**
   * Tries to parse the data source descriptor for directory-to-classname mappings.
   * @param filepath full qualified path to descriptor.cfg, e.g. "/home/jo/projectwork/2014-01-28-11-51-04-909/descriptor.cfg"
   * @returns a map with key = directory, value = classname
   * @throws std::exception if something goes wrong.
   * @warning This method does not check whether any class name is valid, i.e. whether that class has been compiled in!
   */
  QMap<QString, QString> ParseDataSourceDescriptor(const QString& filepath);

  /**
  * \brief Used to switch the manager between playback and live mode.
  */
  void SetIsPlayingBack(bool isPlayingBack);

  mitk::DataStorage::Pointer                                       m_DataStorage; // populated in constructor, so always valid.
  us::ModuleContext*                                               m_ModuleContext;
  std::vector<us::ServiceReference<IGIDataSourceFactoryServiceI> > m_Refs;
  QList<niftk::IGIDataSourceI::Pointer>                            m_Sources;
  QMap<QString, niftk::IGIDataSourceFactoryServiceI*>              m_NameToFactoriesMap;
  QTimer                                                          *m_GuiUpdateTimer;
  int                                                              m_FrameRate;
  QString                                                          m_DirectoryPrefix;
  QString                                                          m_PlaybackPrefix;
  igtl::TimeStamp::Pointer                                         m_TimeStampGenerator;
  bool                                                             m_IsPlayingBack;
  niftk::IGIDataType::IGITimeType                                  m_CurrentTime;

}; // end class;

} // end namespace

#endif
