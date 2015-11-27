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
  * \brief Retrieves the name of all the available data sources
  * (retrieved during the constructor).
  */
  QList<QString> GetAllSources() const;

  /**
  * \brief Sets the base directory into which all recording sessions will be saved.
  */
  void SetDirectoryPrefix(const QString& directoryPrefix);

  /**
  * \brief Sets the update rate, effectively the number of frames per second of rendering.
  */
  void SetFramesPerSecond(const int& framesPerSecond);

  /**
  * \brief Writes the descriptor file for a recording session.
  */
  void WriteDescriptorFile(QString absolutePath);

  /**
  * \brief When creating sources, some will need configuring (e.g. port number).
  */
  bool NeedsStartupGui(QString name);

  /**
  * \brief Adds a source, using the display name of a factory, and configures it with these properties.
  */
  void AddSource(QString name, QList<QMap<QString, QVariant> >& properties);

  /**
  * \brief Removes a source at a given rowIndex.
  */
  void RemoveSource(int rowIndex);

  /**
  * \brief Starts a new recording session, writing to the folder given by the absolutePath.
  */
  void StartRecording(QString absolutePath);

  /**
  * \brief Stops the recording process.
  */
  void StopRecording();

signals:

  /**
  * \brief Emmitted when the manager has asked each data source to update, and they have all updated.
  */
  void UpdateGuiFinishedDataSources();

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
  * \brief Inspects the module registry to populate the list of all data source factories.
  */
  void RetrieveAllDataSources();

  mitk::DataStorage::Pointer                                       m_DataStorage; // populated in constructor, so always valid.
  us::ModuleContext*                                               m_ModuleContext;
  std::vector<us::ServiceReference<IGIDataSourceFactoryServiceI> > m_Refs;
  QList<niftk::IGIDataSourceI::Pointer>                            m_Sources;
  QMap<QString, niftk::IGIDataSourceFactoryServiceI*>              m_NameToFactoriesMap;
  QTimer                                                          *m_GuiUpdateTimer;
  int                                                              m_FrameRate;
  QString                                                          m_DirectoryPrefix;
  igtl::TimeStamp::Pointer                                         m_TimeStampGenerator;

}; // end class;

} // end namespace

#endif
