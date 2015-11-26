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
#include "ui_niftkIGIDataSourceManager.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <niftkIGIDataType.h>
#include <usServiceReference.h>
#include <usModuleContext.h>
#include <mitkDataStorage.h>
#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

#include <QWidget>
#include <QMap>

namespace niftk
{

/**
 * \class IGIDataSourceManager
 * \brief Class to manage a list of IGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * This widget acts like a widget factory, setting up sources, instantiating
 * the appropriate GUI, and loading it into the grid layout owned by this widget.
 */
class NIFTKIGIDATASOURCESMANAGER_EXPORT IGIDataSourceManager : public QWidget, public Ui_IGIDataSourceManager, public itk::Object
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(IGIDataSourceManager, itk::Object);
  mitkNewMacro1Param(IGIDataSourceManager, mitk::DataStorage::Pointer);

  static QString      GetDefaultPath();
  static const int    DEFAULT_FRAME_RATE;
  static const char*  DEFAULT_RECORDINGDESTINATION_ENVIRONMENTVARIABLE;

  /**
   * \brief Creates the base class widgets, and connects signals and slots.
   */
  void setupUi(QWidget* parent);

  /**
  * \brief Called from GUI to start the recording process.
  */
  void StartRecording();

  /**
  * \brief Called from the GUI to stop the recording process.
  */
  void StopRecording();

  /**
  * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
  */
  void SetDirectoryPrefix(const QString& directoryPrefix);

  /**
  * \brief Called from the GUI when the surgical guidance plugin preferences are modified.
  */
  void SetFramesPerSecond(const int& framesPerSecond);

signals:

  /**
  * \brief Emmitted when the manager has asked each data source to update, and they have all updated.
  */
  void UpdateGuiFinishedDataSources(niftk::IGIDataType::IGITimeType timeStamp);

  /**
  * \brief Emmitted when recording has successfully started.
  */
  void RecordingStarted(QString basedirectory);

protected:

  IGIDataSourceManager(mitk::DataStorage::Pointer dataStorage);
  virtual ~IGIDataSourceManager();

  IGIDataSourceManager(const IGIDataSourceManager&); // Purposefully not implemented.
  IGIDataSourceManager& operator=(const IGIDataSourceManager&); // Purposefully not implemented.

private slots:

  /**
  * \brief Adds a data source to the table.
  */
  void OnAddSource();

private:

  mitk::DataStorage::Pointer                                       m_DataStorage; // populated in constructor, so always valid.
  us::ModuleContext*                                               m_ModuleContext;
  std::vector<us::ServiceReference<IGIDataSourceFactoryServiceI> > m_Refs;
  QMap<QString, niftk::IGIDataSourceFactoryServiceI*>              m_NameToFactoriesMap;
  bool                                                             m_SetupGuiHasBeenCalled;

}; // end class;

} // end namespace

#endif
