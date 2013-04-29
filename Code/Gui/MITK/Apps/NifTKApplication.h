/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <application/berryStarter.h>
#include <Poco/Util/MapConfiguration.h>

#include <QApplication>
#include <QMessageBox>
#include <QtSingleApplication>
#include <QtGlobal>
#include <QTime>

#include <mitkCommon.h>
#include <mitkException.h>

#include "mitkNifTKCoreObjectFactory.h"

/**
 * \file NifTKApplication.h
 * \brief Contains classes to startup all NifTK GUI based applications,
 * eg NiftyView, NiftyMIDAS, NiftyIGI etc.
 *
 * \class QtSafeApplication
 * \brief Class derived from QtSingleApplication to catch exceptions and implement the notify() function.
 */
class QtSafeApplication : public QtSingleApplication
{

public:

  QtSafeApplication(int& argc, char** argv) : QtSingleApplication(argc, argv)
  {}

  /**
   * Reimplement notify to catch unhandled exceptions and open an error message.
   *
   * @param receiver
   * @param event
   * @return
   */
  bool notify(QObject* receiver, QEvent* event)
  {
    QString msg;
    try
    {
      return QApplication::notify(receiver, event);
    }
    catch (mitk::Exception& e)
    {
      msg = QString("MITK Exception:\n\n")
            + QString("Description: ")
            + QString(e.GetDescription()) + QString("\n\n")
            + QString("Filename: ") + QString(e.GetFile()) + QString("\n\n")
            + QString("Line: ") + QString::number(e.GetLine());
    }
    catch (Poco::Exception& e)
    {
      msg = QString::fromStdString(e.displayText());
    }
    catch (std::exception& e)
    {
      msg = e.what();
    }
    catch (...)
    {
      msg = "Unknown exception";
    }

    MITK_ERROR << this->applicationName().toStdString() << ": An error occured\n" << msg.toStdString();

    QMessageBox msgBox;
    msgBox.setText("An error occurred. You should save all data and quit the program to prevent possible data loss.");
    msgBox.setDetailedText(msg);
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.addButton(trUtf8("Exit immediately"), QMessageBox::YesRole);
    msgBox.addButton(trUtf8("Ignore"), QMessageBox::NoRole);

    int ret = msgBox.exec();

    switch(ret)
      {
      case 0:
        MITK_ERROR << "The program was closed.";
        this->closeAllWindows();
        break;
      case 1:
        MITK_ERROR << "The top level critical error was ignored by the user. The program may be in a corrupt state and may not behave as expected!";
        break;
      }
    return false;
  }

};


/**
 * \fn ApplicationMain
 * \brief Basically the main() function that can be called for each application.
 * \param argc the number of command line arguments
 * \param argv the command line arguments
 * \param appName the application name - for example "NiftyView", "MIDAS" etc.
 * \param orgName the organisation name - should be "CMIC"
 * \param applicationPlugin the application plugin - for example "uk.ac.ucl.cmic.gui.qt.niftyview"
 */
int ApplicationMain(int argc, char** argv,
    QString appName,
    QString orgName,
    QString applicationPlugin)
{
  // Create a QApplication instance first
  QtSafeApplication myApp(argc, argv);
  myApp.setApplicationName(appName);
  myApp.setOrganizationName(orgName);

  // This function checks if an instance is already running
  // and either sends a message to it (containing the command
  // line arguments) or checks if a new instance was forced by
  // providing the BlueBerry.newInstance command line argument.
  // In the latter case, a path to a temporary directory for
  // the new application's storage directory is returned.
  QString storageDir = handleNewAppInstance(&myApp, argc, argv, "BlueBerry.newInstance");

  // These paths replace the .ini file and are tailored for installation
  // packages created with CPack. If a .ini file is presented, it will
  // overwrite the settings in MapConfiguration
  Poco::Path basePath(argv[0]);
  basePath.setFileName("");

  Poco::Path provFile(basePath);
  provFile.setFileName(appName.toStdString() + ".provisioning");

  Poco::Util::MapConfiguration* sbConfig(new Poco::Util::MapConfiguration());
  if (!storageDir.isEmpty())
  {
    sbConfig->setString(berry::Platform::ARG_STORAGE_DIR, storageDir.toStdString());
  }
  sbConfig->setString(berry::Platform::ARG_PROVISIONING, provFile.toString());
  sbConfig->setString(berry::Platform::ARG_APPLICATION, applicationPlugin.toStdString());

#ifndef Q_OS_WIN
  // Preload the org.mitk.gui.qt.ext plug-in (and hence also QmitkExt) and DICOM libs to speed
  // up a clean-cache start. This also works around bugs in older gcc and glibc implementations,
  // which have difficulties with multiple dynamic opening and closing of shared libraries with
  // many global static initializers. It also helps if dependent libraries have weird static
  // initialization methods and/or missing de-initialization code.
  sbConfig->setString(berry::Platform::ARG_PRELOAD_LIBRARY, "liborg_mitk_gui_qt_ext,libCTKDICOMCore:0.1");
#else
  // On Windows, preload the PlanarFigure.dll, otherwise there's some weird unload cycle
  // that crashes our process.
  sbConfig->setString(berry::Platform::ARG_PRELOAD_LIBRARY, "PlanarFigure");
#endif

  // VTK errors cause problem on windows, as it brings up an annoying error window.
  // We get VTK errors from the Thumbnail widget, as it switches orientation (axial, coronal, sagittal).
  // So, for now we block them completely.  This could be command line driven, or just done on Windows.
  vtkObject::GlobalWarningDisplayOff();

  // This is a NifTK specific override (could make it controlled by command line params).
  // It takes care of registering the default MITK core object factories, which includes
  // the ITK based file reader. It then hunts down the ITK based file reader, and kills
  // it, and replaces it with a more NifTK suitable one.
  RegisterNifTKCoreObjectFactory();

  int returnStatus = -1;

  try
  {
    returnStatus = berry::Starter::Run(argc, argv, sbConfig);
  }
  catch (Poco::Exception& e)
  {
    MITK_ERROR << "Caught Poco::Exception:" << e.displayText();
    returnStatus = -2;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = -3;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = -4;
  }

  return returnStatus;
}
