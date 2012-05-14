/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8038 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <application/berryStarter.h>
#include <Poco/Util/MapConfiguration.h>

#include <QApplication>
#include <QMessageBox>
#include <QtSingleApplication>

#include "mitkNifTKCoreObjectFactory.h"

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

    QString text("An error occurred. You should save all data and quit the program to "
                 "prevent possible data loss.\nSee the error log for details.\n\n");
    text += msg;

    QMessageBox::critical(0, "Error", text);
    return false;
  }

};

int main(int argc, char** argv)
{
  // Create a QApplication instance first
  QtSafeApplication myApp(argc, argv);
  myApp.setApplicationName("NiftyView");
  myApp.setOrganizationName("CMIC");

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
  provFile.setFileName("NiftyView.provisioning");

  Poco::Util::MapConfiguration* sbConfig(new Poco::Util::MapConfiguration());
  if (!storageDir.isEmpty())
  {
    sbConfig->setString(berry::Platform::ARG_STORAGE_DIR, storageDir.toStdString());
  }
  sbConfig->setString(berry::Platform::ARG_PROVISIONING, provFile.toString());
  sbConfig->setString(berry::Platform::ARG_APPLICATION, "uk.ac.ucl.cmic.gui.qt.niftyview");

  // Preload the org.mitk.gui.qt.ext plug-in (and hence also QmitkExt) to speed
  // up a clean-cache start. This also works around bugs in older gcc and glibc implementations,
  // which have difficulties with multiple dynamic opening and closing of shared libraries with
  // many global static initializers. It also helps if dependent libraries have weird static
  // initialization methods and/or missing de-initialization code.
  sbConfig->setString(berry::Platform::ARG_PRELOAD_LIBRARY, "liborg_mitk_gui_qt_ext");

  // VTK errors cause problem on windows, as it brings up an annoying error window.
  // We get VTK errors from the Thumbnail widget, as it switches orientation (axial, coronal, sagittal).
  // So, for now we block them completely.  This could be command line driven, or just done on Windows.
  vtkObject::GlobalWarningDisplayOff();

  // This is a NifTK specific override (could make it controlled by command line params).
  // It takes care of registering the default MITK core object factories, which includes
  // the ITK based file reader. It then hunts down the ITK based file reader, and kills
  // it, and replaces it with a more NifTK suitable one.
  RegisterNifTKCoreObjectFactory();

  return berry::Starter::Run(argc, argv, sbConfig);
}
