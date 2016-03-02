/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkExceptionMacro.h>
#include <niftkVideoTestClientCLP.h>
#include <QmitkVideoTestClient.h>
#include <QApplication>
#include <QTimer>

int main(int argc, char** argv)
{
  int returnStatus = EXIT_FAILURE;
  try
  {
    PARSE_ARGS;
    QApplication app(argc,argv);

    // RAII. Throws if unsuccessful.
    QmitkVideoTestClient client(hostname, port, seconds);
    client.show();

    // This will trigger once event loop starts.
    QTimer::singleShot(200, &client, SLOT(Run()));
    returnStatus = app.exec();
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = EXIT_FAILURE + 2;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = EXIT_FAILURE + 3;
  }
  return returnStatus;
}
