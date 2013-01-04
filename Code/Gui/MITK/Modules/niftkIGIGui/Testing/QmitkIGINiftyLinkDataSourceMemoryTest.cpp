/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <cstdlib>
#include <mitkTestingMacros.h>
#include "QmitkIGITrackerTool.h"
#include "QmitkIGINiftyLinkDataType.h"
#include "OIGTLTrackingDataMessage.h"

/**
 * \brief This test is simply so we can run through valgrind and check
 * for no leaks.
 */
int QmitkIGINiftyLinkDataSourceMemoryTest(int /*argc*/, char* /*argv*/[])
{

  // Message comes in. Here we create a local pointer, not a smart pointer.
  OIGTLTrackingDataMessage* msg = new OIGTLTrackingDataMessage();

  // It gets wrapped in a data type. Here we create a local pointer, not a smart pointer.
  QmitkIGINiftyLinkDataType::Pointer dataType = QmitkIGINiftyLinkDataType::New();
  dataType->SetMessage(msg);

  // It gets added to the buffer of the data storage.
  QmitkIGITrackerTool::Pointer tool = QmitkIGITrackerTool::New();
  tool->AddData(dataType);

  // When we call delete, the tool should correctly tidy up all memory.
  // When this program itself exits, the smart pointer to tool should delete the tool.
  tool->ClearBuffer();

  return EXIT_SUCCESS;
}
