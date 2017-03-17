/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBKMedicalDataSourceWorker.h"
#include <mitkLogMacros.h>

namespace niftk
{

//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::BKMedicalDataSourceWorker()
{
  MITK_INFO << "BKMedicalDataSourceWorker constructed";
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceWorker::~BKMedicalDataSourceWorker()
{
  MITK_INFO << "BKMedicalDataSourceWorker destructed";
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ConnectToHost(QString address, int port)
{
  MITK_INFO << "BKMedicalDataSourceWorker connecting:" << address.toStdString() << ":" << port;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceWorker::ReceiveImages()
{
  MITK_INFO << "BKMedicalDataSourceWorker receiving:";
}

} // end namespace
