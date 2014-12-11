/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkPointSetCropper.h"
#include "mitkPointSetCropperEventInterface.h"

//-----------------------------------------------------------------------------
mitk::PointSetCropperEventInterface::PointSetCropperEventInterface()
{
}


//-----------------------------------------------------------------------------
mitk::PointSetCropperEventInterface::~PointSetCropperEventInterface()
{
}


//-----------------------------------------------------------------------------
void mitk::PointSetCropperEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_PointSetCropper->ExecuteOperation( op );
}
