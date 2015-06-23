/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkICPRegService.h"
#include <mitkDataNode.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

//-----------------------------------------------------------------------------
ICPRegService::ICPRegService()
{

}


//-----------------------------------------------------------------------------
ICPRegService::~ICPRegService()
{

}


//-----------------------------------------------------------------------------
double ICPRegService::SurfaceBasedRegistration(const mitk::DataNode::Pointer& fixedDataSet,
                                               const mitk::DataNode::Pointer& movingDataSet,
                                               vtkMatrix4x4& matrix) const
{
  // not yet implemented.
  return 0.0;
}

} // end namespace
