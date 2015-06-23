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
  m_Registerer = niftk::ICPBasedRegistration::New();
}


//-----------------------------------------------------------------------------
ICPRegService::~ICPRegService()
{

}


//-----------------------------------------------------------------------------
double ICPRegService::Register(const mitk::DataNode::Pointer fixedDataSet,
                               const mitk::DataNode::Pointer movingDataSet,
                               vtkMatrix4x4& matrix) const
{
  return m_Registerer->Update(fixedDataSet, movingDataSet, matrix);
}

} // end namespace
