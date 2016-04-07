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
void ICPRegService::Configure(const us::ServiceProperties& properties)
{
  if (properties.size() == 0)
  {
    return;
  }

  if (properties.find("MaxLandmarks") != properties.end())
  {
    int maxLandmarks = us::any_cast<int>((*(properties.find("MaxLandmarks"))).second);
    m_Registerer->SetMaximumNumberOfLandmarkPointsToUse(maxLandmarks);
    MITK_INFO << "Configured ICPRegService[MaxLandmarks]=" << maxLandmarks;
  }

  if (properties.find("MaxIterations") != properties.end())
  {
    int maxIterations = us::any_cast<int>((*(properties.find("MaxIterations"))).second);
    m_Registerer->SetMaximumIterations(maxIterations);
    MITK_INFO << "Configured ICPRegService[MaxIterations]=" << maxIterations;
  }

  if (properties.find("TLSIterations") != properties.end())
  {
    int tlsIterations = us::any_cast<unsigned int>((*(properties.find("TLSIterations"))).second);
    m_Registerer->SetTLSIterations(tlsIterations);
    MITK_INFO << "Configured ICPRegService[TLSIterations]=" << tlsIterations;
  }

  if (properties.find("TLSPercentage") != properties.end())
  {
    int tlsPercentage = us::any_cast<unsigned int>((*(properties.find("TLSPercentage"))).second);
    m_Registerer->SetTLSPercentage(tlsPercentage);
    MITK_INFO << "Configured ICPRegService[TLSPercentage]=" << tlsPercentage;
  }
}


//-----------------------------------------------------------------------------
double ICPRegService::Register(const mitk::DataNode::Pointer fixedDataSet,
                               const mitk::DataNode::Pointer movingDataSet,
                               vtkMatrix4x4& matrix) const
{
  return m_Registerer->Update(fixedDataSet, movingDataSet, matrix);
}

} // end namespace
