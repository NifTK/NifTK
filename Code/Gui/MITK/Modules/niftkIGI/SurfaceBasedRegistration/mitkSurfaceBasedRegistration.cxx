/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkSurfaceBasedRegistration.h"
#include "niftkVTKIterativeClosestPoint.h"

namespace mitk
{

//-----------------------------------------------------------------------------
SurfaceBasedRegistration::SurfaceBasedRegistration()
{
}


//-----------------------------------------------------------------------------
SurfaceBasedRegistration::~SurfaceBasedRegistration()
{
}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::Update(const mitk::Surface::Pointer fixedNode,
                                      const mitk::Surface::Pointer movingNode,
                                      vtkMatrix4x4* transformMovingToFixed)
{

  if ( m_Method == VTK_ICP ) 
  {
    niftkVTKIterativeClosestPoint * icp = new  niftkVTKIterativeClosestPoint();
    icp->SetMaxLandmarks(m_MaximumNumberOfLandmarkPointsToUse);
    icp->SetMaxIterations(m_MaximumIterations);
    icp->SetSource(movingNode->GetVtkPolyData());
    icp->SetTarget(fixedNode->GetVtkPolyData());

    icp->Run();
    transformMovingToFixed = icp->GetTransform();
  }
  if ( m_Method == NIFTYSIM ) 
  {
    //Not Implemented
  }

}

} // end namespace

