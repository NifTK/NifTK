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
    Update(fixedNode->GetVtkPolyData() , movingNode->GetVtkPolyData(), transformMovingToFixed);
  }
  if ( m_Method == NIFTYSIM ) 
  {
    //Not Implemented
  }
}

void SurfaceBasedRegistration::Update(vtkPolyData* fixedPoly,
                                      vtkPolyData* movingPoly,
                                      vtkMatrix4x4 * transformMovingToFixed)
{
  if ( m_Method == VTK_ICP ) 
  {
    niftkVTKIterativeClosestPoint * icp = new  niftkVTKIterativeClosestPoint();
    icp->SetMaxLandmarks(m_MaximumNumberOfLandmarkPointsToUse);
    icp->SetMaxIterations(m_MaximumIterations);
    icp->SetSource(movingPoly);
    icp->SetTarget(fixedPoly);

    icp->Run();
    transformMovingToFixed = icp->GetTransform();
  }
}


void SurfaceBasedRegistration::Update(const mitk::PointSet::Pointer fixedNode,
                                      const mitk::Surface::Pointer movingNode,
                                      vtkMatrix4x4 * transformMovingToFixed)
{
  vtkPolyData * fixedPoly = vtkPolyData::New();
  PointSetToPolyData ( fixedNode, fixedPoly );
  Update ( fixedPoly, movingNode->GetVtkPolyData(), transformMovingToFixed );
}

void PointSetToPolyData (const  mitk::PointSet::Pointer PointsIn, vtkPolyData* PolyOut )
{}

} // end namespace

