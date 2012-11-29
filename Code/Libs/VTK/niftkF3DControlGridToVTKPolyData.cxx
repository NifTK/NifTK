/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "niftkF3DControlGridToVTKPolyData.h"

#include <vtkVersion.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkVertex.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridGeometryFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkSphereSource.h>

#include "_reg_aladin.h"
#include "_reg_tools.h"
#include "_reg_f3d2.h"


#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif


namespace niftk
{

 
// ---------------------------------------------------------------------------
// Create a VTK polydata object to visualise the control points
// ---------------------------------------------------------------------------

vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataPoints( nifti_image *controlPointGrid )
{
  int x, y, z, index;
  float xInit, yInit, zInit;

  PrecisionTYPE *controlPointPtrX, *controlPointPtrY, *controlPointPtrZ;

  mat44 *splineMatrix;

  int nControlPoints = controlPointGrid->nx*controlPointGrid->ny*controlPointGrid->nz;

  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkIdType id;

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

  vtkSmartPointer<vtkPolyData> vtkControlPoints = vtkSmartPointer<vtkPolyData>::New();
  
  controlPointPtrX = static_cast< PrecisionTYPE * >( controlPointGrid->data );
  controlPointPtrY = &controlPointPtrX[ nControlPoints ];
  controlPointPtrZ = &controlPointPtrY[ nControlPoints ];

  if ( controlPointGrid->sform_code > 0 ) 
    splineMatrix = &( controlPointGrid->sto_xyz );
  else 
    splineMatrix = &( controlPointGrid->qto_xyz );

  for (z=0; z<controlPointGrid->nz; z++)
  {
    index = z*controlPointGrid->nx*controlPointGrid->ny;

    for (y=0; y<controlPointGrid->ny; y++)
    {
      for (x=0; x<controlPointGrid->nx; x++)
      {
	  
	// The initial control point position
	xInit =
	  splineMatrix->m[0][0]*static_cast<float>(x) +
	  splineMatrix->m[0][1]*static_cast<float>(y) +
	  splineMatrix->m[0][2]*static_cast<float>(z) +
	  splineMatrix->m[0][3];
	yInit =
	  splineMatrix->m[1][0]*static_cast<float>(x) +
	  splineMatrix->m[1][1]*static_cast<float>(y) +
	  splineMatrix->m[1][2]*static_cast<float>(z) +
	  splineMatrix->m[1][3];
	zInit =
	  splineMatrix->m[2][0]*static_cast<float>(x) +
	  splineMatrix->m[2][1]*static_cast<float>(y) +
	  splineMatrix->m[2][2]*static_cast<float>(z) +
	  splineMatrix->m[2][3];

	// The final control point position:
	//    controlPointPtrX[index], controlPointPtrY[index], controlPointPtrZ[index];

#if 1
	id = points->InsertNextPoint( -controlPointPtrX[index], 
				      -controlPointPtrY[index],
				      controlPointPtrZ[index] );


#else
	id = points->InsertNextPoint( -xInit, -yInit, zInit );
#endif 
	cells->InsertNextCell( 1 );
	cells->InsertCellPoint( id );

	index++;
      }
    }
  }

  vtkControlPoints->SetPoints( points );
  vtkControlPoints->SetVerts( cells );
  
  return vtkControlPoints;
}

 
// ---------------------------------------------------------------------------
// Create a VTK polydata object to visualise the control points using spheres
// ---------------------------------------------------------------------------

  vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataSpheres( nifti_image *controlPointGrid,
								   float radius )
{
  int x, y, z, index;
  float xInit, yInit, zInit;

  PrecisionTYPE *controlPointPtrX, *controlPointPtrY, *controlPointPtrZ;

  mat44 *splineMatrix;

  int nControlPoints = controlPointGrid->nx*controlPointGrid->ny*controlPointGrid->nz;

  vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

  vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();

#if 0
  sphere->SetThetaResolution(10);
  sphere->SetPhiResolution(5);
#endif

  sphere->SetRadius( radius );


  controlPointPtrX = static_cast< PrecisionTYPE * >( controlPointGrid->data );
  controlPointPtrY = &controlPointPtrX[ nControlPoints ];
  controlPointPtrZ = &controlPointPtrY[ nControlPoints ];

  if ( controlPointGrid->sform_code > 0 ) 
    splineMatrix = &( controlPointGrid->sto_xyz );
  else 
    splineMatrix = &( controlPointGrid->qto_xyz );

  for (z=0; z<controlPointGrid->nz; z++)
  {
    index = z*controlPointGrid->nx*controlPointGrid->ny;

    for (y=0; y<controlPointGrid->ny; y++)
    {
      for (x=0; x<controlPointGrid->nx; x++)
      {
	  
	// The initial control point position
	xInit =
	  splineMatrix->m[0][0]*static_cast<float>(x) +
	  splineMatrix->m[0][1]*static_cast<float>(y) +
	  splineMatrix->m[0][2]*static_cast<float>(z) +
	  splineMatrix->m[0][3];
	yInit =
	  splineMatrix->m[1][0]*static_cast<float>(x) +
	  splineMatrix->m[1][1]*static_cast<float>(y) +
	  splineMatrix->m[1][2]*static_cast<float>(z) +
	  splineMatrix->m[1][3];
	zInit =
	  splineMatrix->m[2][0]*static_cast<float>(x) +
	  splineMatrix->m[2][1]*static_cast<float>(y) +
	  splineMatrix->m[2][2]*static_cast<float>(z) +
	  splineMatrix->m[2][3];

	// The final control point position:
	//    controlPointPtrX[index], controlPointPtrY[index], controlPointPtrZ[index];

	sphere->SetCenter( -controlPointPtrX[index], 
			   -controlPointPtrY[index],
			   controlPointPtrZ[index] );

	sphere->Update();

	vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
	polydataCopy->DeepCopy( sphere->GetOutput() );
	
	appendFilter->AddInput( polydataCopy );
	appendFilter->Update();

	index++;
      }
    }
  }

  return appendFilter->GetOutput();
}


// ---------------------------------------------------------------------------
// Create a VTK polydata object to visualise the deformation
// ---------------------------------------------------------------------------

vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataSurface( PlaneType plane,
								 nifti_image *controlPointGrid,
								 int xSkip,
								 int ySkip,
								 int zSkip )
{
  int i, j, k;
  int x, y, z, index;

  PrecisionTYPE *controlPointGridPtrX, *controlPointGridPtrY, *controlPointGridPtrZ;


  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkIdType id;

  vtkSmartPointer<vtkStructuredGrid> sgrid = vtkStructuredGrid::New();

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

  int nPoints = controlPointGrid->nx*controlPointGrid->ny*controlPointGrid->nz;

  controlPointGridPtrX = static_cast< PrecisionTYPE * >( controlPointGrid->data );
  controlPointGridPtrY = &controlPointGridPtrX[ nPoints ];
  controlPointGridPtrZ = &controlPointGridPtrY[ nPoints ];

  int nxPoints, nyPoints, nzPoints;


  nzPoints = 0;
  for ( z=1; z<controlPointGrid->nz; z += zSkip )
  {
    nyPoints = 0;
    for ( y=1; y<controlPointGrid->ny; y += ySkip )
    {

      index = z*controlPointGrid->nx*controlPointGrid->ny 
	+ y*controlPointGrid->nx + 1;

      nxPoints = 0;
      for ( x=1; x<controlPointGrid->nx; x += xSkip )
      {

	// The final control point position:
	//    controlPointGridPtrX[index], controlPointGridPtrY[index], controlPointGridPtrZ[index];

	id = points->InsertNextPoint( -controlPointGridPtrX[index], 
				      -controlPointGridPtrY[index],
				      controlPointGridPtrZ[index] );

	cells->InsertNextCell( 1 );
	cells->InsertCellPoint( id );

	index += xSkip;

	nxPoints++;
      }
      nyPoints++;
    }
    nzPoints++;
  }

  sgrid->SetDimensions(nxPoints, nyPoints, nzPoints);
  sgrid->SetPoints( points );
  

  // Add the deformation planes to the DataManager
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

  vtkSmartPointer<vtkStructuredGridGeometryFilter> structuredGridFilter = vtkSmartPointer<vtkStructuredGridGeometryFilter>::New();

  structuredGridFilter->SetInput( sgrid );

  switch (plane )
  {
  case PLANE_XY:
  {
    for ( k=0; k<nzPoints; k++ ) 
    {
      vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
      
      structuredGridFilter->SetExtent( 0, nxPoints-1, 
				       0, nyPoints-1, 
				       k, k );

      structuredGridFilter->Update();
      polydataCopy->DeepCopy( structuredGridFilter->GetOutput() );
      
      appendFilter->AddInput( polydataCopy );
      appendFilter->Update();
    }
    
    break;
  }

  case PLANE_XZ:
  {
    for (j=0; j<nyPoints; j++) 
    {
      vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
      
      structuredGridFilter->SetExtent( 0, nxPoints-1, 
				       j, j,
				       0, nzPoints-1 );
      structuredGridFilter->Update();
      polydataCopy->DeepCopy( structuredGridFilter->GetOutput() );
      
      appendFilter->AddInput( polydataCopy );
      appendFilter->Update();
    }
    
    break;
  }

  case PLANE_YZ:
  {
    for (i=0; i<nxPoints; i++) 
    {
      vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
      
      structuredGridFilter->SetExtent( i, i,
				       0, nyPoints-1, 
				       0, nzPoints-1 );
      structuredGridFilter->Update();
      polydataCopy->DeepCopy( structuredGridFilter->GetOutput() );
      
      appendFilter->AddInput( polydataCopy );
      appendFilter->Update();
    }
  }
  }

  vtkSmartPointer<vtkPolyData> polyData = appendFilter->GetOutput();
  return polyData;
}


// ---------------------------------------------------------------------------
// Create a VTK polydata object to visualise the deformation
// ---------------------------------------------------------------------------

void F3DControlGridToVTKPolyDataSurfaces( nifti_image *controlPointGrid,
					  nifti_image *referenceImage,
					  vtkSmartPointer<vtkPolyData> &xyDeformation,
					  vtkSmartPointer<vtkPolyData> &xzDeformation,
					  vtkSmartPointer<vtkPolyData> &yzDeformation )
{
   nifti_image *refinedGrid = nifti_copy_nim_info( controlPointGrid );

  refinedGrid->data = (void *) malloc( refinedGrid->nvox * refinedGrid->nbyper);

  memcpy( refinedGrid->data, controlPointGrid->data,
	  refinedGrid->nvox * refinedGrid->nbyper);

  reg_bspline_refineControlPointGrid( referenceImage, refinedGrid );

  xyDeformation = F3DControlGridToVTKPolyDataSurface( PLANE_XY, refinedGrid, 1, 1, 2 );
  xzDeformation = F3DControlGridToVTKPolyDataSurface( PLANE_YZ, refinedGrid, 2, 1, 1 );
  yzDeformation = F3DControlGridToVTKPolyDataSurface( PLANE_XZ, refinedGrid, 1, 2, 1 );

  nifti_image_free( refinedGrid );
}


} // end namespace niftk

