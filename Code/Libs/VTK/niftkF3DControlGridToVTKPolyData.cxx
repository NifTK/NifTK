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

#include <vtkAppendPolyData.h>
#include <vtkArrowSource.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkGlyph3D.h>
#include <vtkHedgeHog.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSphereSource.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridGeometryFilter.h>
#include <vtkVersion.h>
#include <vtkVertex.h>

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
// Create a reference image corresponding to a given control point grid image
// ---------------------------------------------------------------------------

nifti_image *AllocateReferenceImageGivenControlPointGrid( nifti_image *controlPointGrid )
{
  nifti_image *referenceImage = nifti_copy_nim_info( controlPointGrid );

  referenceImage->dim[0] = referenceImage->ndim = 3;

  referenceImage->dim[1] = referenceImage->nx = controlPointGrid->nx;
  referenceImage->dim[2] = referenceImage->ny = controlPointGrid->ny;
  referenceImage->dim[3] = referenceImage->nz = controlPointGrid->nz;

  referenceImage->dim[4]    = referenceImage->nt = 1;
  referenceImage->pixdim[4] = referenceImage->dt = 1.0;

  referenceImage->dim[5]    = referenceImage->nu = 1;
  referenceImage->pixdim[5] = referenceImage->du = 1.0;

  referenceImage->dim[6]    = referenceImage->nv = 1;
  referenceImage->pixdim[6] = referenceImage->dv = 1.0;

  referenceImage->dim[7]    = referenceImage->nw = 1;
  referenceImage->pixdim[7] = referenceImage->dw = 1.0;

  referenceImage->nvox = 
    referenceImage->nx*
    referenceImage->ny*
    referenceImage->nz*
    referenceImage->nt*
    referenceImage->nu;

  if ( sizeof(PrecisionTYPE) == 8 ) 
    referenceImage->datatype = NIFTI_TYPE_FLOAT64;
  else 
    referenceImage->datatype = NIFTI_TYPE_FLOAT32;

  referenceImage->intent_code = 0;
  referenceImage->nbyper = sizeof(PrecisionTYPE);

  referenceImage->data = (void *) calloc( referenceImage->nvox, 
					  referenceImage->nbyper );

  return referenceImage;
}


// ---------------------------------------------------------------------------
// Create a deformation image corresponding to a given reference image
// ---------------------------------------------------------------------------

nifti_image *AllocateDeformationGivenReferenceImage( nifti_image *referenceImage )
{
  
  nifti_image *deformationFieldImage = nifti_copy_nim_info( referenceImage );

  deformationFieldImage->dim[0] = deformationFieldImage->ndim = 5;

  deformationFieldImage->dim[1] = deformationFieldImage->nx = referenceImage->nx;
  deformationFieldImage->dim[2] = deformationFieldImage->ny = referenceImage->ny;
  deformationFieldImage->dim[3] = deformationFieldImage->nz = referenceImage->nz;

  deformationFieldImage->dim[4]    = deformationFieldImage->nt = 1;
  deformationFieldImage->pixdim[4] = deformationFieldImage->dt = 1.0;

  if ( referenceImage->nz > 1 )
    deformationFieldImage->dim[5]  = deformationFieldImage->nu = 3;
  else 
    deformationFieldImage->dim[5]  = deformationFieldImage->nu = 2;

  deformationFieldImage->pixdim[5] = deformationFieldImage->du = 1.0;

  deformationFieldImage->dim[6]    = deformationFieldImage->nv = 1;
  deformationFieldImage->pixdim[6] = deformationFieldImage->dv = 1.0;

  deformationFieldImage->dim[7]    = deformationFieldImage->nw = 1;
  deformationFieldImage->pixdim[7] = deformationFieldImage->dw = 1.0;

  deformationFieldImage->nvox = 
    deformationFieldImage->nx*
    deformationFieldImage->ny*
    deformationFieldImage->nz*
    deformationFieldImage->nt*
    deformationFieldImage->nu;

  if ( sizeof(PrecisionTYPE) == 8 ) 
    deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
  else 
    deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;

  deformationFieldImage->intent_code = NIFTI_INTENT_VECTOR;
  deformationFieldImage->nbyper = sizeof(PrecisionTYPE);

  deformationFieldImage->data = (void *) calloc( deformationFieldImage->nvox, 
						 deformationFieldImage->nbyper );

  return deformationFieldImage;
}


// ---------------------------------------------------------------------------
// Create a VTK polydata object to visualise the control points
// ---------------------------------------------------------------------------

vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataPoints( nifti_image *controlPointGrid )
{
  int x, y, z, index;

  PrecisionTYPE *controlPointPtrX, *controlPointPtrY, *controlPointPtrZ;

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

  for (z=0; z<controlPointGrid->nz; z++)
  {
    index = z*controlPointGrid->nx*controlPointGrid->ny;

    for (y=0; y<controlPointGrid->ny; y++)
    {
      for (x=0; x<controlPointGrid->nx; x++)
      {
	  
	id = points->InsertNextPoint( -controlPointPtrX[index], 
				      -controlPointPtrY[index],
				      controlPointPtrZ[index] );

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

  PrecisionTYPE *controlPointPtrX, *controlPointPtrY, *controlPointPtrZ;

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

  for (z=0; z<controlPointGrid->nz; z++)
  {
    index = z*controlPointGrid->nx*controlPointGrid->ny;

    for (y=0; y<controlPointGrid->ny; y++)
    {
      for (x=0; x<controlPointGrid->nx; x++)
      {
	  
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
// Create a VTK hedgehog object to visualise the deformation field
// ---------------------------------------------------------------------------

vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataHedgehog( nifti_image *controlPointGrid,
								  int xSkip,
								  int ySkip,
								  int zSkip )
{
  int x, y, z, index;
  float xInit, yInit, zInit;

  PrecisionTYPE *deformationPtrX, *deformationPtrY, *deformationPtrZ;

  mat44 *splineMatrix;


  // Generate the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  nifti_image *referenceImage = AllocateReferenceImageGivenControlPointGrid( controlPointGrid );

  nifti_image *deformation = AllocateDeformationGivenReferenceImage( referenceImage );
  reg_getDeformationFromDisplacement( deformation );

  std::cerr << std::endl << "Deformation field image: " << std::endl << std::endl;
  nifti_image_infodump( deformation );


  reg_spline_getDeformationField( controlPointGrid,
				  referenceImage,
				  deformation,
				  NULL, // mask
				  true, //composition
				  true // bspline
    );

  nifti_image_free( referenceImage );


  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkIdType id;

  vtkSmartPointer<vtkStructuredGrid> sgrid = vtkStructuredGrid::New();

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  vtkSmartPointer<vtkFloatArray> displacements = vtkSmartPointer<vtkFloatArray>::New();
  displacements->SetNumberOfComponents(3);

  int nPoints = deformation->nx*deformation->ny*deformation->nz;

  deformationPtrX = static_cast< PrecisionTYPE * >( deformation->data );
  deformationPtrY = &deformationPtrX[ nPoints ];
  deformationPtrZ = &deformationPtrY[ nPoints ];

  if ( deformation->sform_code > 0 ) 
    splineMatrix = &( deformation->sto_xyz );
  else 
    splineMatrix = &( deformation->qto_xyz );


  float v[3];
  int nxPoints = 0, nyPoints = 0, nzPoints = 0;

  nzPoints = 0;
  for ( z=1; z<deformation->nz; z += zSkip )
  {
    nyPoints = 0;
    for ( y=1; y<deformation->ny; y += ySkip )
    {

      index = z*deformation->nx*deformation->ny 
	+ y*deformation->nx + 1;

      nxPoints = 0;
      for ( x=1; x<deformation->nx; x += xSkip )
      {
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
	//    deformationPtrX[index], deformationPtrY[index], deformationPtrZ[index];

	id = points->InsertNextPoint( -xInit, 
				      -yInit,
				       zInit );

	v[0] = -( deformationPtrX[index] - xInit ); 
	v[1] = -( deformationPtrY[index] - yInit );
	v[2] =    deformationPtrZ[index] - zInit;

	displacements->InsertNextTuple( v );

	index += xSkip;

	nxPoints++;
      }
      nyPoints++;
    }
    nzPoints++;
  }

  sgrid->SetDimensions(nxPoints, nyPoints, nzPoints);
  sgrid->SetPoints( points );  
  sgrid->GetPointData()->SetVectors( displacements );


  vtkSmartPointer<vtkHedgeHog> hedgehog = vtkSmartPointer<vtkHedgeHog>::New();
  hedgehog->SetInput( sgrid );
  hedgehog->SetScaleFactor( 1. );

  vtkSmartPointer<vtkPolyData> polyData = hedgehog->GetOutput();
  return polyData;
}


// ---------------------------------------------------------------------------
// Create a VTK polydata vector field object to visualise the deformation field
// ---------------------------------------------------------------------------

vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataVectorField( nifti_image *controlPointGrid,
								     int xSkip,
								     int ySkip,
								     int zSkip )
{
  int x, y, z, index;
  float xInit, yInit, zInit;

  PrecisionTYPE *deformationPtrX, *deformationPtrY, *deformationPtrZ;

  mat44 *splineMatrix;


  // Generate the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  nifti_image *referenceImage = AllocateReferenceImageGivenControlPointGrid( controlPointGrid );

  nifti_image *deformation = AllocateDeformationGivenReferenceImage( referenceImage );
  reg_getDeformationFromDisplacement( deformation );

  std::cerr << std::endl << "Deformation field image: " << std::endl << std::endl;
  nifti_image_infodump( deformation );


  reg_spline_getDeformationField( controlPointGrid,
				  referenceImage,
				  deformation,
				  NULL, // mask
				  true, //composition
				  true // bspline
    );

  nifti_image_free( referenceImage );


  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkIdType id;

  vtkSmartPointer<vtkStructuredGrid> sgrid = vtkStructuredGrid::New();

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  vtkSmartPointer<vtkFloatArray> displacements = vtkSmartPointer<vtkFloatArray>::New();
  displacements->SetNumberOfComponents(3);

  int nPoints = deformation->nx*deformation->ny*deformation->nz;

  // Setup the arrows
  vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New();
  arrowSource->Update();
 
  vtkSmartPointer<vtkGlyph3D> glyphFilter = vtkSmartPointer<vtkGlyph3D>::New();

  glyphFilter->SetSourceConnection(arrowSource->GetOutputPort());
  glyphFilter->OrientOn();
  glyphFilter->SetVectorModeToUseVector();
  glyphFilter->SetColorModeToColorByVector();
  glyphFilter->SetScaleModeToScaleByVector();
  glyphFilter->SetColorModeToColorByVector();

  deformationPtrX = static_cast< PrecisionTYPE * >( deformation->data );
  deformationPtrY = &deformationPtrX[ nPoints ];
  deformationPtrZ = &deformationPtrY[ nPoints ];

  if ( deformation->sform_code > 0 ) 
    splineMatrix = &( deformation->sto_xyz );
  else 
    splineMatrix = &( deformation->qto_xyz );


  float v[3];
  int nxPoints = 0, nyPoints = 0, nzPoints = 0;

  nzPoints = 0;
  for ( z=1; z<deformation->nz; z += zSkip )
  {
    nyPoints = 0;
    for ( y=1; y<deformation->ny; y += ySkip )
    {

      index = z*deformation->nx*deformation->ny 
	+ y*deformation->nx + 1;

      nxPoints = 0;
      for ( x=1; x<deformation->nx; x += xSkip )
      {
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
	//    deformationPtrX[index], deformationPtrY[index], deformationPtrZ[index];

	id = points->InsertNextPoint( -xInit, 
				      -yInit,
				       zInit );

	v[0] = -( deformationPtrX[index] - xInit ); 
	v[1] = -( deformationPtrY[index] - yInit );
	v[2] =    deformationPtrZ[index] - zInit;

	displacements->InsertNextTuple( v );

	index += xSkip;

	nxPoints++;
      }
      nyPoints++;
    }
    nzPoints++;
  }

  nifti_image_free( deformation );

  sgrid->SetDimensions(nxPoints, nyPoints, nzPoints);
  sgrid->SetPoints( points );  
  sgrid->GetPointData()->SetVectors( displacements );


  glyphFilter->SetInputConnection( sgrid->GetProducerPort() );
  glyphFilter->Update();

  vtkSmartPointer<vtkPolyData> polyData = glyphFilter->GetOutput();
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

  xyDeformation = F3DDeformationToVTKPolyDataSurface( PLANE_XY, refinedGrid, 1, 1, 2 );
  xzDeformation = F3DDeformationToVTKPolyDataSurface( PLANE_YZ, refinedGrid, 2, 1, 1 );
  yzDeformation = F3DDeformationToVTKPolyDataSurface( PLANE_XZ, refinedGrid, 1, 2, 1 );

  nifti_image_free( refinedGrid );
}


// ---------------------------------------------------------------------------
// Create a VTK polydata object to visualise the deformation
// ---------------------------------------------------------------------------

vtkSmartPointer<vtkPolyData> F3DDeformationToVTKPolyDataSurface( PlaneType plane,
								 nifti_image *deformation,
								 int xSkip,
								 int ySkip,
								 int zSkip )
{
  int i, j, k;
  int x, y, z, index;

  PrecisionTYPE *deformationPtrX, *deformationPtrY, *deformationPtrZ;


  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkIdType id;

  vtkSmartPointer<vtkStructuredGrid> sgrid = vtkStructuredGrid::New();

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

  int nPoints = deformation->nx*deformation->ny*deformation->nz;

  deformationPtrX = static_cast< PrecisionTYPE * >( deformation->data );
  deformationPtrY = &deformationPtrX[ nPoints ];
  deformationPtrZ = &deformationPtrY[ nPoints ];

  int nxPoints = 0, nyPoints = 0, nzPoints = 0;


  nzPoints = 0;
  for ( z=1; z<deformation->nz; z += zSkip )
  {
    nyPoints = 0;
    for ( y=1; y<deformation->ny; y += ySkip )
    {

      index = z*deformation->nx*deformation->ny 
	+ y*deformation->nx + 1;

      nxPoints = 0;
      for ( x=1; x<deformation->nx; x += xSkip )
      {

	// The final control point position:
	//    deformationPtrX[index], deformationPtrY[index], deformationPtrZ[index];

	id = points->InsertNextPoint( -deformationPtrX[index], 
				      -deformationPtrY[index],
				      deformationPtrZ[index] );

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
  

  // Create the orthogonal deformation planes
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

void F3DDeformationToVTKPolyDataSurfaces( nifti_image *controlPointGrid,
					  nifti_image *targetImage,
					  vtkSmartPointer<vtkPolyData> &xyDeformation,
					  vtkSmartPointer<vtkPolyData> &xzDeformation,
					  vtkSmartPointer<vtkPolyData> &yzDeformation )
{
  nifti_image *refinedGrid = nifti_copy_nim_info( controlPointGrid );

  refinedGrid->data = (void *) malloc( refinedGrid->nvox * refinedGrid->nbyper);

  memcpy( refinedGrid->data, controlPointGrid->data,
	  refinedGrid->nvox * refinedGrid->nbyper);
  
  reg_bspline_refineControlPointGrid( targetImage, refinedGrid );

  // Generate the deformation field

  nifti_image *referenceImage = AllocateReferenceImageGivenControlPointGrid( refinedGrid );

  nifti_image *deformationFieldImage = AllocateDeformationGivenReferenceImage( referenceImage );
  reg_getDeformationFromDisplacement( deformationFieldImage );

  std::cerr << std::endl << "Deformation field image: " << std::endl << std::endl;
  nifti_image_infodump( deformationFieldImage );


  reg_spline_getDeformationField( refinedGrid,
				  referenceImage,
				  deformationFieldImage,
				  NULL, // mask
				  true, //composition
				  true // bspline
    );


  xyDeformation = F3DDeformationToVTKPolyDataSurface( PLANE_XY, deformationFieldImage, 1, 1, 2 );
  xzDeformation = F3DDeformationToVTKPolyDataSurface( PLANE_YZ, deformationFieldImage, 2, 1, 1 );
  yzDeformation = F3DDeformationToVTKPolyDataSurface( PLANE_XZ, deformationFieldImage, 1, 2, 1 );

  nifti_image_free( referenceImage );
  nifti_image_free( deformationFieldImage );

  nifti_image_free( refinedGrid );
}


} // end namespace niftk

