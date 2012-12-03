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

#ifndef __NIFTKF3DCONTROLGRIDTOVTKPOLYDATA_H
#define __NIFTKF3DCONTROLGRIDTOVTKPOLYDATA_H


#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"

#include <ostream>
#include <stdio.h>
#include <string>

#include "nifti1_io.h"

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace niftk
{

  typedef enum {
    PLANE_XY,           //!< Create the 'xy' plane deformation field
    PLANE_XZ,           //!< Create the 'xz' plane deformation field
    PLANE_YZ,           //!< Create the 'yz' plane deformation field
  } PlaneType;                                             
  

  /// Create a reference image corresponding to a given control point grid image
  nifti_image *AllocateReferenceImageGivenControlPointGrid( nifti_image *controlPointGrid );

  /// Create a deformation image corresponding to a given reference image
  nifti_image *AllocateDeformationGivenReferenceImage( nifti_image *referenceImage );

  /// Create a VTK polydata object to visualise the control points
  vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataPoints( nifti_image *controlPointGrid );
    
  /// Create a VTK polydata object to visualise the control points using spheres
  vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataSpheres( nifti_image *controlPointGrid,
								   float radius );

  /// Create VTK polydata objects to visualise the deformations
  void F3DControlGridToVTKPolyDataSurfaces( nifti_image *controlPointGrid,
					    nifti_image *referenceImage,
					    vtkSmartPointer<vtkPolyData> &xyDeformation,
					    vtkSmartPointer<vtkPolyData> &xzDeformation,
					    vtkSmartPointer<vtkPolyData> &yzDeformation );

  /// Create a VTK hedgehog object to visualise the deformation field
  vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataHedgehog( nifti_image *deformation,
								    int xSkip,
								    int ySkip,
								    int zSkip );


  /// Create a VTK polydata vector field object to visualise the deformation field
  vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataVectorField( nifti_image *deformation,
								       int xSkip,
								       int ySkip,
								       int zSkip );


  /// Create a VTK polydata object to visualise the deformation
  vtkSmartPointer<vtkPolyData> F3DDeformationToVTKPolyDataSurface( PlaneType plane,
								   nifti_image *controlPointGrid,
								   int xSkip,
								   int ySkip,
								   int zSkip );

  /// Create VTK polydata objects to visualise the deformations
  void F3DDeformationToVTKPolyDataSurfaces( nifti_image *controlPointGrid,
					    nifti_image *referenceImage,
					    vtkSmartPointer<vtkPolyData> &xyDeformation,
					    vtkSmartPointer<vtkPolyData> &xzDeformation,
					    vtkSmartPointer<vtkPolyData> &yzDeformation );



} // end namespace niftk

#endif  //__NIFTKF3DCONTROLGRIDTOVTKPOLYDATA_H
