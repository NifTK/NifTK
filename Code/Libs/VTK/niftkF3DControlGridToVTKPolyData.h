/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __NIFTKF3DCONTROLGRIDTOVTKPOLYDATA_H
#define __NIFTKF3DCONTROLGRIDTOVTKPOLYDATA_H


#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"
#include "niftkVTKWin32ExportHeader.h"

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
  
  /// Calculate the number of control points to skip when plotting the deformation field
  extern "C++" NIFTKVTK_WINEXPORT 
    unsigned int ComputeControlGridSkipFactor( nifti_image *controlPointGrid,
					       unsigned int subSamplingFactor,
					       unsigned int maxGridDimension );

  /// Create a reference image corresponding to a given control point grid image
  extern "C++" NIFTKVTK_WINEXPORT 
    nifti_image *AllocateReferenceImageGivenControlPointGrid( nifti_image *controlPointGrid );

  /// Create a deformation image corresponding to a given reference image
  extern "C++" NIFTKVTK_WINEXPORT 
    nifti_image *AllocateDeformationGivenReferenceImage( nifti_image *referenceImage );

  /// Create a VTK polydata object to visualise the control points
  extern "C++" NIFTKVTK_WINEXPORT 
    vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataPoints( nifti_image *controlPointGrid );
    
  /// Create a VTK polydata object to visualise the control points using spheres
  extern "C++" NIFTKVTK_WINEXPORT 
    vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataSpheres( nifti_image *controlPointGrid,
								     float radius );

  /// Create VTK polydata objects to visualise the deformations
  extern "C++" NIFTKVTK_WINEXPORT 
    void F3DControlGridToVTKPolyDataSurfaces( nifti_image *controlPointGrid,
					      nifti_image *referenceImage,
					      int controlGridSkipFactor,
					      vtkSmartPointer<vtkPolyData> &xyDeformation,
					      vtkSmartPointer<vtkPolyData> &xzDeformation,
					      vtkSmartPointer<vtkPolyData> &yzDeformation );

  /// Create a VTK hedgehog object to visualise the deformation field
  extern "C++" NIFTKVTK_WINEXPORT 
    vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataHedgehog( nifti_image *deformation,
								      int xSkip,
								      int ySkip,
								      int zSkip );


  /// Create a VTK polydata vector field object to visualise the deformation field (using VTK arrow glyphs)
  extern "C++" NIFTKVTK_WINEXPORT 
    vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataVectorField( nifti_image *deformation,
									 int xSkip,
									 int ySkip,
									 int zSkip );


  /// Create a VTK polydata object to visualise the deformation
  extern "C++" NIFTKVTK_WINEXPORT 
    vtkSmartPointer<vtkPolyData> F3DDeformationToVTKPolyDataSurface( PlaneType plane,
								     nifti_image *controlPointGrid,
								     int xSkip,
								     int ySkip,
								     int zSkip );

  /// Create VTK polydata objects to visualise the deformations
  extern "C++" NIFTKVTK_WINEXPORT 
    void F3DDeformationToVTKPolyDataSurfaces( nifti_image *controlPointGrid,
					      nifti_image *referenceImage,
					      int controlGridSkipFactor,
					      vtkSmartPointer<vtkPolyData> &xyDeformation,
					      vtkSmartPointer<vtkPolyData> &xzDeformation,
					      vtkSmartPointer<vtkPolyData> &yzDeformation );



} // end namespace niftk

#endif  //__NIFTKF3DCONTROLGRIDTOVTKPOLYDATA_H
