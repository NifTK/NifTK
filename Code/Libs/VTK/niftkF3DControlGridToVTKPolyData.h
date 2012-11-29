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

  
/// Create a VTK polydata object to visualise the control points
vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataPoints( nifti_image *controlPointGrid );
    
/// Create a VTK polydata object to visualise the control points using spheres
vtkSmartPointer<vtkPolyData> F3DControlGridToVTKPolyDataSpheres( nifti_image *controlPointGrid,
								 float radius );

/// Create a VTK polydata object to visualise the deformation
void F3DControlGridToVTKPolyDataSurfaces( nifti_image *controlPointGrid,
					  vtkSmartPointer<vtkPolyData> &xyDeformation,
					  vtkSmartPointer<vtkPolyData> &xzDeformation,
					  vtkSmartPointer<vtkPolyData> &yzDeformation );



} // end namespace niftk

#endif  //__NIFTKF3DCONTROLGRIDTOVTKPOLYDATA_H
