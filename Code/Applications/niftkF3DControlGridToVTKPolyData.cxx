/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: jhh$

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

/*!
 * \file niftkF3DControlGridToVTKPolyData.cxx
 * \page niftkF3DControlGridToVTKPolyData
 * \section niftkF3DControlGridToVTKPolyDataSummary Creates VTK polydata objects to help visualise a NiftyReg reg_f3d deformation.
 *
 */

#include <ostream>
#include <stdio.h>
#include <string>

#include "NifTKConfigure.h"

#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>

#include <vcl_cmath.h>

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"

#include "niftkF3DControlGridToVTKPolyData.h"
#include "niftkF3DControlGridToVTKPolyDataCLP.h"



// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputImage.length() == 0 || fileInputControlGrid.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }


  // Read the input control grid image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  nifti_image *controlPointGrid = NULL;

  std::cout << "Reading input control grid: " << fileInputControlGrid << std::endl;
  controlPointGrid = reg_io_ReadImageFile( fileInputControlGrid.c_str() );

  if ( controlPointGrid == NULL )
  {
    std::cerr << "ERROR: Error reading the control point image: " 
	      << fileInputControlGrid << std::endl;
    return EXIT_FAILURE;
  }

  reg_checkAndCorrectDimension( controlPointGrid );

  int nControlPoints = controlPointGrid->nx*controlPointGrid->ny*controlPointGrid->nz;

  std::cout << "Number of control points: " 
	    << nControlPoints << std::endl
	    << "Control point grid dimensions: " 
	    << controlPointGrid->nx << " x " 
	    << controlPointGrid->ny << " x " 
	    << controlPointGrid->nz << std::endl
	    << "Control point grid spacing: " 
	    << controlPointGrid->dx << " x " 
	    << controlPointGrid->dy << " x " 
	    << controlPointGrid->dz << std::endl;


  // Read the input reference image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  nifti_image *referenceImage = NULL;

  std::cout << "Reading reference image: " << fileInputImage << std::endl;
  referenceImage = reg_io_ReadImageFile( fileInputImage.c_str() );

  if ( referenceImage == NULL )
  {
    std::cerr << "ERROR: Error reading the reference image: " 
	      << fileInputImage << std::endl;
    return EXIT_FAILURE;
  }

  reg_checkAndCorrectDimension( referenceImage );

  if ( radius <= 0.) {

    radius = vcl_sqrt( referenceImage->dx*referenceImage->dx +
                       referenceImage->dy*referenceImage->dy +
                       referenceImage->dz*referenceImage->dz );
  }


  // Create a VTK polydata file with a point for each control grid position
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputControlPoints.length() ) 
  {

    vtkSmartPointer<vtkPolyData> vtkControlPoints = vtkSmartPointer<vtkPolyData>::New();
  
    std::cout << "Generating control point..." << std::endl;
    vtkControlPoints = niftk::F3DControlGridToVTKPolyDataPoints( controlPointGrid );
  
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

    writer->SetFileName( fileOutputControlPoints.c_str() );
    writer->SetInput( vtkControlPoints );
    writer->SetFileType( VTK_BINARY );
    
    std::cout << "Writing control points: " << fileOutputControlPoints << std::endl;
    writer->Write();
  }


  // Create a VTK polydata file with a sphere for each control grid position
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputControlSpheres.length() ) 
  {

    vtkSmartPointer<vtkPolyData> vtkControlPointSpheres = vtkSmartPointer<vtkPolyData>::New();

    std::cout << "Generating control point spheres..." << std::endl;
    vtkControlPointSpheres = niftk::F3DControlGridToVTKPolyDataSpheres( controlPointGrid, radius );

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

    writer->SetFileName( fileOutputControlSpheres.c_str() );
    writer->SetInput( vtkControlPointSpheres );
    writer->SetFileType( VTK_BINARY );
    
    std::cout << "Writing control point spheres: " << fileOutputControlSpheres << std::endl;
    writer->Write();
  }


  // Create a VTK polydata file with a surface for each orthogonal deformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputDeformationPlaneXY.length() ||
       fileOutputDeformationPlaneYZ.length() ||
       fileOutputDeformationPlaneXZ.length() )
  {

    vtkSmartPointer<vtkPolyData> xyDeformation = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyData> xzDeformation = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyData> yzDeformation = vtkSmartPointer<vtkPolyData>::New();

    std::cout << "Generating deformation visualisation..." << std::endl;
    niftk::F3DControlGridToVTKPolyDataSurfaces( controlPointGrid, referenceImage,
						xyDeformation, xzDeformation, yzDeformation );

    if ( fileOutputDeformationPlaneXY.length() )
    {
      vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

      writer->SetFileName( fileOutputDeformationPlaneXY.c_str() );
      writer->SetInput( xyDeformation );
      writer->SetFileType( VTK_BINARY );
    
      std::cout << "Writing xy visualisation: " << fileOutputDeformationPlaneXY << std::endl;
      writer->Write();
    }

    if ( fileOutputDeformationPlaneXZ.length() )
    {
      vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

      writer->SetFileName( fileOutputDeformationPlaneXZ.c_str() );
      writer->SetInput( xzDeformation );
      writer->SetFileType( VTK_BINARY );
    
      std::cout << "Writing xz visualisation: " << fileOutputDeformationPlaneXZ << std::endl;
      writer->Write();
    }

    if ( fileOutputDeformationPlaneYZ.length() )
    {
      vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

      writer->SetFileName( fileOutputDeformationPlaneYZ.c_str() );
      writer->SetInput( yzDeformation );
      writer->SetFileType( VTK_BINARY );
    
      std::cout << "Writing yz visualisation: " << fileOutputDeformationPlaneYZ << std::endl;
      writer->Write();
    }
  }

  return EXIT_SUCCESS;
}
