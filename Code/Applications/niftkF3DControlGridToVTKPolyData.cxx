/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkF3DControlGridToVTKPolyData.cxx
 * \page niftkF3DControlGridToVTKPolyData
 * \section niftkF3DControlGridToVTKPolyDataSummary Creates VTK polydata objects to help visualise a NiftyReg reg_f3d deformation.
 *
 */

#include <ostream>
#include <stdio.h>
#include <string>

#include <NifTKConfigure.h>

#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>

#include <vcl_cmath.h>

#include <_reg_ReadWriteImage.h>
#include <_reg_resampling.h>
#include <_reg_globalTransformation.h>
#include <_reg_localTransformation.h>
#include <_reg_tools.h>

#include <niftkF3DControlGridToVTKPolyData.h>
#include <niftkF3DControlGridToVTKPolyDataCLP.h>



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


  // Calculate the control grid skip factor to limit the size of the VTK data generated
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int controlGridSkipFactor = niftk::ComputeControlGridSkipFactor( controlPointGrid,
									    subSamplingFactor,
									    maxGridDimension );

  std::cout << "Plotting deformation for every " << controlGridSkipFactor << " control points"
	    << std::endl;


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


  // Create a VTK hedgehog object to visualise the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputHedghog.length() ) 
  {

    vtkSmartPointer<vtkPolyData> vtkDeformationHedgehog = vtkSmartPointer<vtkPolyData>::New();

    std::cout << "Generating deformation hedgehog..." << std::endl;
    vtkDeformationHedgehog = niftk::F3DControlGridToVTKPolyDataHedgehog( controlPointGrid, 
									 controlGridSkipFactor, 
									 controlGridSkipFactor, 
									 controlGridSkipFactor );

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

    writer->SetFileName( fileOutputHedghog.c_str() );
    writer->SetInput( vtkDeformationHedgehog );
    writer->SetFileType( VTK_BINARY );
    
    std::cout << "Writing deformation hedgehog: " << fileOutputHedghog << std::endl;
    writer->Write();
  }


  // Create a VTK vector field object to visualise the deformation field (using VTK arrow glyphs)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputDeformationVectorField.length() ) 
  {

    vtkSmartPointer<vtkPolyData> vtkDeformationVectorField = vtkSmartPointer<vtkPolyData>::New();

    std::cout << "Generating deformation vector field..." << std::endl;
    vtkDeformationVectorField = 
      niftk::F3DControlGridToVTKPolyDataVectorField( controlPointGrid, 
						     controlGridSkipFactor, 
						     controlGridSkipFactor, 
						     controlGridSkipFactor );

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

    writer->SetFileName( fileOutputDeformationVectorField.c_str() );
    writer->SetInput( vtkDeformationVectorField );
    writer->SetFileType( VTK_BINARY );
    
    std::cout << "Writing deformation vector field: " 
	      << fileOutputDeformationVectorField << std::endl;

    writer->Write();
  }


  // Create a VTK polydata file with a surface for each orthogonal plane of control points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputControlGridPlaneXY.length() ||
       fileOutputControlGridPlaneYZ.length() ||
       fileOutputControlGridPlaneXZ.length() )
  {

    vtkSmartPointer<vtkPolyData> xyControlGrid = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyData> xzControlGrid = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyData> yzControlGrid = vtkSmartPointer<vtkPolyData>::New();

    std::cout << "Generating control grid visualisation..." << std::endl;
    niftk::F3DControlGridToVTKPolyDataSurfaces( controlPointGrid, referenceImage,
						controlGridSkipFactor,
						xyControlGrid, xzControlGrid, yzControlGrid );
    if ( fileOutputControlGridPlaneXY.length() )
    {
      vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

      writer->SetFileName( fileOutputControlGridPlaneXY.c_str() );
      writer->SetInput( xyControlGrid );
      writer->SetFileType( VTK_BINARY );
    
      std::cout << "Writing xy visualisation: " << fileOutputControlGridPlaneXY << std::endl;
      writer->Write();
    }

    if ( fileOutputControlGridPlaneXZ.length() )
    {
      vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

      writer->SetFileName( fileOutputControlGridPlaneXZ.c_str() );
      writer->SetInput( xzControlGrid );
      writer->SetFileType( VTK_BINARY );
    
      std::cout << "Writing xz visualisation: " << fileOutputControlGridPlaneXZ << std::endl;
      writer->Write();
    }

    if ( fileOutputControlGridPlaneYZ.length() )
    {
      vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();

      writer->SetFileName( fileOutputControlGridPlaneYZ.c_str() );
      writer->SetInput( yzControlGrid );
      writer->SetFileType( VTK_BINARY );
    
      std::cout << "Writing yz visualisation: " << fileOutputControlGridPlaneYZ << std::endl;
      writer->Write();
    }
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
    niftk::F3DDeformationToVTKPolyDataSurfaces( controlPointGrid, referenceImage, 
						controlGridSkipFactor,
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

  nifti_image_free( controlPointGrid );
  nifti_image_free( referenceImage );


  return EXIT_SUCCESS;
}
