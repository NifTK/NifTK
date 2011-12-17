/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 17:44:42 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7864 $
 Last modified by  : $Author: kkl $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkTransformFileWriter.h"
#include "itkGE_TomosynthesisGeometry.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl
	      << "  Create a set of tomosynthesis projection matrices."
	      << std::endl << std::endl

	      << "  " << exec 
	      << " -sz3D Nx Ny Nz -res3D dx dy dz" 
	      << " -sz2D Nx Ny Nz -res2D dx dy dz" 
	      << " -o OutputFilestem " << std::endl << std::endl

	      << "*** [mandatory] ***" << std::endl << std::endl
	      << "    -sz2D   <int>   <int>            The size of the 2D projection images in pixels" << std::endl
	      << "    -res2D  <float> <float>          The resolution of the 2D projection images in mm" << std::endl
	      << "    -sz3D   <int>   <int>   <int>    The size of the 3D volume in voxels" << std::endl
	      << "    -res3D  <float> <float> <float>  The resolution of the 3D volume in mm" << std::endl
	      << "    -o      <filename>               Output the output file stem for the transformation files" << std::endl << std::endl;
  }


/**
 * \brief Create a set of tomosynthesis projection matrices.
 */
int main(int argc, char** argv)
{
  std::string fileOutputFilestem;

  typedef float IntensityType;
  typedef itk::GE_TomosynthesisGeometry< IntensityType > GE_TomosynthesisGeometryType;

  GE_TomosynthesisGeometryType::ProjectionSizeType pProjectionSize;
  GE_TomosynthesisGeometryType::ProjectionSpacingType pProjectionSpacing;

  GE_TomosynthesisGeometryType::VolumeSizeType pVolumeSize;
  GE_TomosynthesisGeometryType::VolumeSpacingType pVolumeSpacing;

  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  for(int i=1; i < argc; i++){

    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }

    

    else if(strcmp(argv[i], "-sz2D") == 0) {
      pProjectionSize[0] = atoi(argv[++i]);
      pProjectionSize[1] = atoi(argv[++i]);
      std::cout << "Set -sz2D="
				    << niftk::ConvertToString((int) pProjectionSize[0]) << " "
				    << niftk::ConvertToString((int) pProjectionSize[1]) << std::endl;
    }
    else if(strcmp(argv[i], "-res2D") == 0) {
      pProjectionSpacing[0] = atof(argv[++i]);
      pProjectionSpacing[1] = atof(argv[++i]);
      std::cout << "Set -res2D="
				    << niftk::ConvertToString(pProjectionSpacing[0]) << " "
				    << niftk::ConvertToString(pProjectionSpacing[1]) << std::endl;
    }

    else if(strcmp(argv[i], "-sz3D") == 0) {
      pVolumeSize[0] = atoi(argv[++i]);
      pVolumeSize[1] = atoi(argv[++i]);
      pVolumeSize[2] = atoi(argv[++i]);
      std::cout << "Set -sz3D="
				    << niftk::ConvertToString((int) pVolumeSize[0]) << " "
				    << niftk::ConvertToString((int) pVolumeSize[1]) << " "
				    << niftk::ConvertToString((int) pVolumeSize[2]) << std::endl;
    }
    else if(strcmp(argv[i], "-res3D") == 0) {
      pVolumeSpacing[0] = atof(argv[++i]);
      pVolumeSpacing[1] = atof(argv[++i]);
      pVolumeSpacing[2] = atof(argv[++i]);
      std::cout << "Set -res3D="
    		  << niftk::ConvertToString(pVolumeSpacing[0]) << " "
    		  << niftk::ConvertToString(pVolumeSpacing[1]) << " "
    		  << niftk::ConvertToString(pVolumeSpacing[2]) << std::endl;
    }

    else if(strcmp(argv[i], "-o") == 0) {
      fileOutputFilestem = argv[++i];
      std::cout << "Set -o=" << fileOutputFilestem << std::endl;
    }

    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputFilestem.length() == 0 ) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }
      

  // Create the tomosynthesis geometry
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  char filename[256];

  GE_TomosynthesisGeometryType::Pointer geometry = GE_TomosynthesisGeometryType::New();
  
  geometry->SetProjectionSize(pProjectionSize);
  geometry->SetProjectionSpacing(pProjectionSpacing);

  geometry->SetVolumeSize(pVolumeSize);
  geometry->SetVolumeSpacing(pVolumeSpacing);

  GE_TomosynthesisGeometryType::EulerAffineTransformPointerType pAffineTransform;
  GE_TomosynthesisGeometryType::PerspectiveProjectionTransformPointerType pPerspectiveTransform;

  typedef itk::TransformFileWriter TransformFileWriterType;
  TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

  unsigned int iProjection;

  for (iProjection=0; iProjection<geometry->GetNumberOfProjections(); iProjection++) {

    // Get and write the affine transform

    try {
      pAffineTransform = geometry->GetAffineTransform(iProjection);
    }

    catch( itk::ExceptionObject & err ) { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }                

    sprintf(filename, "%s_%02d.tAffine", fileOutputFilestem.c_str(), iProjection);

    transformFileWriter->SetInput(pAffineTransform);
    transformFileWriter->SetFileName(std::string(filename)); 
    transformFileWriter->Update();         
    
    std::cout << "Writing affine transform: " << filename << std::endl;
  

    // Get and write the perspective transform

    try {
      pPerspectiveTransform = geometry->GetPerspectiveTransform(iProjection);
    }

    catch( itk::ExceptionObject & err ) { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }                

    sprintf(filename, "%s_%02d.tPerspective", fileOutputFilestem.c_str(), iProjection);

    transformFileWriter->SetInput(pPerspectiveTransform);
    transformFileWriter->SetFileName(std::string(filename)); 
    transformFileWriter->Update();         

    std::cout << "Writing perspective transform: " << filename << std::endl;
  }


  std::cout << "Done" << std::endl;
  
  return EXIT_SUCCESS;   
}


