/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkEulerAffineTransform.h>
#include <itkPerspectiveProjectionTransform.h>
#include <itkTransformFileWriter.h>
#include <itkAffineTransform.h>

/*!
 * \file niftkCreateAffineTransform.cxx
 * \page niftkCreateAffineTransform
 * \section niftkCreateAffineTransformSummary Creates an ITK affine transformation from user specified parameters.
 *
 * \section niftkCreateAffineTransformCaveat Caveats
 */

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    
    std::cout << "  " << std::endl
	      << "  Creates an ITK affine transformation from user specified parameters." << std::endl
	      << "  (transformation order is: UnChangeOrigin Translate Rx Ry Rz Scale Skew ChangeOrigin)." << std::endl
	      << "  If your origin is (0,0,0), then the order is the same as SPM, but we specify rotations in degrees not radians." << std::endl << std::endl

	      << "  " << exec 
	      << " [-ot AffineTransform | -om AffineMatrix | -omp AffineMatrix] [options]" << std::endl << "  " << std::endl

	      << "*** [mandatory, at least one of] ***" << std::endl << std::endl
	      << "    -ot <filename>        Output UCL transformation" << std::endl
	      << "    -om <filename>        Output matrix transformation as ITK" << std::endl  
        << "    -omp <filename>       Output matrix transformation as plain 4 x 4 text file" << std::endl << std::endl
	      << "*** [options]   ***" << std::endl << std::endl
	      << "    -cx <float>           Origin of the transformation in 'x' [0]" << std::endl
	      << "    -cy <float>           Origin of the transformation in 'y' [0]" << std::endl
	      << "    -cz <float>           Origin of the transformation in 'z' [0]" << std::endl
	      << "    -tx <float>           Translation along the 'x' axis (mm) [0]" << std::endl
	      << "    -ty <float>           Translation along the 'y' axis (mm) [0]" << std::endl
	      << "    -tz <float>           Translation along the 'z' axis (mm) [0]" << std::endl
	      << "    -rx <float>           Rotation about the 'x' axis (degrees) [0]" << std::endl
	      << "    -ry <float>           Rotation about the 'y' axis (degrees) [0]" << std::endl
	      << "    -rz <float>           Rotation about the 'z' axis (degrees) [0]" << std::endl
	      << "    -sx <float>           Scale factor along the 'x' axis [1]" << std::endl
	      << "    -sy <float>           Scale factor along the 'y' axis [1]" << std::endl
	      << "    -sz <float>           Scale factor along the 'z' axis [1]" << std::endl
	      << "    -k1 <float>           Skew factor 1 [0]" << std::endl
	      << "    -k2 <float>           Skew factor 2 [0]" << std::endl
	      << "    -k3 <float>           Skew factor 3 [0]" << std::endl << std::endl;
  }


/**
 * \brief Create an affine transformation with various formats.
 */
int main(int argc, char** argv)
{
  std::string fileOutputTransformation;	// The output transformation file 
  std::string fileOutputMatrix;	        // The output transformation matrix
  std::string fileOutputPlain;          // The output transformation matrix
  
  typedef itk::EulerAffineTransform<double, 3, 3> EulerAffineTransformType;

  EulerAffineTransformType::InputPointType center;
  EulerAffineTransformType::ParametersType parameters;

  // Initialise the parameters

  center.Fill(0.);

  parameters.SetSize(12);

  parameters.Fill(0.);

  parameters[6] = 1.;		// Scale factor along the 'x' axis
  parameters[7] = 1.;		// Scale factor along the 'y' axis
  parameters[8] = 1.;		// Scale factor along the 'z' axis

  

    // Parse command line args
  for(int i=1; i < argc; i++){

    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }

    else if(strcmp(argv[i], "-cx") == 0) { 
      center[0] = atof(argv[++i]); 
      std::cout << "Set -cx=" << niftk::ConvertToString(center[0])<< std::endl;
    }
    else if(strcmp(argv[i], "-cy") == 0) { 
      center[1] = atof(argv[++i]); 
      std::cout << "Set -cy=" << niftk::ConvertToString(center[1])<< std::endl;
    }
    else if(strcmp(argv[i], "-cz") == 0) { 
      center[2] = atof(argv[++i]); 
      std::cout << "Set -cz=" << niftk::ConvertToString(center[2])<< std::endl;
    }

    else if(strcmp(argv[i], "-tx") == 0) { 
      parameters[0] = atof(argv[++i]); 
      std::cout << "Set -tx=" << niftk::ConvertToString(parameters[0])<< std::endl;
    }
    else if(strcmp(argv[i], "-ty") == 0) { 
      parameters[1] = atof(argv[++i]); 
      std::cout << "Set -ty=" << niftk::ConvertToString(parameters[1])<< std::endl;
    }
    else if(strcmp(argv[i], "-tz") == 0) { 
      parameters[2] = atof(argv[++i]); 
      std::cout << "Set -tz=" << niftk::ConvertToString(parameters[2])<< std::endl;
    }

    else if(strcmp(argv[i], "-rx") == 0) { 
      parameters[3] = atof(argv[++i]); 
      std::cout << "Set -rx=" << niftk::ConvertToString(parameters[3])<< std::endl;
    }
    else if(strcmp(argv[i], "-ry") == 0) { 
      parameters[4] = atof(argv[++i]); 
      std::cout << "Set -ry=" << niftk::ConvertToString(parameters[4])<< std::endl;
    }
    else if(strcmp(argv[i], "-rz") == 0) { 
      parameters[5] = atof(argv[++i]); 
      std::cout << "Set -rz=" << niftk::ConvertToString(parameters[5])<< std::endl;
    }

    else if(strcmp(argv[i], "-sx") == 0) { 
      parameters[6] = atof(argv[++i]); 
      std::cout << "Set -sx=" << niftk::ConvertToString(parameters[6])<< std::endl;
    }
    else if(strcmp(argv[i], "-sy") == 0) { 
      parameters[7] = atof(argv[++i]); 
      std::cout << "Set -sy=" << niftk::ConvertToString(parameters[7])<< std::endl;
    }
    else if(strcmp(argv[i], "-sz") == 0) { 
      parameters[8] = atof(argv[++i]); 
      std::cout << "Set -sz=" << niftk::ConvertToString(parameters[8])<< std::endl;
    }

    else if(strcmp(argv[i], "-k1") == 0) { 
      parameters[ 9] = atof(argv[++i]); 
      std::cout << "Set -k1=" << niftk::ConvertToString(parameters[ 9])<< std::endl;
    }
    else if(strcmp(argv[i], "-k2") == 0) { 
      parameters[10] = atof(argv[++i]); 
      std::cout << "Set -k2=" << niftk::ConvertToString(parameters[10])<< std::endl;
    }
    else if(strcmp(argv[i], "-k3") == 0) { 
      parameters[11] = atof(argv[++i]); 
      std::cout << "Set -k3=" << niftk::ConvertToString(parameters[11])<< std::endl;
    }

    else if(strcmp(argv[i], "-ot") == 0) {
      fileOutputTransformation = argv[++i];
      std::cout << "Set -ot=" << fileOutputTransformation<< std::endl;
    }

    else if(strcmp(argv[i], "-om") == 0) {
      fileOutputMatrix = argv[++i];
      std::cout << "Set -om=" << fileOutputMatrix<< std::endl;
    }
    else if(strcmp(argv[i], "-omp") == 0) {
      fileOutputPlain = argv[++i];
      std::cout << "Set -omp=" << fileOutputPlain<< std::endl;
    }

    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args

  if ((fileOutputTransformation.length() == 0) && (fileOutputMatrix.length() == 0) && (fileOutputPlain.length() == 0)) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }


  // Create the affine transformation
  
  typedef itk::EulerAffineTransform<double, 3, 3> EulerAffineTransformType;
  EulerAffineTransformType::Pointer transform = EulerAffineTransformType::New();

  std::cout << "Origin = " << center << std::endl;
  std::cout << "Parameters = " << parameters << std::endl;

  transform->SetCenter(center);
  transform->SetParameters(parameters);


  // Save the transform (as 12 parameter UCLEulerAffine transform).

  if (fileOutputTransformation.length() > 0) {

    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

    transformFileWriter->SetInput(transform);
    transformFileWriter->SetFileName(fileOutputTransformation); 

    transformFileWriter->Update();         
  }


  // Save the transform (as 16 parameter matrix transform).

  if (fileOutputMatrix.length() > 0) {

    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

    transformFileWriter->SetInput(transform->GetFullAffineTransform());
    transformFileWriter->SetFileName(fileOutputMatrix); 

    transformFileWriter->Update(); 
  }

  if (fileOutputPlain.length() > 0)
    {
	  transform->SaveFullAffineMatrix(fileOutputPlain);
    }
  
  std::cout << "Done"<< std::endl;
  
  return EXIT_SUCCESS;   
}


