/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-06-04 15:11:19 +0100 (Fri, 04 Jun 2010) $
 Revision          : $Revision: 3349 $
 Last modified by  : $Author: ma $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkEulerAffineTransform.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkCreateEulerAffineTransformMatrix.h"


void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl
	      << "  Create a affine transformation matrix of a 3D image volume" << std::endl << std::endl

	      << "  " << exec 
	      << " -im Input3Dimage -g AffineTransform -o Output3Dimage "
	      << "-f FocalLength -u0 Origin3DinX -v0 Origin3DinY [options]" << std::endl << "  " << std::endl

	      << "*** [mandatory] ***" << std::endl << std::endl
	      << "    -im   <filename>        Input 3D image volume " << std::endl
	      << "    -g    <filename>        Affine transformation " << std::endl
	      << "    -o    <filename>        Output 3D affine transformed image" << std::endl << std::endl

	      << "*** [options]   ***" << std::endl << std::endl
	      << "    -sz   <int> <int> <int>      		The size of the 3D projection image [100 x 100]" << std::endl
	      << "    -res  <float> <float> <float>   The resolution of the 3D projection image [1mm x 1mm]" << std::endl
	      << "    -o3D  <float> <float> <float>  	The origin of the 3D projection image [0mm x 0mm]" << std::endl << std::endl
	      << "    -st                     				Perform single threaded execution [multi-threaded]" << std::endl << std::endl;
  }


/**
 * \brief Project a 3D image volume into 3D.
 */
int main(int argc, char** argv)
{
  bool flgSingleThreadedExecution = false; // Perform single threaded execution

  std::string fileInputImage3D;
  std::string fileOutputImage3D;
  std::string fileAffineTransform3D;

  typedef float IntensityType;

  typedef itk::CreateEulerAffineTransformMatrix< IntensityType > AffineTransformerType;

  // The dimensions in pixels of the 3D image
  AffineTransformerType::OutputImageSizeType nPixels3D;
  // The resolution in mm of the 3D image
  AffineTransformerType::OutputImageSpacingType spacing3D;
  // The origin in mm of the 3D image
  AffineTransformerType::OutputImagePointType origin3D;

  typedef AffineTransformerType::InputImageType InputImageType; 
  typedef AffineTransformerType::OutputImageType OutputImageType;
  typedef AffineTransformerType::EulerAffineTransformType EulerAffineTransformType;

  typedef itk::ImageFileReader< InputImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  
  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  nPixels3D[0] = 100;
  nPixels3D[1] = 100;
  nPixels3D[2] = 100;

  spacing3D[0] = 1.;
  spacing3D[1] = 1.;
  spacing3D[2] = 1.;

  origin3D[0] = 0.;
  origin3D[1] = 0.;
  origin3D[2] = 0.;

  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-im") == 0) {
      fileInputImage3D = argv[++i];
      std::cout << std::string("Set -im=") << fileInputImage3D;
    }
    else if(strcmp(argv[i], "-o") == 0) {
      fileOutputImage3D = argv[++i];
      std::cout << std::string("Set -o=") << fileOutputImage3D;
    }
    else if(strcmp(argv[i], "-g") == 0) {
      fileAffineTransform3D = argv[++i];
      std::cout << std::string("Set -g=") << fileAffineTransform3D;
    }
    else if(strcmp(argv[i], "-sz") == 0) {
      nPixels3D[0] = atoi(argv[++i]);
      nPixels3D[1] = atoi(argv[++i]);
			nPixels3D[2] = atoi(argv[++i]);
      std::cout << std::string("Set -sz=")
				    << niftk::ConvertToString((int) nPixels3D[0]) << " "
				    << niftk::ConvertToString((int) nPixels3D[1]) << " "
				    << niftk::ConvertToString((int) nPixels3D[2]);
    }
    else if(strcmp(argv[i], "-res") == 0) {
      spacing3D[0] = atof(argv[++i]);
      spacing3D[1] = atof(argv[++i]);
			spacing3D[2] = atof(argv[++i]);
      std::cout << std::string("Set -res=")
      << niftk::ConvertToString(spacing3D[0]) << " "
      << niftk::ConvertToString(spacing3D[1]) << " "
      << niftk::ConvertToString(spacing3D[2]);
    }
    else if(strcmp(argv[i], "-o3D") == 0) {
      origin3D[0] = atof(argv[++i]);
      origin3D[1] = atof(argv[++i]);
			origin3D[2] = atof(argv[++i]);
      std::cout << std::string("Set -o3D=")
      << niftk::ConvertToString(origin3D[0]) << " "
      << niftk::ConvertToString(origin3D[1]) << " "
      << niftk::ConvertToString(origin3D[2]);
    }
    else if(strcmp(argv[i], "-st") == 0) {
      flgSingleThreadedExecution = true;
      std::cout << std::string("Set -st");
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileInputImage3D.length() == 0 || fileOutputImage3D.length() == 0 || fileAffineTransform3D.length() == 0) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }


  // Load the input image volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType::Pointer inputImageReader  = InputImageReaderType::New();
  
  inputImageReader->SetFileName( fileInputImage3D );

  try { 
    std::cout << std::string("Reading input 3D volume: ") <<  fileInputImage3D;
    inputImageReader->Update();
    std::cout << std::string("Done");
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  

  // Create the forward projector
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  AffineTransformerType::Pointer affineTransformer = AffineTransformerType::New();

  affineTransformer->SetInput( inputImageReader->GetOutput() );
  affineTransformer->SetTransformedImageSize(nPixels3D);
  affineTransformer->SetTransformedImageSpacing(spacing3D);
  affineTransformer->SetTransformedImageOrigin(origin3D);

  if (flgSingleThreadedExecution)
    affineTransformer->SetSingleThreadedExecution();


  // Load the affine transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  AffineTransformerType::EulerAffineTransformType::Pointer affineTransform;

  itk::TransformFactory<EulerAffineTransformType>::RegisterTransform();

  itk::TransformFileReader::Pointer transformFileReader;
  transformFileReader = itk::TransformFileReader::New();
  transformFileReader->SetFileName(fileAffineTransform3D);

  try {
    std::cout << "Reading 3D affine transform from:" << fileAffineTransform3D << std::endl; 
    transformFileReader->Update();
    std::cout << "Done" << std::endl; 
  }  
  catch (itk::ExceptionObject& exceptionObject) {
    std::cerr << "ERROR: Failed to load 3D affine transform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }


  typedef itk::TransformFileReader::TransformListType *TransformListType;
  TransformListType transforms = transformFileReader->GetTransformList();

  std::cout << "Number of transforms = " << transforms->size() << std::endl;

  itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();

  if (! strcmp((*it)->GetNameOfClass(),"EulerAffineTransform")) 
    affineTransform = static_cast<EulerAffineTransformType*>((*it).GetPointer());

  else {
    std::cerr << "ERROR: Failed to cast transform to affine" << std::endl;
    return EXIT_FAILURE;    
  }

  affineTransform->Print(std::cout);

  affineTransformer->SetAffineTransform(affineTransform);


  // Perform the affine transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  affineTransformer->Update();


  // Write the output projected image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputImage3D );
  writer->SetInput( affineTransformer->GetOutput() );

  try { 
    std::cout << std::string("Writing output to file: ") << fileOutputImage3D;
    writer->Update();
    std::cout << std::string("Done");
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to write output to file: " << err << std::endl; 
    return EXIT_FAILURE;
  }         


  std::cout << std::string("Done");
  
  return EXIT_SUCCESS;   
}

