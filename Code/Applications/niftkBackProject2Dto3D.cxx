/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkBackwardImageProjector2Dto3D.h"
#include "itkNIFTKTransformIOFactory.h"

#include "itkLogHelper.h"

/*!
 * \file niftkBackProject2Dto3D.cxx
 * \page niftkBackProject2Dto3D
 * \section niftkBackProject2Dto3DSummary Back projects a 2D image into a 3D volume.
 *
 *
 * \li Dimensions: Input must be 2D, output is 3D.
 * \li Pixel type: Scalars only, of type float.
 *
 * \section niftkBackProject2Dto3DCaveat Caveats
 */

void Usage(char *exec)
  {
	niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl
	      << "  Back projects a 2D image into a 3D volume" << std::endl << std::endl

	      << "  " << exec 
	      << " -im2D Input2Dimage -g AffineTransform -o Output3Dimage "
	      << "-f FocalLength -u0 Origin2DinX -v0 Origin2DinY [options]" << std::endl
	      << "  " << std::endl
	      << "*** [mandatory] ***" << std::endl << std::endl
	      << "    -im2D <filename>        Input 2D image" << std::endl
	      << "    -g    <filename>        Affine transformation " << std::endl
	      << "    -o    <filename>        Output 3D back-projected image" << std::endl << std::endl
	      << "*** [options]   ***" << std::endl << std::endl
	      << "    -v                             Output verbose info" << endl
	      << "    -dbg                           Output debugging info" << endl << endl
	      << "    -im3D <filename>               Input 3D image" << std::endl
	      << "    -s3D   <int> <int> <int>        The size of the 3D back-projection image [100 x 100 x 100]" << std::endl
	      << "    -r3D  <float> <float> <float>  The resolution of the 3D back-projection image [1mm x 1mm x 1mm]" << std::endl
	      << "    -o3D  <float> <float> <float>  The origin of the 3D back-projection image [0mm x 0mm x 0mm]" << std::endl << std::endl
	      << "    -f    <float>                  Focal length of the projection in mm [1000]" << std::endl
	      << "    -u0   <float>                  The location of the projection normal on the 2D plane in x (mm) [0]" << std::endl
	      << "    -v0   <float>                  The location of the projection normal on the 2D plane in y (mm) [0]" << std::endl
	      << "    -st                            Perform single threaded execution [multi-threaded]" << std::endl << std::endl;
  }


/**
 * \brief Project a 3D image volume into 2D.
 */
int main(int argc, char** argv)
{
  typedef float IntensityType;
  typedef itk::BackwardImageProjector2Dto3D< IntensityType > BackwardProjectorType;

  typedef BackwardProjectorType::OutputImageType ImageType3D; 
  typedef BackwardProjectorType::InputImageType ImageType2D;  
  typedef BackwardProjectorType::EulerAffineTransformType EulerAffineTransformType;
  typedef BackwardProjectorType::PerspectiveProjectionTransformType PerspectiveProjectionTransformType;

  typedef itk::ImageFileReader< ImageType3D >  InputImageReaderType3D;
  typedef itk::ImageFileReader< ImageType2D >  InputImageReaderType2D;

  typedef itk::ImageFileWriter< ImageType3D > OutputImageWriterType;
  
  itk::ObjectFactoryBase::RegisterFactory(itk::NIFTKTransformIOFactory::New());


  bool flgDebug = false;

  bool flgSingleThreadedExecution = false; // Perform single threaded execution

  std::string fileInputImage3D;
  std::string fileInputImage2D;
  std::string fileOutputImage3D;
  std::string fileAffineTransform3D;

  bool flgInputImage3D_SizeSet = false;	// Has the user specified the 3D image size?
  bool flgInputImage3D_ResSet = false;	// Has the user specified the 3D image resolution?

  double focalLength = 1000.;	 // The focal length of the 3D to 2D projection
  double u0 = 0.;		 // The origin in x of the 2D projection image
  double v0 = 0.;		 // The origin in y of the 2D projection image

  // The dimensions in pixels of the 3D image
  BackwardProjectorType::OutputImageSizeType nVoxels3D;
  // The resolution in mm of the 3D image
  BackwardProjectorType::OutputImageSpacingType spacing3D;
  // The origin in mm of the 3D image
  BackwardProjectorType::OutputImagePointType origin3D;

  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  nVoxels3D[0] = 100;
  nVoxels3D[1] = 100;
  nVoxels3D[2] = 100;

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
    else if(strcmp(argv[i], "-v") == 0) {
      cout << "Verbose output enabled" << endl;
    }
    else if(strcmp(argv[i], "-dbg") == 0) {
      flgDebug = true;
      cout << "Debugging output enabled" << endl;
    }
    else if(strcmp(argv[i], "-im3D") == 0) {
      fileInputImage3D = argv[++i];
      std::cout << "Set -im3D=" << fileInputImage3D<< std::endl;
    }
    else if(strcmp(argv[i], "-im2D") == 0) {
      fileInputImage2D = argv[++i];
      std::cout << "Set -im2D=" << fileInputImage2D<< std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0) {
      fileOutputImage3D = argv[++i];
      std::cout << "Set -o=" << fileOutputImage3D<< std::endl;
    }
    else if(strcmp(argv[i], "-g") == 0) {
      fileAffineTransform3D = argv[++i];
      std::cout << "Set -g=" << fileAffineTransform3D<< std::endl;
    }
    else if(strcmp(argv[i], "-s3D") == 0) {
      nVoxels3D[0] = atoi(argv[++i]);
      nVoxels3D[1] = atoi(argv[++i]);
      nVoxels3D[2] = atoi(argv[++i]);
      std::cout << "Set -s3D="
				    << niftk::ConvertToString((int) nVoxels3D[0]) << " "
		    		<< niftk::ConvertToString((int) nVoxels3D[1]) << " "
		    		<< niftk::ConvertToString((int) nVoxels3D[2])<< std::endl;
      flgInputImage3D_SizeSet = true;
    }
    else if(strcmp(argv[i], "-r3D") == 0) {
      spacing3D[0] = atof(argv[++i]);
      spacing3D[1] = atof(argv[++i]);
      spacing3D[2] = atof(argv[++i]);
      std::cout << "Set -r3D="
				    << niftk::ConvertToString(spacing3D[0]) << " "
				    << niftk::ConvertToString(spacing3D[1]) << " "
				    << niftk::ConvertToString(spacing3D[2])<< std::endl;
      flgInputImage3D_ResSet = true;
    }
    else if(strcmp(argv[i], "-o3D") == 0) {
      origin3D[0] = atof(argv[++i]);
      origin3D[1] = atof(argv[++i]);
      origin3D[2] = atof(argv[++i]);
      std::cout << "Set -o3D="
				    << niftk::ConvertToString(origin3D[0]) << " "
				    << niftk::ConvertToString(origin3D[1]) << " "
				    << niftk::ConvertToString(origin3D[2])<< std::endl;
    }
    else if(strcmp(argv[i], "-f") == 0) {
      focalLength = atof(argv[++i]);
      std::cout << "Set -f=" << niftk::ConvertToString(focalLength)<< std::endl;
    }
    else if(strcmp(argv[i], "-u0") == 0) {
      u0 = atof(argv[++i]);
      std::cout << "Set -u0=" << niftk::ConvertToString(u0)<< std::endl;
    }
    else if(strcmp(argv[i], "-v0") == 0) {
      v0 = atof(argv[++i]);
      std::cout << "Set -v0=" << niftk::ConvertToString(v0)<< std::endl;
    }
    else if(strcmp(argv[i], "-st") == 0) {
      flgSingleThreadedExecution = true;
      std::cout << "Set -st"<< std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputImage2D.length() == 0 || fileOutputImage3D.length() == 0 || fileAffineTransform3D.length() == 0 ) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  if ( fileInputImage3D.length() != 0 && ((flgInputImage3D_SizeSet == true) || (flgInputImage3D_ResSet == true)) ) {
    std::cerr << "ERROR: Command line options '-im3D' and '-s3D' or '-r3D' are exclusive." << std::endl;
    Usage(argv[0]);
    return EXIT_FAILURE;
  }
      

  // Create the backward projector
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  BackwardProjectorType::Pointer backwardProjector = BackwardProjectorType::New();

  if (flgSingleThreadedExecution)
    backwardProjector->SetSingleThreadedExecution();


  // Load the input 3D image or create it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputImage3D.length() != 0 ) {

    InputImageReaderType3D::Pointer inputImageReader3D  = InputImageReaderType3D::New();
  
    inputImageReader3D->SetFileName( fileInputImage3D );

    try { 
      std::cout << "Reading input 3D volume: " <<  fileInputImage3D<< std::endl;
      inputImageReader3D->Update();
      std::cout << "Done"<< std::endl;
    } 
    catch( itk::ExceptionObject & err ) { 
      std::cerr << "Failed to load input image: " << err << std::endl; 
      return EXIT_FAILURE;
    }         
  }

  else {

    backwardProjector->SetBackProjectedImageSize( nVoxels3D );
    backwardProjector->SetBackProjectedImageSpacing( spacing3D );
    backwardProjector->SetBackProjectedImageOrigin( origin3D );

  }

  
  // Load the 2D input image
  // ~~~~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType2D::Pointer inputImageReader2D  = InputImageReaderType2D::New();

  inputImageReader2D->SetFileName( fileInputImage2D );

  try { 
    std::cout << "Reading input 2D image: " <<  fileInputImage2D<< std::endl;
    inputImageReader2D->Update();
    std::cout << "Done"<< std::endl;
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  backwardProjector->SetInput( inputImageReader2D->GetOutput() );



  // Load the affine transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  BackwardProjectorType::EulerAffineTransformType::Pointer affineTransform;

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
    std::cerr << "Failed to load 3D affine transform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }


  typedef itk::TransformFileReader::TransformListType * TransformListType;
  TransformListType transforms = transformFileReader->GetTransformList();
  std::cout << "Number of transforms = " << transforms->size() << std::endl;

  itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();

  if (! strcmp((*it)->GetNameOfClass(),"EulerAffineTransform")) 
    affineTransform = static_cast<EulerAffineTransformType*>((*it).GetPointer());

  else {
    std::cerr << "Failed to cast transform top affine" << std::endl;
    return EXIT_FAILURE;    
  }

  affineTransform->Print(std::cout);

  backwardProjector->SetAffineTransform(affineTransform);


  // Create the perspective projection transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PerspectiveProjectionTransformType::Pointer perspectiveTransform;

  perspectiveTransform = PerspectiveProjectionTransformType::New();

  perspectiveTransform->SetFocalDistance(focalLength);
  perspectiveTransform->SetOriginIn2D(u0, v0);

  backwardProjector->SetPerspectiveTransform(perspectiveTransform);


  // Perform the back projection
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  backwardProjector->Update();


  // Write the output projected image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputImage3D );
  writer->SetInput( backwardProjector->GetOutput() );

  std::cout << "Backprojector output: " << backwardProjector->GetOutput() << std::endl;

  try { 
    std::cout << "Writing output to file: " << fileOutputImage3D<< std::endl;
    writer->Update();
    std::cout << "Done"<< std::endl;
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to write output to file: " << err << std::endl; 
    return EXIT_FAILURE;
  }         


  std::cout << "Done"<< std::endl;
  
  return EXIT_SUCCESS;   
}


