/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author: mjc $

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
#include "itkImageProjector2D3D.h"
#include "itkPerspectiveProjectionTransform.h"


void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl
	      << "  Projects a 3D image volume into 2D" << std::endl << std::endl

	      << "  " << exec 
	      << " -im Input3Dimage -g AffineTransform -o Output2Dimage "
	      << "-f FocalLength -u0 Origin2DinX -v0 Origin2DinY [options]" << std::endl << "  " << std::endl

	      << "*** [mandatory] ***" << std::endl << std::endl
	      << "    -im   <filename>        Input 3D image volume " << std::endl
	      << "    -g    <filename>        Affine transformation " << std::endl
	      << "    -o    <filename>        Output 2D projection image" << std::endl << std::endl

	      << "*** [options]   ***" << std::endl << std::endl
	      << "    -sz   <int> <int>       The size of the 2D projection image [100 x 100]" << std::endl
	      << "    -res  <float> <float>   The resolution of the 2D projection image [1mm x 1mm]" << std::endl
	      << "    -o2D  <float> <float>   The origin of the 2D projection image [0mm x 0mm]" << std::endl << std::endl
	      << "    -p    <filename>        Perspective projection transformation (or specify: '-f', '-u0' and 'v0')." << std::endl
	      << "    -f    <float>           Focal length of the projection in mm [1000]" << std::endl
	      << "    -u0   <float>           The location of the projection normal on the 2D plane in x (mm) [0]" << std::endl
	      << "    -v0   <float>           The location of the projection normal on the 2D plane in y (mm) [0]" << std::endl
	      << "    -t    <float>           Threshold above which to integrate 3D voxel intensities [0]" << std::endl << std::endl
	      << "    -st                     Perform single threaded execution [multi-threaded]" << std::endl << std::endl;
  }


/**
 * \brief Project a 3D image volume into 2D.
 */
int main(int argc, char** argv)
{
  bool flgSingleThreadedExecution = false; // Perform single threaded execution

  std::string fileInputImage3D;
  std::string fileOutputImage2D;
  std::string fileAffineTransform3D;
  std::string filePerspectiveTransform;

  double focalLength = 1000.;	// The focal length of the 3D to 2D projection
  double u0 = 0.;		// The origin in x of the 2D projection image
  double v0 = 0.;		// The origin in y of the 2D projection image
  double threshold = 0.;	// The ray integration threshold

  typedef float IntensityType;

  typedef itk::ImageProjector2D3D< IntensityType > ForwardProjectorType;

  // The dimensions in pixels of the 2D image
  ForwardProjectorType::OutputImageSizeType nPixels2D;
  // The resolution in mm of the 2D image
  ForwardProjectorType::OutputImageSpacingType spacing2D;
  // The origin in mm of the 2D image
  ForwardProjectorType::OutputImagePointType origin2D;

  typedef ForwardProjectorType::InputImageType InputImageType; 
  typedef ForwardProjectorType::OutputImageType OutputImageType;
  typedef ForwardProjectorType::EulerAffineTransformType EulerAffineTransformType;
  typedef ForwardProjectorType::PerspectiveProjectionTransformType PerspectiveProjectionTransformType;

  typedef itk::ImageFileReader< InputImageType >  InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  
  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  nPixels2D[0] = 100;
  nPixels2D[1] = 100;

  spacing2D[0] = 1.;
  spacing2D[1] = 1.;

  origin2D[0] = 0.;
  origin2D[1] = 0.;

  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-im") == 0) {
      fileInputImage3D = argv[++i];
      std::cout << "Set -im=" + fileInputImage3D;
    }
    else if(strcmp(argv[i], "-o") == 0) {
      fileOutputImage2D = argv[++i];
      std::cout << "Set -o=" + fileOutputImage2D;
    }
    else if(strcmp(argv[i], "-g") == 0) {
      fileAffineTransform3D = argv[++i];
      std::cout << "Set -g=" + fileAffineTransform3D;
    }
    else if(strcmp(argv[i], "-sz") == 0) {
      nPixels2D[0] = atoi(argv[++i]);
      nPixels2D[1] = atoi(argv[++i]);
      std::cout << "Set -sz="
				    << niftk::ConvertToString((int) nPixels2D[0]) << " "
				    << niftk::ConvertToString((int) nPixels2D[1]);
    }
    else if(strcmp(argv[i], "-res") == 0) {
      spacing2D[0] = atof(argv[++i]);
      spacing2D[1] = atof(argv[++i]);
      std::cout << "Set -res="
				    << niftk::ConvertToString(spacing2D[0]) << " "
				    << niftk::ConvertToString(spacing2D[1]);
    }
    else if(strcmp(argv[i], "-o2D") == 0) {
      origin2D[0] = atof(argv[++i]);
      origin2D[1] = atof(argv[++i]);
      std::cout << "Set -o2D="
				    << niftk::ConvertToString(origin2D[0]) << " "
				    << niftk::ConvertToString(origin2D[1]);
    }
    else if(strcmp(argv[i], "-p") == 0) {
      filePerspectiveTransform = argv[++i];
      std::cout << "Set -p=" << filePerspectiveTransform;
    }
    else if(strcmp(argv[i], "-f") == 0) {
      focalLength = atof(argv[++i]);
      std::cout << "Set -f=" << niftk::ConvertToString(focalLength);
    }
    else if(strcmp(argv[i], "-u0") == 0) {
      u0 = atof(argv[++i]);
      std::cout << "Set -u0=" << niftk::ConvertToString(u0);
    }
    else if(strcmp(argv[i], "-v0") == 0) {
      v0 = atof(argv[++i]);
      std::cout << "Set -v0=" << niftk::ConvertToString(v0);
    }
    else if(strcmp(argv[i], "-t") == 0) {
      threshold = atof(argv[++i]);
      std::cout << "Set -t=" << niftk::ConvertToString(threshold);
    }
    else if(strcmp(argv[i], "-st") == 0) {
      flgSingleThreadedExecution = true;
      std::cout << "Set -st";
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileInputImage3D.length() == 0 || fileOutputImage2D.length() == 0 || fileAffineTransform3D.length() == 0) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }


  // Load the input image volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType::Pointer inputImageReader  = InputImageReaderType::New();
  
  inputImageReader->SetFileName( fileInputImage3D );

  try { 
    std::cout << "Reading input 3D volume: " <<  fileInputImage3D;
    inputImageReader->Update();
    std::cout << "Done";
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  

  // Create the forward projector
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  ForwardProjectorType::Pointer forwardProjector = ForwardProjectorType::New();

  forwardProjector->SetInput( inputImageReader->GetOutput() );
  forwardProjector->SetProjectedImageSize(nPixels2D);
  forwardProjector->SetProjectedImageSpacing(spacing2D);
  forwardProjector->SetProjectedImageOrigin(origin2D);
  forwardProjector->SetRayIntegrationThreshold(threshold);

  if (flgSingleThreadedExecution)
    forwardProjector->SetSingleThreadedExecution();


  // Load the affine transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  ForwardProjectorType::EulerAffineTransformType::Pointer affineTransform;

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

  forwardProjector->SetAffineTransform(affineTransform);


  // Load or create the perspective projection transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PerspectiveProjectionTransformType::Pointer perspectiveTransform;

  if (filePerspectiveTransform.length() != 0) {

    itk::TransformFactory<PerspectiveProjectionTransformType>::RegisterTransform();

    itk::TransformFileReader::Pointer transformFileReader;
    transformFileReader = itk::TransformFileReader::New();
    transformFileReader->SetFileName(filePerspectiveTransform);

    try {
      std::cout << "Reading 3D perspective transform from:" << filePerspectiveTransform << std::endl; 
      transformFileReader->Update();
      std::cout << "Done" << std::endl; 
    }  
    catch (itk::ExceptionObject& exceptionObject) {
      std::cerr << "ERROR: Failed to load 3D perspective transform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }

    typedef itk::TransformFileReader::TransformListType *TransformListType;
    TransformListType transforms = transformFileReader->GetTransformList();

    std::cout << "Number of transforms = " << transforms->size() << std::endl;

    itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();

    if (! strcmp((*it)->GetNameOfClass(),"PerspectiveProjectionTransform")) 
      perspectiveTransform = static_cast<PerspectiveProjectionTransformType*>((*it).GetPointer());

    else {
      std::cerr << "ERROR: Failed to cast transform to perspective" << std::endl;
      return EXIT_FAILURE;    
    }
  }

  else {
    perspectiveTransform = PerspectiveProjectionTransformType::New();

    perspectiveTransform->SetFocalDistance(focalLength);
    perspectiveTransform->SetOriginIn2D(u0, v0);
  }


  perspectiveTransform->Print(std::cout);

  forwardProjector->SetPerspectiveTransform(perspectiveTransform);


  // Perform the forwards projection
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  forwardProjector->Update();


  // Write the output projected image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputImage2D );
  writer->SetInput( forwardProjector->GetOutput() );

  try { 
    std::cout << "Writing output to file: " << fileOutputImage2D;
    writer->Update();
    std::cout << "Done";
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to write output to file: " << err << std::endl; 
    return EXIT_FAILURE;
  }         


  std::cout << "Done";
  
  return EXIT_SUCCESS;   
}


