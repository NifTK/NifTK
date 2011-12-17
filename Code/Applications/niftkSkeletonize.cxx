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

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkScalarImageToHistogramGenerator.h"
#include "itkRelabelComponentImageFilter.h"

// Standard ITK 2D binary thinning
#include "itkBinaryThinningImageFilter.h"

// H. Homman's 3D binary thinning
#include "itkBinaryThinningImageFilter3D.h"

// Just for chamfer distance
#include "itkSkeletonizeImageFilter.h"
#include "itkChamferDistanceTransformImageFilter.h"

// NifTK Pudney implementation
#include "itkSkeletonizeBinaryImageFilter.h"

void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements binary thinning using either:" << std::endl;
  std::cout << "    2D - ITK's standard itkBinaryThinningFilter, implementing algorithm in Gonzales and Woods, Digital Image Processing, Addison Wesley, 491-494, (1993)." << std::endl;
  std::cout << "    3D - H. Homman's ITK Journal implementation of Lee et. al. CVGIP Vol. 56, No. 6, November, pp 462-478, 1994, available http://hdl.handle.net/1926/1292" << std::endl;
  std::cout << "    3D - Distance Ordered Homotopic Thinning (DOHT)', Pudney (1998), Computer Vision And Image Understanding, Vol 72, No. 3, pp. 404-413" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "  NOTE: The DOHT implementation uses the itkChamferDistanceTransformImageFilter.h from http://hdl.handle.net/1926/304" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i <filename>        Input binary image [0-1] " << std::endl;
  std::cout << "    -o <filename>        Output skeletonized binary image [0-1]" << std::endl << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;
  std::cout << "    -b <int>   [0-255]   Background value." << std::endl;
  std::cout << "    -f <int>   [1-255]   Foreground value." << std::endl;
  std::cout << "    -connectedOnInput    Perform connected component analysis on input, and pick largest connected component." << std::endl;  
  std::cout << "    -connectedOnOutput   Perform connected component analysis on output, and pick largest connected component." << std::endl;  
  std::cout << "    -doht                Use DOHT" << std::endl;
  std::cout << "    -t <float> [6]       For DOHT, Tau, distance threshold for preservation of non-witness points." << std::endl;
  std::cout << "    -noMedialAxis        For DOHT, don't do medial axis check" << std::endl;
  std::cout << "    -noMedialSurface     For DOHT, don't do medial surface check" << std::endl;
  std::cout << "    -noMaximalBall       For DOHT, don't do maximal ball check" << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  int backgroundValue;
  int foregroundValue;
  double tau;
  bool useDoht;
  bool doConnectedOnInput;
  bool doConnectedOnOutput;
  bool doMedialAxisCheck;
  bool doMedialSurfaceCheck;
  bool doMaximalBallCheck;
};

typedef short InputPixelType;
typedef short OutputPixelType;

/**
 * \brief Implements 3 different skeletonization algorithms.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.backgroundValue = 0;
  args.foregroundValue = 1;
  args.useDoht = false;
  args.doConnectedOnInput = false;
  args.doConnectedOnOutput = false;
  args.doMedialAxisCheck = true;
  args.doMedialSurfaceCheck = true;
  args.doMaximalBallCheck = true;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -i=" << args.inputImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -o=" << args.outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-b") == 0){
      args.backgroundValue=atoi(argv[++i]);
      std::cout << "Set -b=" << niftk::ConvertToString(args.backgroundValue) << std::endl;
    }
    else if(strcmp(argv[i], "-f") == 0){
      args.foregroundValue=atoi(argv[++i]);
      std::cout << "Set -f=" << niftk::ConvertToString(args.foregroundValue) << std::endl;
    }
    else if(strcmp(argv[i], "-t") == 0){
      args.tau=atoi(argv[++i]);
      std::cout << "Set -t=" << niftk::ConvertToString(args.tau) << std::endl;
    }
    else if(strcmp(argv[i], "-doht") == 0){
      args.useDoht=true;
      std::cout << "Set -doht=" << niftk::ConvertToString(args.useDoht) << std::endl;
    }
    else if(strcmp(argv[i], "-connectedOnInput") == 0){
      args.doConnectedOnInput=true;
      std::cout << "Set -connectedOnInput=" << niftk::ConvertToString(args.doConnectedOnInput) << std::endl;
    }
    else if(strcmp(argv[i], "-connectedOnOutput") == 0){
      args.doConnectedOnOutput=true;
      std::cout << "Set -connectedOnOutput=" << niftk::ConvertToString(args.doConnectedOnOutput) << std::endl;
    }        
    else if(strcmp(argv[i], "-noMedialAxis") == 0){
      args.doMedialAxisCheck = false;
      std::cout << "Set -noMedialAxis=" << niftk::ConvertToString(args.doMedialAxisCheck) << std::endl;
    }    
    else if(strcmp(argv[i], "-noMedialSurface") == 0){
      args.doMedialSurfaceCheck = false;
      std::cout << "Set -noMedialSurface=" << niftk::ConvertToString(args.doMedialSurfaceCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-noMaximalBall") == 0){
      args.doMaximalBallCheck = false;
      std::cout << "Set -noMaximalBall=" << niftk::ConvertToString(args.doMaximalBallCheck) << std::endl;
    }        
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }
  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if (args.backgroundValue == args.foregroundValue)
    {
      std::cout << "The background and foreground value must be different" << std::endl;
      return EXIT_FAILURE;
    }

  if (args.backgroundValue > args.foregroundValue)
    {
	    std::cout << "The background value must be less than the foreground value" << std::endl;
	    return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage);

  // So for 2D we just run the standard itkBinaryThinningImageFilter that comes with ITK
  if (dims == 2)
    {
	    const unsigned int Dimension = 2;
	    typedef itk::Image< InputPixelType, Dimension >  InputImageType;
	    typedef itk::ImageFileReader< InputImageType  >   InputImageReaderType;
	    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

	    InputImageReaderType::Pointer  imageReader  = InputImageReaderType::New();
	    imageReader->SetFileName(  args.inputImage );

	    try
	      {
	        std::cout << "Loading 2D input image:" << args.inputImage << std::endl;
	        imageReader->Update();
	        std::cout << "Done" << std::endl;
	      }
	    catch( itk::ExceptionObject & err )
	      {
	        std::cerr << "ExceptionObject caught !";
	        std::cerr << err << std::endl;
	        return -2;
	      }

	    typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> ThresholdFilterType;
	    ThresholdFilterType::Pointer thresholdFilter = ThresholdFilterType::New();
      ThresholdFilterType::Pointer thresholdBackFilter = ThresholdFilterType::New();
      ThresholdFilterType::Pointer pickLargestInputComponentFilter = ThresholdFilterType::New();
      ThresholdFilterType::Pointer pickLargestOutputComponentFilter = ThresholdFilterType::New();

	    typedef itk::ConnectedComponentImageFilter<InputImageType, InputImageType> ConnectedFilterType;
	    ConnectedFilterType::Pointer connectedInputFilter = ConnectedFilterType::New();
	    ConnectedFilterType::Pointer connectedOutputFilter = ConnectedFilterType::New();
	  
	    typedef itk::RelabelComponentImageFilter<InputImageType, InputImageType> RelabelFilterType;
	    RelabelFilterType::Pointer relabelInputFilter = RelabelFilterType::New();
	    RelabelFilterType::Pointer relabelOutputFilter = RelabelFilterType::New();
	  
      typedef itk::BinaryThinningImageFilter<InputImageType, InputImageType> Thinning2DFilterType;
	    Thinning2DFilterType::Pointer thinning2DFilter = Thinning2DFilterType::New();

	    typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
      CastFilterType::Pointer castFilter = CastFilterType::New();

      typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
      OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

	    thresholdFilter->SetInput(imageReader->GetOutput());
	    thresholdFilter->SetInsideValue(1);
	    thresholdFilter->SetOutsideValue(0);
	    thresholdFilter->SetLowerThreshold(args.foregroundValue);
	    thresholdFilter->SetUpperThreshold(std::numeric_limits<InputPixelType>::max());
	    thresholdFilter->UpdateLargestPossibleRegion();
	  
	    if (args.doConnectedOnInput)
	      {
	        std::cout << "Doing connected components on input" << std::endl;
	      
	        connectedInputFilter->SetInput(thresholdFilter->GetOutput());
	        connectedInputFilter->SetFullyConnected(true);
	        connectedInputFilter->UpdateLargestPossibleRegion();

	        relabelInputFilter->SetInput(connectedInputFilter->GetOutput());
	        relabelInputFilter->UpdateLargestPossibleRegion();
	      
	        pickLargestInputComponentFilter->SetInput(relabelInputFilter->GetOutput());
	        pickLargestInputComponentFilter->SetInsideValue(1);
	        pickLargestInputComponentFilter->SetOutsideValue(0);
	        pickLargestInputComponentFilter->SetLowerThreshold(1);
	        pickLargestInputComponentFilter->SetUpperThreshold(1);
	        pickLargestInputComponentFilter->UpdateLargestPossibleRegion();
	      
	        thinning2DFilter->SetInput(pickLargestInputComponentFilter->GetOutput());
	      }
	    else
	      {
	        thinning2DFilter->SetInput(thresholdFilter->GetOutput());
	      }

	    // Run the thinning
	    thinning2DFilter->UpdateLargestPossibleRegion();
	  
  	  if (args.doConnectedOnOutput)
	      {
	        std::cout << "Doing connected components on output" << std::endl;
	      
          connectedOutputFilter->SetInput(thinning2DFilter->GetOutput());
          connectedOutputFilter->SetFullyConnected(true);
          connectedOutputFilter->UpdateLargestPossibleRegion();

          relabelOutputFilter->SetInput(connectedOutputFilter->GetOutput());
          relabelOutputFilter->UpdateLargestPossibleRegion();
        
          pickLargestOutputComponentFilter->SetInput(relabelOutputFilter->GetOutput());
          pickLargestOutputComponentFilter->SetInsideValue(1);
          pickLargestOutputComponentFilter->SetOutsideValue(0);
          pickLargestOutputComponentFilter->SetLowerThreshold(1);
          pickLargestOutputComponentFilter->SetUpperThreshold(1);
          pickLargestOutputComponentFilter->UpdateLargestPossibleRegion();
	      
          thresholdBackFilter->SetInput(pickLargestOutputComponentFilter->GetOutput());
        
	      }
	    else
	      {
	        thresholdBackFilter->SetInput(thinning2DFilter->GetOutput());
	      }
	    
      thresholdBackFilter->SetInsideValue(args.foregroundValue);
	    thresholdBackFilter->SetOutsideValue(args.backgroundValue);
	    thresholdBackFilter->SetLowerThreshold(1);
	    thresholdBackFilter->SetUpperThreshold(std::numeric_limits<InputPixelType>::max());

	    castFilter->SetInput(thresholdBackFilter->GetOutput());    
	    writer->SetInput(castFilter->GetOutput());
	    writer->SetFileName(args.outputImage);
	    writer->Update();

    }
  else if (dims == 3)
    {

	    const unsigned int Dimension = 3;
	    typedef itk::Image< InputPixelType, Dimension > InputImageType;
	    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
	    typedef itk::Image< float, Dimension > DistanceTransformImageType;

	    typedef itk::ImageFileReader< InputImageType  > InputImageReaderType;
	    InputImageReaderType::Pointer  imageReader  = InputImageReaderType::New();

	    imageReader->SetFileName(  args.inputImage );

      try
	      {
	        std::cout << "Loading 3D input image:" << args.inputImage << std::endl;
	        imageReader->Update();
	        std::cout << "Done" << std::endl;
	      }
	    catch( itk::ExceptionObject & err )
	      {
	        std::cerr <<"ExceptionObject caught !";
	        std::cerr << err << std::endl;
	        return -2;
	      }

	    typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> ThresholdFilterType;
	    ThresholdFilterType::Pointer thresholdFilter = ThresholdFilterType::New();
	    ThresholdFilterType::Pointer thresholdBackFilter = ThresholdFilterType::New();
      ThresholdFilterType::Pointer pickLargestInputComponentFilter = ThresholdFilterType::New();
      ThresholdFilterType::Pointer pickLargestOutputComponentFilter = ThresholdFilterType::New();

      typedef itk::ConnectedComponentImageFilter<InputImageType, InputImageType> ConnectedFilterType;
	    ConnectedFilterType::Pointer connectedInputFilter = ConnectedFilterType::New();
	    ConnectedFilterType::Pointer connectedOutputFilter = ConnectedFilterType::New();
	    
	    typedef itk::RelabelComponentImageFilter<InputImageType, InputImageType> RelabelFilterType;
	    RelabelFilterType::Pointer relabelInputFilter = RelabelFilterType::New();
	    RelabelFilterType::Pointer relabelOutputFilter = RelabelFilterType::New();
	    
	    typedef itk::SkeletonizeImageFilter<InputImageType, itk::Connectivity<Dimension, 0> > SkeletonizerFilterType;
	    typedef itk::ChamferDistanceTransformImageFilter<InputImageType, SkeletonizerFilterType::OrderingImageType> DistanceMapFilterType;
	    DistanceMapFilterType::Pointer chamferFilter = DistanceMapFilterType::New();

	    typedef itk::CastImageFilter<SkeletonizerFilterType::OrderingImageType, DistanceTransformImageType> CastToFloatFilterType;
	    CastToFloatFilterType::Pointer castToFloatFilter = CastToFloatFilterType::New();

	    typedef itk::SkeletonizeBinaryImageFilter<InputImageType, InputImageType> SkeletonFilterType;
	    SkeletonFilterType::Pointer skeletonFilterForRuleA = SkeletonFilterType::New();
	    SkeletonFilterType::Pointer skeletonFilterForRuleB = SkeletonFilterType::New();

	    typedef itk::BinaryThinningImageFilter3D<InputImageType, InputImageType> Thinning3DFilterType;
	    Thinning3DFilterType::Pointer thinning3DFilter = Thinning3DFilterType::New();

	    typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
	    CastFilterType::Pointer castFilter = CastFilterType::New();

      typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
	    OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

	    // Check we have exactly 2 values
	    unsigned long int pixelNumber = 0;
	    itk::ImageRegionConstIterator<InputImageType> checkIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
	    for (checkIterator.GoToBegin(); !checkIterator.IsAtEnd(); ++checkIterator)
	      {
		      InputPixelType tmp = checkIterator.Get();
		      if (tmp != args.backgroundValue && tmp != args.foregroundValue)
		        {
			        std::cerr << "Pixel number " << niftk::ConvertToString((int)pixelNumber)
			          << " is " << niftk::ConvertToString((int)tmp)
			        << " which does not equal the background (" << niftk::ConvertToString((int)args.backgroundValue)
			        << ") or foreground (" << niftk::ConvertToString((int)args.foregroundValue)
			        << ") value";
			        return -3;
		        }
		      pixelNumber++;
	      }

	    thresholdFilter->SetInput(imageReader->GetOutput());
	    thresholdFilter->SetInsideValue(1);
	    thresholdFilter->SetOutsideValue(0);
	    thresholdFilter->SetLowerThreshold(args.foregroundValue);
	    thresholdFilter->SetUpperThreshold(std::numeric_limits<InputPixelType>::max());
	    thresholdFilter->UpdateLargestPossibleRegion();

      if (args.doConnectedOnInput)
	      {
	        std::cout << "Doing connected components on input" << std::endl;
	        
	        connectedInputFilter->SetInput(thresholdFilter->GetOutput());
	        connectedInputFilter->SetFullyConnected(true);
	        connectedInputFilter->UpdateLargestPossibleRegion();

	        relabelInputFilter->SetInput(connectedInputFilter->GetOutput());
	        relabelInputFilter->UpdateLargestPossibleRegion();
	        
	        pickLargestInputComponentFilter->SetInput(relabelInputFilter->GetOutput());
	        pickLargestInputComponentFilter->SetInsideValue(1);
	        pickLargestInputComponentFilter->SetOutsideValue(0);
	        pickLargestInputComponentFilter->SetLowerThreshold(1);
	        pickLargestInputComponentFilter->SetUpperThreshold(1);
	        pickLargestInputComponentFilter->UpdateLargestPossibleRegion();
	      }    

      if (args.useDoht)
	      {

		      std::cout << "Doing DOHT, starting with Chamfer Distance transform" << std::endl;

          unsigned int weights[] = { 3, 4, 5 };
          chamferFilter->SetDistanceFromObject(false);
          chamferFilter->SetWeights(weights, weights+3);
          if (args.doConnectedOnInput)
            {
              chamferFilter->SetInput(pickLargestInputComponentFilter->GetOutput());    
            }
          else
            {
              chamferFilter->SetInput(thresholdFilter->GetOutput());    
            }
          chamferFilter->UpdateLargestPossibleRegion();
      
		      castToFloatFilter->SetInput(chamferFilter->GetOutput());
		      castToFloatFilter->UpdateLargestPossibleRegion();

		      if (args.doMaximalBallCheck)
		        {
		          std::cout << "Doing Rule B" << std::endl;
		          
		          if (args.doConnectedOnInput)
		            {
		              skeletonFilterForRuleB->SetInput(pickLargestInputComponentFilter->GetOutput());    
		            }
		          else
		            {
		              skeletonFilterForRuleB->SetInput(thresholdFilter->GetOutput());
		            }
		          skeletonFilterForRuleB->SetDistanceTransform(castToFloatFilter->GetOutput());
		          skeletonFilterForRuleB->SetTau(args.tau);
		          skeletonFilterForRuleB->UseRuleB();
		          skeletonFilterForRuleB->UpdateLargestPossibleRegion();

		          skeletonFilterForRuleA->SetInput(skeletonFilterForRuleB->GetOutput());
		        }
		      else
		        {
		          if (args.doConnectedOnInput)
		            {
		              skeletonFilterForRuleA->SetInput(pickLargestInputComponentFilter->GetOutput());    
		            }
		          else
		            {
		              skeletonFilterForRuleA->SetInput(thresholdFilter->GetOutput());
		            }
		        }

		      std::cout << "Doing Rule A" << std::endl;
		      
          skeletonFilterForRuleA->SetDistanceTransform(castToFloatFilter->GetOutput());
          skeletonFilterForRuleA->UseRuleA();
          skeletonFilterForRuleA->SetCheckMedialAxis(args.doMedialAxisCheck);
          skeletonFilterForRuleA->SetCheckMedialSurface(args.doMedialSurfaceCheck);
          skeletonFilterForRuleA->UpdateLargestPossibleRegion();
      
	      }
	    else
	      {
          std::cout << "Doing itkBinaryThinningImageFilter3D" << std::endl;

		      thinning3DFilter->SetInput(thresholdFilter->GetOutput());

	      }

      if (args.doConnectedOnOutput)
        {
          std::cout << "Doing connected components on output" << std::endl;
        
          if (args.useDoht)
            {
              connectedOutputFilter->SetInput(skeletonFilterForRuleA->GetOutput());   
            }
          else
            {
              connectedOutputFilter->SetInput(thinning3DFilter->GetOutput()); 
            }
          connectedOutputFilter->SetFullyConnected(true);
          connectedOutputFilter->UpdateLargestPossibleRegion();

          relabelOutputFilter->SetInput(connectedOutputFilter->GetOutput());
          relabelOutputFilter->UpdateLargestPossibleRegion();
        
          pickLargestOutputComponentFilter->SetInput(relabelOutputFilter->GetOutput());
          pickLargestOutputComponentFilter->SetInsideValue(1);
          pickLargestOutputComponentFilter->SetOutsideValue(0);
          pickLargestOutputComponentFilter->SetLowerThreshold(1);
          pickLargestOutputComponentFilter->SetUpperThreshold(1);
          pickLargestOutputComponentFilter->UpdateLargestPossibleRegion();
        
          thresholdBackFilter->SetInput(pickLargestOutputComponentFilter->GetOutput());
        
        }
      else
        {
          if (args.useDoht)
            {
              thresholdBackFilter->SetInput(skeletonFilterForRuleA->GetOutput());  
            }
          else
            {
              thresholdBackFilter->SetInput(thinning3DFilter->GetOutput());  
            }
        }

      std::cout << "Thresholding, casting and writing" << std::endl;
      
	    thresholdBackFilter->SetInsideValue(args.foregroundValue);
	    thresholdBackFilter->SetOutsideValue(args.backgroundValue);
	    thresholdBackFilter->SetLowerThreshold(args.foregroundValue);
	    thresholdBackFilter->SetUpperThreshold(std::numeric_limits<InputPixelType>::max());

	    castFilter->SetInput(thresholdBackFilter->GetOutput());

  	  writer->SetInput(castFilter->GetOutput());
	    writer->SetFileName(args.outputImage);
	    writer->Update();

    }
  else
    {
      std::cout << "Unsuported image dimension" << std::endl;
      exit( EXIT_FAILURE );
    }

  return EXIT_SUCCESS;
}
