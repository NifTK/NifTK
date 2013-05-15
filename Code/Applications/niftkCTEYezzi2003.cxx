/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkCheckForThreeLevelsFilter.h>
#include <itkLaplacianSolverImageFilter.h>
#include <itkScalarImageToNormalizedGradientVectorImageFilter.h>
#include <itkRelaxStreamlinesFilter.h>
#include <itkOrderedTraversalStreamlinesFilter.h>
#include <itkCastImageFilter.h>

/*!
 * \file niftkCTEYezzi2003.cxx
 * \page niftkCTEYezzi2003
 * \section niftkCTEYezzi2003Summary Implements Yezzi and Prince, IEEE TMI Vol. 22, No. 10, Oct 2003, for cortical thickness.
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements Yezzi and Prince, IEEE TMI Vol. 22, No. 10, Oct 2003," << std::endl;
  std::cout << "  using Diep et. al ISBI 2007 to cope with anisotropic voxel sizes, and optionally, initialize boundaries." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i    <filename>        Input segmented image, with exactly 3 label values " << std::endl;
  std::cout << "    -o    <filename>        Output distance image" << std::endl << std::endl;      
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -w    <float> [1]       Label for white matter" << std::endl;
  std::cout << "    -g    <float> [2]       Label for grey matter" << std::endl;
  std::cout << "    -c    <float> [3]       Label for extra-cerebral matter" << std::endl;  
  std::cout << "    -low  <float> [0]       Low Potential (voltage)" << std::endl;
  std::cout << "    -high <float> [10000]   High Potential (voltage)" << std::endl;
  std::cout << "    -le   <float> [0.00001] Laplacian relaxation convergence ratio (epsilon)" << std::endl;
  std::cout << "    -li   <int>   [200]     Laplacian relaxation max iterations" << std::endl;
  std::cout << "    -pe   <float> [0.00001] PDE relaxation convergence ratio (epsilon)" << std::endl;
  std::cout << "    -pi   <int>   [100]     PDE relaxation max iterations" << std::endl;
  std::cout << "    -ot                     Use ordered traversal (which ignores -pe and -pi)" << std::endl;
  std::cout << "    -sigma <float> [0]      Sigma for smoothing of vector normals. Default 0 (i.e. off)." << std::endl;  
  std::cout << "    -lapl <filename>        Write out Laplacian image" << std::endl; 
  std::cout << "    -noOpt                  Don't use Gauss-Siedel optimisation in Laplacian iterations" << std::endl;  
  std::cout << "    -dontInitBoundary       Don't initialize the boundary to minus half mean voxel spacing, see section 2.2.1 in Diep et. al. ISBI 2007." << std::endl;
  
}

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  std::string laplacianImage;
  double grey;
  double white;
  double csf;
  double low;
  double high;
  double laplaceRatio;
  double pdeRatio;
  int laplaceIters;
  int pdeIters;
  bool orderedTraversal;
  bool dontUseGaussSeidel;
  bool initBoundary;
  bool sigma;
};

template <int Dimension> 
int DoMain(arguments args)
{
  typedef  float ScalarType;
  typedef  float OutputScalarType;

  typedef typename itk::Image< ScalarType, Dimension >  InputImageType; 
  typedef typename itk::Image< OutputScalarType, Dimension >  OutputImageType;  
  typedef typename itk::ImageFileReader< InputImageType  > InputImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  
  typename InputImageReaderType::Pointer  imageReader  = InputImageReaderType::New();
  imageReader->SetFileName(  args.inputImage );
  

  try 
    { 
      std::cout << "Loading input image:" << args.inputImage << std::endl;
      imageReader->Update();
      std::cout << "Done" << std::endl;
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  // Setup objects to build registration.
  typedef typename itk::CheckForThreeLevelsFilter<InputImageType> CheckFilterType;
  typedef typename itk::LaplacianSolverImageFilter<InputImageType, ScalarType> LaplaceFilterType;
  typedef typename itk::ScalarImageToNormalizedGradientVectorImageFilter<InputImageType, ScalarType> NormalsFilterType;
  typedef typename itk::RelaxStreamlinesFilter<InputImageType, ScalarType, Dimension> RelaxFilterType;
  typedef typename itk::OrderedTraversalStreamlinesFilter<InputImageType, ScalarType, Dimension> OrderedTraversalFilterType;
  typedef typename itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;

  typename CheckFilterType::Pointer checkFilter = CheckFilterType::New();
  checkFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  checkFilter->SetInput(imageReader->GetOutput());

  typename LaplaceFilterType::Pointer laplaceFilter = LaplaceFilterType::New();
  laplaceFilter->SetInput(checkFilter->GetOutput());
  laplaceFilter->SetLowVoltage(args.low);
  laplaceFilter->SetHighVoltage(args.high);
  laplaceFilter->SetMaximumNumberOfIterations(args.laplaceIters);
  laplaceFilter->SetEpsilonConvergenceThreshold(args.laplaceRatio);
  laplaceFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  laplaceFilter->SetUseGaussSeidel(!args.dontUseGaussSeidel);
  
  typename NormalsFilterType::Pointer normalsFilter = NormalsFilterType::New();
  normalsFilter->SetInput(laplaceFilter->GetOutput());
  normalsFilter->SetUseMillimetreScaling(true);
  normalsFilter->SetDivideByTwo(true);
  normalsFilter->SetNormalize(true);
  normalsFilter->SetSigma(args.sigma);
  normalsFilter->SetDerivativeType(NormalsFilterType::DERIVATIVE_OF_GAUSSIAN);
  
  typename RelaxFilterType::Pointer relaxFilter = RelaxFilterType::New();
  typename OrderedTraversalFilterType::Pointer orderedTraversalFilter = OrderedTraversalFilterType::New();
  typename CastFilterType::Pointer castFilter = CastFilterType::New();
  
  // We are using EITHER the relaxFilter, or the orderedTraversalFilter
  relaxFilter->SetScalarImage(laplaceFilter->GetOutput());
  relaxFilter->SetVectorImage(normalsFilter->GetOutput());
  relaxFilter->SetSegmentedImage(checkFilter->GetOutput());
  relaxFilter->SetInitializeBoundaries(args.initBoundary);
  relaxFilter->SetLowVoltage(args.low);
  relaxFilter->SetHighVoltage(args.high);
  relaxFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  relaxFilter->SetMaximumNumberOfIterations(args.pdeIters);
  relaxFilter->SetEpsilonConvergenceThreshold(args.pdeRatio);  
  
  orderedTraversalFilter->SetScalarImage(laplaceFilter->GetOutput());
  orderedTraversalFilter->SetVectorImage(normalsFilter->GetOutput());
  orderedTraversalFilter->SetLowVoltage(args.low);
  orderedTraversalFilter->SetHighVoltage(args.high);
  
  if (args.orderedTraversal)
    {
      castFilter->SetInput(orderedTraversalFilter->GetOutput());
    }
  else
    {
      castFilter->SetInput(relaxFilter->GetOutput());
    }
  
  // And Write the output.
  typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
  outputImageWriter->SetFileName(  args.outputImage );
  outputImageWriter->SetInput(castFilter->GetOutput());
  outputImageWriter->Update();

  // Optionally write out Laplacian image.
  if (args.laplacianImage.length() > 0)
    {
      castFilter->SetInput(laplaceFilter->GetOutput());
      
      outputImageWriter->SetFileName(args.laplacianImage);
      outputImageWriter->Update(); 
    }

  return 0;
}

/**
 * \brief Implements Yezzi and Prince, IEEE TMI Vol. 22, No. 10, Oct 2003.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.white = 1;
  args.grey = 2;
  args.csf = 3;
  args.low = 0;
  args.high = 10000;
  args.laplaceRatio = 0.00001;
  args.pdeRatio = 0.00001;
  args.laplaceIters = 200;
  args.pdeIters = 100;
  args.orderedTraversal = false;
  args.dontUseGaussSeidel = false;
  args.initBoundary = true;
  args.sigma = 0;
  
  
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
    else if(strcmp(argv[i], "-lapl") == 0){
      args.laplacianImage=argv[++i];
      std::cout << "Set -lapl=" << args.laplacianImage << std::endl;
    }    
    else if(strcmp(argv[i], "-li") == 0){
      args.laplaceIters=atoi(argv[++i]);
      std::cout << "Set -li=" << niftk::ConvertToString(args.laplaceIters) << std::endl;
    }
    else if(strcmp(argv[i], "-pi") == 0){
      args.pdeIters=atoi(argv[i]);
      std::cout << "Set -pi=" << niftk::ConvertToString(args.pdeIters) << std::endl;
    }
    else if(strcmp(argv[i], "-c") == 0){
      args.csf=atof(argv[++i]);
      std::cout << "Set -c=" << niftk::ConvertToString(args.csf) << std::endl;
    }
    else if(strcmp(argv[i], "-g") == 0){
      args.grey=atof(argv[++i]);
      std::cout << "Set -g=" << niftk::ConvertToString(args.grey) << std::endl;
    }
    else if(strcmp(argv[i], "-w") == 0){
      args.white=atof(argv[++i]);
      std::cout << "Set -w=" << niftk::ConvertToString(args.white) << std::endl;
    }
    else if(strcmp(argv[i], "-low") == 0){
      args.low=atof(argv[++i]);
      std::cout << "Set -low=" << niftk::ConvertToString(args.low) << std::endl;
    }
    else if(strcmp(argv[i], "-high") == 0){
      args.high=atof(argv[++i]);
      std::cout << "Set -high=" << niftk::ConvertToString(args.high) << std::endl;
    }
    else if(strcmp(argv[i], "-le") == 0){
      args.laplaceRatio=atof(argv[++i]);
      std::cout << "Set -le=" << niftk::ConvertToString(args.laplaceRatio) << std::endl;
    }
    else if(strcmp(argv[i], "-pe") == 0){
      args.pdeRatio=atof(argv[++i]);
      std::cout << "Set -pe=" << niftk::ConvertToString(args.pdeRatio) << std::endl;
    }
    else if(strcmp(argv[i], "-ot") == 0){
      args.orderedTraversal=true;
      std::cout << "Set -ot=" << niftk::ConvertToString(args.orderedTraversal) << std::endl;
    }
    else if(strcmp(argv[i], "-noOpt") == 0){
      args.dontUseGaussSeidel=true;
      std::cout << "Set -noOpt=" << niftk::ConvertToString(args.dontUseGaussSeidel) << std::endl;
    }
    else if(strcmp(argv[i], "-dontInitBoundary") == 0){
      args.initBoundary=false;
      std::cout << "Set -dontInitBoundary=" << niftk::ConvertToString(args.initBoundary) << std::endl;
    } 
    else if(strcmp(argv[i], "-sigma") == 0){
      args.sigma=atof(argv[++i]);
      std::cout << "Set -sigma=" << niftk::ConvertToString(args.sigma) << std::endl;
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

  if(args.laplaceIters < 1 ){
    std::cerr << argv[0] << "\tThe laplaceIters must be >= 1" << std::endl;
    return -1;
  }

  if(args.pdeIters < 1 ){
    std::cerr << argv[0] << "\tThe pdeIters must be >= 1" << std::endl;
    return -1;
  }

  if(args.laplaceRatio < 0 ){
    std::cerr << argv[0] << "\tThe laplaceRatio must be > 0" << std::endl;
    return -1;
  }

  if(args.pdeRatio < 0 ){
    std::cerr << argv[0] << "\tThe pdeRatio must be > 0" << std::endl;
    return -1;
  }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  int result;
  
  switch ( dims )
    {
      case 2:
        result = DoMain<2>(args);
        break;
      case 3:
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
}
