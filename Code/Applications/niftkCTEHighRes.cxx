/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCheckForThreeLevelsFilter.h"
#include "itkCorrectGMUsingPVMapFilter.h"
#include "itkCorrectGMUsingNeighbourhoodFilter.h"
#include "itkHighResLaplacianSolverImageFilter.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"
#include "itkHighResRelaxStreamlinesFilter.h"
#include "itkCastImageFilter.h"

/*!
 * \file niftkCTEHighRes.cxx
 * \page niftkCTEHighRes
 * \section niftkCTEHighResSummary Implements a high resolution version of Bourgeat et. al. ISBI 2008 to calculate cortical thickness.
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements a high resolution version of Bourgeat et. al. ISBI 2008, to calculate cortical thickness" << std::endl;
  std::cout << "  using Yezzi and Prince, IEEE TMI Vol. 22, No. 10, Oct 2003 for solving the thickness PDE by relaxation, " << std::endl;
  std::cout << "  using Diep et. al ISBI 2007 to cope with anisotropic voxel sizes." << std::endl;
  std::cout << "  Also implements Acosta et. al. MIA 13 (2009) 730-743 doi:10.1016/j.media.2009.07.03 if you specify -acosta flag," << std::endl;
  std::cout << "  where the difference is the way the GM map is corrected (section 2.3 in Acosta's paper, not 2.3.1 in Bourgeats paper). " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i    <filename>        Input segmented image, with exactly 3 label values " << std::endl;
  std::cout << "    -o    <filename>        Output distance image" << std::endl << std::endl;      
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -gmpv <filename>        Input grey matter PV map " << std::endl;  
  std::cout << "    -w    <float> [1]       Label for white matter" << std::endl;
  std::cout << "    -g    <float> [2]       Label for grey matter" << std::endl;
  std::cout << "    -c    <float> [3]       Label for extra-cerebral matter" << std::endl;  
  std::cout << "    -low  <float> [0]       Low Potential (voltage)" << std::endl;
  std::cout << "    -high <float> [10000]   High Potential (voltage)" << std::endl;
  std::cout << "    -le   <float> [0.00001] Laplacian relaxation convergence ratio (epsilon)" << std::endl;
  std::cout << "    -li   <int>   [200]     Laplacian relaxation max iterations" << std::endl;
  std::cout << "    -pe   <float> [0.00001] PDE relaxation convergence ratio (epsilon)" << std::endl;
  std::cout << "    -pi   <int>   [200]     PDE relaxation max iterations" << std::endl;
  std::cout << "    -t    <float> [0.5]     Threshold to iterate ray-casting towards" << std::endl;
  std::cout << "    -m    <float> [0.001]   Min step size in dichotomy search" << std::endl;
  std::cout << "    -n    <float>           Max distance for dichotomy search. Default is unset, so filter works out voxel diagonal length." << std::endl; 
  std::cout << "    -sigma <float>          Sigma for smoothing of vector normals. Default off." << std::endl;  
  std::cout << "    -lapl <filename>        Write out Laplacian image" << std::endl; 
  std::cout << "    -AcostaCorrection       Do GM correction as described in Acosta 2009, Section 2.3, and not Bourgeat 2008, section 2.3.1." << std::endl;
  std::cout << "    -BourgeatCorrection     Do GM correction as described in Bourgeat 2008, section 2.3.1" << std::endl;
  std::cout << "    -s    <float> [1.0]     Section 2.3.1. Threshold to re-classify GM" << std::endl;  
  std::cout << "    -noCSFCheck             Section 2.3.1. Don't do CSF check" << std::endl;
  std::cout << "    -noGreyCheck            Section 2.3.1. Don't do grey matter check" << std::endl;
  std::cout << "    -noLagrangian           Don't do Lagrangian Initialisation, so we solve PDE, with CSF and WM borders initialised to -half the voxel diagonal length (as in Diep et. al.)." << std::endl;
  std::cout << "    -vmf  <int>   [1]       Voxel Multiplication Factor, e.g. 2 means twice as many voxels in each dimension" << std::endl;  
}

struct arguments
{
  std::string inputImage;
  std::string gmpvmapImage;
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
  double segThreshold;
  double rayThreshold;
  double minStep;
  double maxDist;   
  bool doCSFCheck;
  bool doGreyCheck;
  bool doAcostaCorrection;
  bool doBourgeatCorrection;
  double sigma;
  bool userSetMaxDistance;
  bool useLagrangianInitialisation;  
  int voxelMultiplicationFactor;
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

  typename InputImageReaderType::Pointer  pvMapReader  = InputImageReaderType::New();
  pvMapReader->SetFileName(  args.gmpvmapImage );

  

  try 
    { 
      std::cout << "Loading input image:" + args.inputImage << std::endl;
      imageReader->Update();
      std::cout << "Done" << std::endl;

      if (args.gmpvmapImage.length() > 0)
        {
          std::cout << "Loading input grey matter pv map:" << args.gmpvmapImage << std::endl;
          pvMapReader->Update();
          std::cout << "Done" << std::endl;
          
        }
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  // Setup objects to build registration.
  typedef typename itk::CheckForThreeLevelsFilter<InputImageType> CheckForThreeLevelsFilterType;
  typedef typename itk::CorrectGMUsingPVMapFilter<InputImageType> CorrectUsingBourgeatFilterType;
  typedef typename itk::CorrectGMUsingNeighbourhoodFilter<InputImageType> CorrectGMUsingAcostaFilterType;
  typedef typename itk::HighResLaplacianSolverImageFilter<InputImageType, ScalarType> LaplaceFilterType;
  typedef typename itk::ScalarImageToNormalizedGradientVectorImageFilter<InputImageType, ScalarType> NormalsFilterType;
  typedef typename itk::HighResRelaxStreamlinesFilter<InputImageType, ScalarType, Dimension> RelaxFilterType;

  typename CheckForThreeLevelsFilterType::Pointer checkForThreeLevelsFilter = CheckForThreeLevelsFilterType::New();
  checkForThreeLevelsFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  checkForThreeLevelsFilter->SetSegmentedImage(imageReader->GetOutput());
  
  typename CorrectUsingBourgeatFilterType::Pointer correctGMUsingPVFilter = CorrectUsingBourgeatFilterType::New();
  correctGMUsingPVFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  correctGMUsingPVFilter->SetSegmentedImage(imageReader->GetOutput());
  correctGMUsingPVFilter->SetGMPVMap(pvMapReader->GetOutput());
  correctGMUsingPVFilter->SetGreyMatterThreshold(args.segThreshold);
  correctGMUsingPVFilter->SetDoCSFCheck(args.doCSFCheck);
  correctGMUsingPVFilter->SetDoGreyMatterCheck(args.doGreyCheck);
  
  typename CorrectGMUsingAcostaFilterType::Pointer correctGMUsingNeighbourhoodFilter = CorrectGMUsingAcostaFilterType::New();
  correctGMUsingNeighbourhoodFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  correctGMUsingNeighbourhoodFilter->SetSegmentedImage(imageReader->GetOutput());
    
  typename LaplaceFilterType::Pointer laplaceFilter = LaplaceFilterType::New();
  laplaceFilter->SetLowVoltage(args.low);
  laplaceFilter->SetHighVoltage(args.high);
  laplaceFilter->SetMaximumNumberOfIterations(args.laplaceIters);
  laplaceFilter->SetEpsilonConvergenceThreshold(args.laplaceRatio);
  laplaceFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  laplaceFilter->SetUseGaussSeidel(true);
  laplaceFilter->SetVoxelMultiplicationFactor(args.voxelMultiplicationFactor);
  if (args.doAcostaCorrection)
    {
      laplaceFilter->SetInput(correctGMUsingNeighbourhoodFilter->GetOutput());    
    }
  else if (args.doBourgeatCorrection)
    {
      laplaceFilter->SetSegmentedImage(correctGMUsingPVFilter->GetOutput());    
    }
  else
    {
      laplaceFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput());    
    }
  
  typename NormalsFilterType::Pointer normalsFilter = NormalsFilterType::New();
  normalsFilter->SetInput(laplaceFilter->GetOutput());
  normalsFilter->SetUseMillimetreScaling(true);
  normalsFilter->SetDivideByTwo(true);
  normalsFilter->SetNormalize(true);
  normalsFilter->SetSigma(args.sigma);  
  normalsFilter->SetDerivativeType(NormalsFilterType::DERIVATIVE_OF_GAUSSIAN);
  
  /****************************************************************************
   * We MUST run the laplacian bit first, so we can correctly set up PDE
   ***************************************************************************/
  normalsFilter->Update();
  
  typename RelaxFilterType::Pointer relaxFilter = RelaxFilterType::New();
  relaxFilter->SetScalarImage(laplaceFilter->GetOutput());
  relaxFilter->SetVectorImage(normalsFilter->GetOutput());
  if (args.doAcostaCorrection)
    {
      relaxFilter->SetSegmentedImage(correctGMUsingNeighbourhoodFilter->GetOutput());    
    }
  else if (args.doBourgeatCorrection)
    {
      relaxFilter->SetSegmentedImage(correctGMUsingPVFilter->GetOutput());    
    }
  else
    {
      relaxFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput());    
    }
  if (args.useLagrangianInitialisation)
    {
      relaxFilter->SetGMPVMap(pvMapReader->GetOutput());    
    }
  relaxFilter->SetLowVoltage(args.low);
  relaxFilter->SetHighVoltage(args.high);
  relaxFilter->SetMaximumNumberOfIterations(args.pdeIters);
  relaxFilter->SetEpsilonConvergenceThreshold(args.pdeRatio);  
  relaxFilter->SetStepSizeThreshold(args.minStep);
  relaxFilter->SetGreyMatterPercentage(args.rayThreshold); 
  relaxFilter->SetLabelThresholds(args.grey, args.white, args.csf);
  relaxFilter->SetMaximumSearchDistance(args.maxDist);
  relaxFilter->SetHighResLaplacianMap(laplaceFilter->GetMapOfVoxels());
  relaxFilter->SetVoxelMultiplicationFactor(args.voxelMultiplicationFactor);
  if (args.userSetMaxDistance)
    {
      std::cout << "Set n=" << niftk::ConvertToString((double)args.maxDist) << std::endl;
      relaxFilter->SetMaximumSearchDistance(args.maxDist);    
    }

  // And Write the output.
  typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
  outputImageWriter->SetFileName(args.outputImage);
  outputImageWriter->SetInput(relaxFilter->GetOutput());
  outputImageWriter->Update();

  // Optionally write out Laplacian image.
  if (args.laplacianImage.length() > 0)
    {
      outputImageWriter->SetFileName(args.laplacianImage);
      outputImageWriter->SetInput(laplaceFilter->GetOutput());
      outputImageWriter->Update(); 
    }

  return 0;
}

/**
 * \brief 
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
  args.pdeIters = 200;
  args.segThreshold = 1;
  args.rayThreshold = 0.5;
  args.minStep = 0.001;
  args.maxDist = 10;
  args.userSetMaxDistance = false;
  args.doCSFCheck = true;
  args.doGreyCheck = true;
  args.doAcostaCorrection = false;
  args.doBourgeatCorrection = false;
  args.sigma = 0;
  args.useLagrangianInitialisation = true;
  args.voxelMultiplicationFactor = 1;
  
  
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
    else if(strcmp(argv[i], "-gmpv") == 0){
      args.gmpvmapImage=argv[++i];
      std::cout << "Set -gmpv=" << args.gmpvmapImage << std::endl;
    }
    else if(strcmp(argv[i], "-li") == 0){
      args.laplaceIters=atoi(argv[++i]);
      std::cout << "Set -li=" << niftk::ConvertToString(args.laplaceIters) << std::endl;
    }
    else if(strcmp(argv[i], "-pi") == 0){
      args.pdeIters=atoi(argv[++i]);
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
    else if(strcmp(argv[i], "-s") == 0){
      args.segThreshold=atof(argv[++i]);
      std::cout << "Set -s=" << niftk::ConvertToString(args.segThreshold) << std::endl;
    }
    else if(strcmp(argv[i], "-t") == 0){
      args.rayThreshold=atof(argv[++i]);
      std::cout << "Set -t=" << niftk::ConvertToString(args.rayThreshold) << std::endl;
    }
    else if(strcmp(argv[i], "-m") == 0){
      args.minStep=atof(argv[++i]);
      std::cout << "Set -m=" << niftk::ConvertToString(args.minStep) << std::endl;
    }
    else if(strcmp(argv[i], "-n") == 0){
      args.maxDist=atof(argv[++i]);
      args.userSetMaxDistance = true;
      std::cout << "Set -n=" << niftk::ConvertToString(args.maxDist) << std::endl;
    }
    else if(strcmp(argv[i], "-noCSFCheck") == 0){
      args.doCSFCheck = false;
      std::cout << "Set -noCSFCheck=" << niftk::ConvertToString(args.doCSFCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-noGreyCheck") == 0){
      args.doGreyCheck = false;
      std::cout << "Set -noGreyCheck=" << niftk::ConvertToString(args.doGreyCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-AcostaCorrection") == 0){
      args.doAcostaCorrection = true;
      std::cout << "Set -AcostaCorrection=" << niftk::ConvertToString(args.doAcostaCorrection) << std::endl;
    }
    else if(strcmp(argv[i], "-BourgeatCorrection") == 0){
      args.doBourgeatCorrection = true;
      std::cout << "Set -BourgeatCorrection=" << niftk::ConvertToString(args.doBourgeatCorrection) << std::endl;
    }            
    else if(strcmp(argv[i], "-sigma") == 0){
      args.sigma=atof(argv[++i]);
      std::cout << "Set -sigma=" << niftk::ConvertToString(args.sigma) << std::endl;
    }
    else if(strcmp(argv[i], "-noLagrangian") == 0){
      args.useLagrangianInitialisation = false;
      std::cout << "Set -noLagrangian=" << niftk::ConvertToString(args.useLagrangianInitialisation) << std::endl;
    }
    else if(strcmp(argv[i], "-vmf") == 0){
      args.voxelMultiplicationFactor=atoi(argv[++i]);
      std::cout << "Set -vmf=" << niftk::ConvertToString(args.voxelMultiplicationFactor) << std::endl;
    }    
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }        
  }
  
  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0 )
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

  if(args.segThreshold <= 0 || args.segThreshold > 1){
    std::cerr << argv[0] << "\tThe segThreshold must be > 0 and <= 1" << std::endl;
    return -1;
  }

  if(args.rayThreshold <= 0 || args.rayThreshold > 1){
    std::cerr << argv[0] << "\tThe rayThreshold must be > 0 and <= 1" << std::endl;
    return -1;
  }

  if(args.minStep <= 0){
    std::cerr << argv[0] << "\tThe minStep must be > 0" << std::endl;
    return -1;
  }

  if(args.maxDist < 0){
    std::cerr << argv[0] << "\tThe maxDist must be >= 0" << std::endl;
    return -1;
  }

  if(args.voxelMultiplicationFactor < 1 ){
    std::cerr << argv[0] << "\tThe voxel multiplication factor must be >= 1" << std::endl;
    return -1;
  }

  if (args.doAcostaCorrection && args.doBourgeatCorrection)
    {
      std::cerr << argv[0] << "\tThe -AcostaCorrection and -BourgeatCorrection are mutually exclusive" << std::endl;
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
