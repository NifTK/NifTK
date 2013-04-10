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
#include "itkLaplacianSolverImageFilter.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"
#include "itkLagrangianInitializedRelaxStreamlinesFilter.h"
#include "itkRelaxStreamlinesFilter.h"
#include "itkCastImageFilter.h"
#include "itkJorgesInitializationRelaxStreamlinesFilter.h"
#include "itkSubtractConstantFromImageFilter.h"
#include "itkZeroCrossingImageFilter.h"

/*!
 * \file niftkCTEBourgeat2008.cxx
 * \page niftkCTEBourgeat2008
 * \section niftkCTEBourgeat2008Summary Implements Bourgeat et. al. ISBI 2008 to calculate cortical thickness.
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements Bourgeat et. al. ISBI 2008," << std::endl;
  std::cout << "  Includes solving the thickness PDE by relaxation as described in Yezzi and Prince, IEEE TMI Vol. 22, No. 10, Oct 2003." << std::endl;
  std::cout << "  Includes coping with anisotropic voxel sizes, as described in Diep et. al ISBI 2007." << std::endl;
  std::cout << "  Includes the GM correction described in Acosta et. al. MIA 13 (2009) 730-743 doi:10.1016/j.media.2009.07.03, section 2.3." << std::endl;
  std::cout << "  Includes the GM correction described in Bourgeat et. al. ISBI 2008, section 2.3.1." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -gmpv <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i    <filename>        Input segmented image, with exactly 3 label values " << std::endl;
  std::cout << "    -gmpv <filename>        Input grey matter PV map " << std::endl;
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
  std::cout << "    -pi   <int>   [200]     PDE relaxation max iterations" << std::endl;
  std::cout << "    -t    <float> [0.5]     Threshold to iterate ray-casting towards" << std::endl;
  std::cout << "    -m    <float> [0.001]   Min step size in dichotomy search" << std::endl;
  std::cout << "    -n    <float>           Max distance for dichotomy search. Default is unset, so filter works out voxel diagonal length." << std::endl;
  std::cout << "    -sigma <float>          Sigma for smoothing of vector normals. Default off." << std::endl;
  std::cout << "    -max <float> [10]       Max length/thickness." << std::endl;
  std::cout << "    -lapl <filename>        Write out Laplacian image" << std::endl;
  std::cout << "    -midline <filename>     Write out the midline if the Laplacian image." << std::endl;
  std::cout << "    -method [int] [1]       Method:"  << std::endl;
  std::cout << "                            1. No correction of GM grid, we just check that label image has 3 labels." << std::endl;
  std::cout << "                            2. Correct the GM, as describe in Acosta's paper, section 2.3, and use Lagrangian initialization" << std::endl;
  std::cout << "                            3. Correct the GM, as described in Bourgeat's paper, section 2.3.1, and use Lagrangian initialization" << std::endl;
  std::cout << "                            4. Do not correct the GM, and initialize the CSF/WM boundary voxels to minus half the voxel diagonal (images should really be isotropic), as described in Diep's paper" << std::endl;
  std::cout << "                            5. Use Jorge's method, no Lagrangian initialization." << std::endl;
  std::cout << "                            6. Correct the GM, as described in Acosta's paper, then do Jorge's initialization, rather than Lagrangian" << std::endl;
  std::cout << "    -notFullyConnected      For method 2. Don't do fully connected (27 in 3D, 8 in 2D) neighbourhood search, just do (6 in 3D, 4 in 2D) connected. " << std::endl;  
  std::cout << "    -s    <float> [1.0]     For method 3. Threshold to re-classify GM" << std::endl;  
  std::cout << "    -noCSFCheck             For method 3. Don't do CSF check" << std::endl;
  std::cout << "    -noGreyCheck            For method 3. Don't do grey matter check" << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string gmpvmapImage;
  std::string outputImage;
  std::string laplacianImage;
  std::string midlineImage;
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
  double sigma;
  bool userSetMaxDistance;
  bool fullyConnected;
  int method;
  double maxLength;
};

template <int Dimension> 
int DoMain(arguments args)
{
  typedef  float ScalarType;
  typedef  float OutputScalarType;
  
  typedef typename itk::Image< ScalarType, Dimension >       InputImageType; 
  typedef typename itk::Image< OutputScalarType, Dimension > OutputImageType;  
  typedef typename itk::ImageFileReader< InputImageType  >   InputImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >   OutputImageWriterType;
  
  typename InputImageReaderType::Pointer  imageReader  = InputImageReaderType::New();
  imageReader->SetFileName(  args.inputImage );
    
  typename InputImageReaderType::Pointer  pvMapReader  = InputImageReaderType::New();
  pvMapReader->SetFileName(  args.gmpvmapImage );
  

  try 
    { 
      std::cout << "Loading input image:" << args.inputImage << std::endl;
      imageReader->Update();
      std::cout << "Done" << std::endl;
      
      std::cout << "Loading input grey matter pv map:" << args.gmpvmapImage << std::endl;
      pvMapReader->Update();
      std::cout << "Done" << std::endl;
      
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  // Setup objects to build registration.
  typedef typename itk::CheckForThreeLevelsFilter<InputImageType>                                          CheckForThreeLevelsFilterType;
  typedef typename itk::CorrectGMUsingPVMapFilter<InputImageType>                                          CorrectUsingBourgeatFilterType;
  typedef typename itk::CorrectGMUsingNeighbourhoodFilter<InputImageType>                                  CorrectGMUsingAcostaFilterType;
  typedef typename itk::LaplacianSolverImageFilter<InputImageType, ScalarType>                             LaplaceFilterType;
  typedef typename itk::ScalarImageToNormalizedGradientVectorImageFilter<InputImageType, ScalarType>       NormalsFilterType;
  typedef typename itk::RelaxStreamlinesFilter<InputImageType, ScalarType, Dimension>                      NoLagrangianFilterType;
  typedef typename itk::LagrangianInitializedRelaxStreamlinesFilter<InputImageType, ScalarType, Dimension> LagrangianFilterType;
  typedef typename itk::JorgesInitializationRelaxStreamlinesFilter<InputImageType, ScalarType, Dimension>  JorgesInitializationFilterType;

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
  correctGMUsingNeighbourhoodFilter->SetUseFullNeighbourHood(args.fullyConnected);
  
  typename LaplaceFilterType::Pointer laplaceFilter = LaplaceFilterType::New();
  laplaceFilter->SetLowVoltage(args.low);
  laplaceFilter->SetHighVoltage(args.high);
  laplaceFilter->SetMaximumNumberOfIterations(args.laplaceIters);
  laplaceFilter->SetEpsilonConvergenceThreshold(args.laplaceRatio);
  laplaceFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  laplaceFilter->SetUseGaussSeidel(true);
  
  typename NormalsFilterType::Pointer normalsFilter = NormalsFilterType::New();
  normalsFilter->SetScalarImage(laplaceFilter->GetOutput());
  normalsFilter->SetUseMillimetreScaling(true);
  normalsFilter->SetDivideByTwo(true);
  normalsFilter->SetNormalize(true);
  normalsFilter->SetSigma(args.sigma);  
  normalsFilter->SetDerivativeType(NormalsFilterType::DERIVATIVE_OF_GAUSSIAN);

  typename JorgesInitializationFilterType::Pointer jorgesInitializationFilter = JorgesInitializationFilterType::New();
  jorgesInitializationFilter->SetScalarImage(laplaceFilter->GetOutput());
  jorgesInitializationFilter->SetVectorImage(normalsFilter->GetOutput());    
  jorgesInitializationFilter->SetGMPVMap(pvMapReader->GetOutput());
  jorgesInitializationFilter->SetLowVoltage(args.low);
  jorgesInitializationFilter->SetHighVoltage(args.high);
  jorgesInitializationFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  jorgesInitializationFilter->SetMaximumNumberOfIterations(args.pdeIters);
  jorgesInitializationFilter->SetEpsilonConvergenceThreshold(args.pdeRatio);  
  jorgesInitializationFilter->SetMaximumLength(args.maxLength);

  typename LagrangianFilterType::Pointer lagrangianFilter = LagrangianFilterType::New();
  lagrangianFilter->SetScalarImage(laplaceFilter->GetOutput());
  lagrangianFilter->SetVectorImage(normalsFilter->GetOutput());
  lagrangianFilter->SetGMPVMap(pvMapReader->GetOutput());
  lagrangianFilter->SetLowVoltage(args.low);
  lagrangianFilter->SetHighVoltage(args.high);
  lagrangianFilter->SetMaximumNumberOfIterations(args.pdeIters);
  lagrangianFilter->SetEpsilonConvergenceThreshold(args.pdeRatio); 
  lagrangianFilter->SetStepSizeThreshold(args.minStep);
  lagrangianFilter->SetGreyMatterPercentage(args.rayThreshold); 
  lagrangianFilter->SetLabelThresholds(args.grey, args.white, args.csf);
  lagrangianFilter->SetMaximumLength(args.maxLength);
  if (args.userSetMaxDistance)
    {
      std::cout << "Set n=" << niftk::ConvertToString((double)args.maxDist) << std::endl;
      lagrangianFilter->SetMaximumSearchDistance(args.maxDist);    
    }
  
  typename NoLagrangianFilterType::Pointer nolagrangianFilter = NoLagrangianFilterType::New();
  nolagrangianFilter->SetScalarImage(laplaceFilter->GetOutput());
  nolagrangianFilter->SetVectorImage(normalsFilter->GetOutput());
  nolagrangianFilter->SetInitializeBoundaries(true);
  nolagrangianFilter->SetLowVoltage(args.low);
  nolagrangianFilter->SetHighVoltage(args.high);
  nolagrangianFilter->SetLabelThresholds(args.grey, args.white, args.csf); 
  nolagrangianFilter->SetMaximumNumberOfIterations(args.pdeIters);
  nolagrangianFilter->SetEpsilonConvergenceThreshold(args.pdeRatio);  
  nolagrangianFilter->SetMaximumLength(args.maxLength);

  typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
  outputImageWriter->SetFileName(args.outputImage);

  typename OutputImageType::Pointer L0Image;
  typename OutputImageType::Pointer L1Image;

  if (args.method == 1)
    {
      std::cout << "Not correcting GM grid, just checking for 3 labels." << std::endl;
      laplaceFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput());
      lagrangianFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput());
      outputImageWriter->SetInput(lagrangianFilter->GetOutput());

      L0Image = lagrangianFilter->GetL0Image();
      L1Image = lagrangianFilter->GetL1Image();
    }
  else if (args.method == 2)
    {
      // Acosta's method, where we correct GM using neighbourhood, then do Lagrangian initialization.
      std::cout << "Doing Acosta's method" << std::endl;
      laplaceFilter->SetInput(correctGMUsingNeighbourhoodFilter->GetOutput());
      lagrangianFilter->SetSegmentedImage(correctGMUsingNeighbourhoodFilter->GetOutput());
      outputImageWriter->SetInput(lagrangianFilter->GetOutput());

      L0Image = lagrangianFilter->GetL0Image();
      L1Image = lagrangianFilter->GetL1Image();

    }
  else if (args.method == 3)
    {
      // Bourgeat's method, where we correct GM using PV rules, then do Lagrangian initialization.
      std::cout << "Doing Bourgeat's method" << std::endl;
      laplaceFilter->SetSegmentedImage(correctGMUsingPVFilter->GetOutput());
      lagrangianFilter->SetSegmentedImage(correctGMUsingPVFilter->GetOutput());
      outputImageWriter->SetInput(lagrangianFilter->GetOutput());

      L0Image = lagrangianFilter->GetL0Image();
      L1Image = lagrangianFilter->GetL1Image();
    }
  else if (args.method == 4)
    {
      // Diep's method, where we initialize boundaries to half the voxel diagonal, and don't do Lagrangian initialization.
      std::cout << "Doing Dieps's method" << std::endl;
      laplaceFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput());
      nolagrangianFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput()); 
      outputImageWriter->SetInput(nolagrangianFilter->GetOutput());

      L0Image = nolagrangianFilter->GetL0Image();
      L1Image = nolagrangianFilter->GetL1Image();
    }
  else if (args.method == 5)
    {
      // Jorge's method, where we initialize boundaries using his rules, and don't do Lagrangian initialization.
      std::cout << "Doing Jorge's method" << std::endl;
      laplaceFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput());
      jorgesInitializationFilter->SetSegmentedImage(checkForThreeLevelsFilter->GetOutput()); 
      outputImageWriter->SetInput(jorgesInitializationFilter->GetOutput());

      L0Image = jorgesInitializationFilter->GetL0Image();
      L1Image = jorgesInitializationFilter->GetL1Image();
    }
  else if (args.method == 6)
    {
      // Jorge's method, where we initialize boundaries using his rules, and don't do Lagrangian initialization.
      std::cout << "Doing Jorge's method, but correcting GM using Acosta's method" << std::endl;
      laplaceFilter->SetSegmentedImage(correctGMUsingNeighbourhoodFilter->GetOutput());
      jorgesInitializationFilter->SetSegmentedImage(correctGMUsingNeighbourhoodFilter->GetOutput()); 
      outputImageWriter->SetInput(jorgesInitializationFilter->GetOutput());      

      L0Image = jorgesInitializationFilter->GetL0Image();
      L1Image = jorgesInitializationFilter->GetL1Image();

    }
  

  // And Write the output, which forces the update.
  outputImageWriter->Update();
    
  // Optionally write out Laplacian image.
  if (args.laplacianImage.length() > 0)
    {
      outputImageWriter->SetInput(laplaceFilter->GetOutput());
      outputImageWriter->SetFileName(args.laplacianImage);
      outputImageWriter->Update(); 
    }

  // Optionally write out the midline image, generated from middle of Laplacian
  if (args.midlineImage.length() > 0)
    {
	  typedef typename itk::SubtractConstantFromImageFilter<OutputImageType, double, OutputImageType> SubtractFilterType;
	  typedef typename itk::ZeroCrossingImageFilter<OutputImageType, OutputImageType> ZeroCrossingFilterType;

	  typename SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();
	  typename ZeroCrossingFilterType::Pointer zeroCrossingFilter = ZeroCrossingFilterType::New();

	  subtractFilter->SetInput(laplaceFilter->GetOutput());
	  subtractFilter->SetConstant((args.high + args.low)/2.0);

	  zeroCrossingFilter->SetInput(subtractFilter->GetOutput());

      outputImageWriter->SetInput(zeroCrossingFilter->GetOutput());
      outputImageWriter->SetFileName(args.midlineImage);
      outputImageWriter->Update();
    }
  return 0;  
}

/**
 * \brief Implements Bourgeat et. al. ISBI 2008.
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
  args.sigma = 0;
  args.fullyConnected = true;
  args.method = 1;
  args.maxLength = 10;
  
  
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
    else if(strcmp(argv[i], "-gmpv") == 0){
      args.gmpvmapImage=argv[++i];
      std::cout << "Set -gmpv=" << args.gmpvmapImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -o=" << args.outputImage << std::endl;
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
    else if(strcmp(argv[i], "-li") == 0){
      args.laplaceIters=atoi(argv[++i]);
      std::cout << "Set -li=" << niftk::ConvertToString(args.laplaceIters) << std::endl;
    }
    else if(strcmp(argv[i], "-pi") == 0){
      args.pdeIters=atoi(argv[++i]);
      std::cout << "Set -pi=" << niftk::ConvertToString(args.pdeIters) << std::endl;
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
    else if(strcmp(argv[i], "-sigma") == 0){
      args.sigma=atof(argv[++i]);
      std::cout << "Set -sigma=" << niftk::ConvertToString(args.sigma) << std::endl;
    }
    else if(strcmp(argv[i], "-lapl") == 0){
      args.laplacianImage=argv[++i];
      std::cout << "Set -lapl=" << args.laplacianImage << std::endl;
    }
    else if(strcmp(argv[i], "-midline") == 0){
      args.midlineImage=argv[++i];
      std::cout << "Set -midline=" << args.midlineImage << std::endl;
    }
    else if(strcmp(argv[i], "-s") == 0){
      args.segThreshold=atof(argv[++i]);
      std::cout << "Set -s=" << niftk::ConvertToString(args.segThreshold) << std::endl;
    }
    else if(strcmp(argv[i], "-noCSFCheck") == 0){
      args.doCSFCheck = false;
      std::cout << "Set -noCSFCheck=" << niftk::ConvertToString(args.doCSFCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-noGreyCheck") == 0){
      args.doGreyCheck = false;
      std::cout << "Set -noGreyCheck=" << niftk::ConvertToString(args.doGreyCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-notFullyConnected") == 0){
      args.fullyConnected = false;
      std::cout << "Set -notFullyConnected=" << niftk::ConvertToString(args.fullyConnected) << std::endl;
    } 
    else if(strcmp(argv[i], "-method") == 0){
      args.method=atoi(argv[++i]);
      std::cout << "Set -method=" << niftk::ConvertToString(args.method) << std::endl;
    }
    else if(strcmp(argv[i], "-max") == 0){
      args.maxLength=atof(argv[++i]);
      std::cout << "Set -max=" << niftk::ConvertToString(args.maxLength) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }        
  }
  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0 || args.gmpvmapImage.length() == 0)
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

  if (args.method < 1 || args.method > 6)
    {
      std::cerr << argv[0] << "\tThe method must be >= 1 and <= 6" << std::endl;
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
