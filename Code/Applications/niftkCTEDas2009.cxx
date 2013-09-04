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
#include <itkCommandLineHelper.h>
#include <itkVector.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegistrationFilter.h>
#include <itkAddImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkTwinThresholdBoundaryFilter.h>
#include <itkShiftScaleImageFilter.h>
#include <itkImageToImageFilter.h>
#include <itkRegistrationBasedCTEFilter.h>
#include <itkRegistrationBasedCorticalThicknessFilter.h>
#include <itkCastImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkMultiplyImageFilter.h>

/*!
 * \file niftkCTEDas2009.cxx
 * \page niftkCTEDas2009
 * \section niftkCTEDas2009Summary Implements "Registration based cortical thickness measurement" S. R. Das et. al., NeuroImage 45 (2009) 867-879
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements \"Registration based cortical thickness measurement\" S. R. Das et. al., NeuroImage 45 (2009) 867-879" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -gmpv <filename> -wmpv <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i    <filename>         Input segmented image, with exactly 3 label values " << std::endl;
  std::cout << "    -gmpv <filename>         Input grey matter PV map " << std::endl;
  std::cout << "    -wmpv <filename>         Input white matter PV map " << std::endl;
  std::cout << "    -o    <filename>         Output thickness (scalar) image" << std::endl << std::endl;      
  std::cout << "*** [options]   ***" << std::endl << std::endl; 
  std::cout << "    -w    <float> [1]        Label for white matter" << std::endl;
  std::cout << "    -g    <float> [2]        Label for grey matter" << std::endl;
  std::cout << "    -c    <float> [3]        Label for extra-cerebral matter" << std::endl;    
  std::cout << "    -tpi  <filename>         Thickness prior image " << std::endl; 
  std::cout << "    -tpc  <float> [6.0]      Thickness prior constant " << std::endl;
  std::cout << "    -mi   <int>   [100]      Max iterations" << std::endl;
  std::cout << "    -m    <int>   [20]       Number of steps in ODE integration" << std::endl;
  std::cout << "    -l    <float> [1.0]      Gradient descent parameter Lambda" << std::endl;
  std::cout << "    -su   <float> [1.5]      Sigma (in mm) for Gaussian kernel when smoothing update field" << std::endl;
  std::cout << "    -sd   <float> [0.0]      Sigma (in mm) for Gaussian kernel when smoothing deformation field" << std::endl;
  std::cout << "    -e    <float> [0.0001]   Convergence threshold to stop iteration, let's call this Epsilon" << std::endl;
  std::cout << "    -a    <float> [1.0]      Weighting [0-1] for cost function, 1 gives all image similarity, 0 gives all velocity field energy" << std::endl;
  std::cout << "    -iso                     Resample to isotropic voxels" << std::endl;
  std::cout << "    -rescale [low high]      Rescale PV maps from [low-high] to [0-1], useful for testing with integer [0-255] images" << std::endl;
  std::cout << "    -old                     Default uses itkRegistrationBasedCorticalThicknessFilter.h, so the -old flag switches to itkRegistrationBasedCTEFilter.h" << std::endl;
  std::cout << "    -resampled <filename>    Output the transformed moving image at the end." << std::endl;
  std::cout << "    -displacement <filename> Output a vector image showing the displacement of the GW boundary. " << std::endl;
  std::cout << "    -st   <float> [0]        Sigma (in mm) for Gaussian kernel when smoothing output thickness field" << std::endl;
  std::cout << "    -sw   <int>   [10]       Kernel width for Gaussian kernel when smoothing output thickness field" << std::endl;
}

struct arguments
{
  std::string labelImage;
  std::string gmpvmapImage;
  std::string wmpvmapImage;
  std::string tauImage;
  std::string outputThicknessImage;
  std::string outputTransformedMovingImage;
  std::string outputDisplacementImage;
  double grey;
  double white;
  double csf;  
  double sigmaUpdate;
  double sigmaDeformation;
  double sigmaThickness;
  double epsilon;
  double lambda;
  double low;
  double high;
  double tau;
  double alpha;
  int    maxIterations;
  int    m;
  bool   resampleToIsotropic;
  bool   rescalePVMaps;
  bool   useThicknessPriorImage;
  bool   useOld;
  int    kernelWidth;
  bool   blurOutput;
};

template <int Dimension> 
int DoMain(arguments args)
{
  typedef float ScalarType;
  typedef float OutputScalarType;
  
  typedef typename itk::Vector< ScalarType, Dimension >      VectorPixelType;
  typedef typename itk::Image< VectorPixelType, Dimension >  VectorImageType;
  typedef typename itk::Image< ScalarType, Dimension >       InputImageType; 
  typedef typename itk::Image< OutputScalarType, Dimension > OutputImageType;  
  typedef typename itk::ImageFileReader< InputImageType  >   InputImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >   OutputImageWriterType;
  typedef typename itk::ImageFileWriter< VectorImageType >   OutputVectorImageWriterType;
  typedef typename itk::DiscreteGaussianImageFilter<OutputImageType, OutputImageType> GaussianFilterType;
  typedef typename itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyFilterType;
  
  typename InputImageReaderType::Pointer  labelReader  = InputImageReaderType::New();
  labelReader->SetFileName(args.labelImage);

  typename InputImageReaderType::Pointer  gmpvmapReader  = InputImageReaderType::New();
  gmpvmapReader->SetFileName(args.gmpvmapImage);

  typename InputImageReaderType::Pointer  wmpvmapReader  = InputImageReaderType::New();
  wmpvmapReader->SetFileName(args.wmpvmapImage);

  typename InputImageReaderType::Pointer  tauReader  = InputImageReaderType::New();
  

  try 
    {
      std::cout << "Loading label image:" << args.labelImage << std::endl;
      labelReader->Update();
      std::cout << "Done" << std::endl;
      
      std::cout << "Loading grey matter pv map:" << args.gmpvmapImage << std::endl;
      gmpvmapReader->Update();
      std::cout << "Done" << std::endl;

      std::cout << "Loading white matter pv map:" + args.wmpvmapImage << std::endl;
      wmpvmapReader->Update();
      std::cout << "Done" << std::endl;

      if (args.useThicknessPriorImage) 
        {
          std::cout << "Loading tau thickness prior:" + args.tauImage << std::endl;
          tauReader->SetFileName(args.tauImage);
          tauReader->Update();
          std::cout << "Done" << std::endl;
        }
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  // These are pointers to hold the images we will be using as input.
  typename InputImageType::Pointer labelImage = labelReader->GetOutput();     // Label image
  typename InputImageType::Pointer gmpvmapImage = gmpvmapReader->GetOutput(); // Grey matter probability map
  typename InputImageType::Pointer wmpvmapImage = wmpvmapReader->GetOutput(); // White matter probability map
  typename InputImageType::Pointer tauImage = InputImageType::New();          // Thickness prior map

  // If user specified a thickness prior image, use that,
  // otherwise, generate an image, with a set constant value (useful for testing).
  if (args.useThicknessPriorImage)
    {
      std::cout << "Using tau thickness prior image, instead of constant value" << std::endl;
      tauImage = tauReader->GetOutput();  
    }
  else
    {
      std::cout << "Generating a tau thickness prior image with constant value:" <<  niftk::ConvertToString(args.tau) << std::endl;
      tauImage->SetRegions(gmpvmapImage->GetLargestPossibleRegion());
      tauImage->SetSpacing(gmpvmapImage->GetSpacing());
      tauImage->SetOrigin(gmpvmapImage->GetOrigin());
      tauImage->SetDirection(gmpvmapImage->GetDirection());
      tauImage->Allocate();
      tauImage->FillBuffer(args.tau);
    }

  if (args.rescalePVMaps)
    {
      std::cout << "Rescaling PV maps to [0-1]" << std::endl;
      typedef itk::ShiftScaleImageFilter<InputImageType, InputImageType> RescaleFilterType;
      typename RescaleFilterType::Pointer rescaleGMFilter = RescaleFilterType::New();
      rescaleGMFilter->SetInput(gmpvmapReader->GetOutput());
      rescaleGMFilter->SetShift(-args.low);
      rescaleGMFilter->SetScale(1.0/(args.high - args.low));
      rescaleGMFilter->Update();
      gmpvmapImage = rescaleGMFilter->GetOutput();
      
      typename RescaleFilterType::Pointer rescaleWMFilter = RescaleFilterType::New();
      rescaleWMFilter->SetInput(wmpvmapReader->GetOutput());
      rescaleWMFilter->SetShift(-args.low);
      rescaleWMFilter->SetScale(1.0/(args.high - args.low));      
      rescaleWMFilter->Update();
      wmpvmapImage = rescaleWMFilter->GetOutput();
      
      std::cout << "Done" << std::endl;
    }

  if (args.resampleToIsotropic)
    {
      std::cout << "Resampling to IsoTropic 1mm voxels" << std::endl;
      typedef itk::ImageRegistrationFilter<InputImageType, InputImageType, Dimension, double, double> RegistrationFilterType;
      typename RegistrationFilterType::Pointer registrationFilter = RegistrationFilterType::New();
      typename InputImageType::SpacingType spacing;
      spacing.Fill(1);
      labelImage = registrationFilter->ResampleToVoxelSize(labelImage, 0, (itk::InterpolationTypeEnum)2, spacing);
      gmpvmapImage = registrationFilter->ResampleToVoxelSize(gmpvmapImage, 0, (itk::InterpolationTypeEnum)2, spacing);
      wmpvmapImage = registrationFilter->ResampleToVoxelSize(wmpvmapImage, 0, (itk::InterpolationTypeEnum)2, spacing);
      tauImage = registrationFilter->ResampleToVoxelSize(tauImage, 0, (itk::InterpolationTypeEnum)2, spacing);
      std::cout << "Done" << std::endl;
    }
  
  typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdFilterType;
  
  // We also need P_{wg} = P_{w} + P_{g} (see paper).
  
  std::cout << "Adding P(w) + P(g) to make P(wg)" << std::endl;
  typedef typename itk::AddImageFilter<InputImageType, InputImageType> AddFilterType;
  typename AddFilterType::Pointer addFilter = AddFilterType::New();
  addFilter->SetInput1(gmpvmapImage);
  addFilter->SetInput2(wmpvmapImage);
  addFilter->Update();
  std::cout << "Done" << std::endl;
  
  // Im putting GM/WM through 2 threshold filters to clamp it to [0-1], as it makes testing easier.
  typedef typename itk::ThresholdImageFilter<InputImageType> ThresholdFilterType;
  typename ThresholdFilterType::Pointer clampGMToZeroFilter = ThresholdFilterType::New();
  clampGMToZeroFilter->SetInput(addFilter->GetOutput());
  clampGMToZeroFilter->SetOutsideValue(0);
  clampGMToZeroFilter->ThresholdBelow(0);
  clampGMToZeroFilter->SetInPlace(true);
  clampGMToZeroFilter->Update();
  
  typename ThresholdFilterType::Pointer clampGMToOneFilter = ThresholdFilterType::New();
  clampGMToOneFilter->SetInput(clampGMToZeroFilter->GetOutput());
  clampGMToOneFilter->SetOutsideValue(1);
  clampGMToOneFilter->ThresholdAbove(1);
  clampGMToOneFilter->SetInPlace(true);
  clampGMToOneFilter->Update();
  
  // Im putting WM through 2 threshold filters to clamp it to [0-1], as it makes testing easier.
  typename ThresholdFilterType::Pointer clampWMToZeroFilter = ThresholdFilterType::New();
  clampWMToZeroFilter->SetInput(wmpvmapImage);
  clampWMToZeroFilter->SetOutsideValue(0);
  clampWMToZeroFilter->ThresholdBelow(0);
  clampWMToZeroFilter->SetInPlace(true);
  clampWMToZeroFilter->Update();
  
  typename ThresholdFilterType::Pointer clampWMToOneFilter = ThresholdFilterType::New();
  clampWMToOneFilter->SetInput(clampWMToZeroFilter->GetOutput());
  clampWMToOneFilter->SetOutsideValue(1);
  clampWMToOneFilter->ThresholdAbove(1);
  clampWMToOneFilter->SetInPlace(true);
  clampWMToOneFilter->Update();

  // Also, I'm thresholding GM and WM to get a good Grey White interface, and also a GM Mask.
  typename BinaryThresholdFilterType::Pointer thresholdedGreyImage = BinaryThresholdFilterType::New();
  thresholdedGreyImage->SetInput(labelImage);
  thresholdedGreyImage->SetOutsideValue(0);
  thresholdedGreyImage->SetInsideValue(1);
  thresholdedGreyImage->SetUpperThreshold(args.grey);
  thresholdedGreyImage->SetLowerThreshold(args.grey);
  thresholdedGreyImage->Update();

  typename BinaryThresholdFilterType::Pointer thresholdedWhiteImage = BinaryThresholdFilterType::New();
  thresholdedWhiteImage->SetInput(labelImage);
  thresholdedWhiteImage->SetOutsideValue(0);
  thresholdedWhiteImage->SetInsideValue(1);
  thresholdedWhiteImage->SetUpperThreshold(args.white);
  thresholdedWhiteImage->SetLowerThreshold(args.white);
  thresholdedWhiteImage->Update();

  // We also need the GWI (see paper).
  std::cout << "Extracting GWI" << std::endl;
  typedef typename itk::TwinThresholdBoundaryFilter<InputImageType> BoundaryFilterType;
  typename BoundaryFilterType::Pointer boundaryFilter = BoundaryFilterType::New();
  boundaryFilter->SetInput1(thresholdedGreyImage->GetOutput());
  boundaryFilter->SetInput2(thresholdedWhiteImage->GetOutput());
  boundaryFilter->SetThresholdForInput1(0.5);
  boundaryFilter->SetThresholdForInput2(0.5);
  boundaryFilter->SetTrue(1);
  boundaryFilter->SetFalse(0);
  boundaryFilter->Update();
  std::cout << "Done" << std::endl;

  // I test this algorithm with PNG images, which have a data range of 0-255, but they must be rescaled from 0-1.
  // So best to check inputs, as several times now, I've messed this up.
  
  typedef typename itk::MinimumMaximumImageCalculator<InputImageType> MinMaxType;
  typename MinMaxType::Pointer minMaxChecker = MinMaxType::New();

  minMaxChecker->SetImage(clampWMToOneFilter->GetOutput());
  minMaxChecker->Compute();
  if (minMaxChecker->GetMinimum() < 0 || minMaxChecker->GetMaximum() > 1)
    {
      std::cerr <<"The white matter model image should be thresholded 0-1, or have probabilities 0-1. It is currently out of range.";
      return EXIT_FAILURE;
    }

  minMaxChecker->SetImage(clampGMToOneFilter->GetOutput());
  minMaxChecker->Compute();
  if (minMaxChecker->GetMinimum() < 0 || minMaxChecker->GetMaximum() > 1)
    {
      std::cerr <<"The WM + GM image should have probabilities 0-1. It is currently out of range.";
      return EXIT_FAILURE;
    }

  minMaxChecker->SetImage(boundaryFilter->GetOutput());
  minMaxChecker->Compute();
  if (minMaxChecker->GetMinimum() < 0 || minMaxChecker->GetMaximum() > 1)
    {
      std::cerr <<"The GWI image should be thresholded 0-1. It is currently out of range.";
      return EXIT_FAILURE;
    }

  std::cout << "Performing registration" << std::endl;
  
  // Base class pointer.
  typedef typename itk::ImageToImageFilter<InputImageType, InputImageType> BaseFilterType;
  typename BaseFilterType::Pointer basePointer;
  
  // The Big-But-Unfortunately-Old Big Daddy
  typedef typename itk::RegistrationBasedCTEFilter<InputImageType, ScalarType> OldRegistrationFilterType;
  typename OldRegistrationFilterType::Pointer oldRegistrationFilter = OldRegistrationFilterType::New();
  oldRegistrationFilter->SetWhitePlusGreyMatterPVMap(clampGMToOneFilter->GetOutput());
  oldRegistrationFilter->SetWhiteMatterPVMap(clampWMToOneFilter->GetOutput());           
  oldRegistrationFilter->SetThicknessPriorMap(tauImage);                      
  oldRegistrationFilter->SetGWI(boundaryFilter->GetOutput());
  oldRegistrationFilter->SetM(args.m);
  oldRegistrationFilter->SetLambda(args.lambda);
  oldRegistrationFilter->SetSigma(args.sigmaUpdate);
  oldRegistrationFilter->SetEpsilon(args.epsilon);
  oldRegistrationFilter->SetAlpha(args.alpha);
  oldRegistrationFilter->SetMaxIterations(args.maxIterations);

  // The Big-Shiny-And-New Big Daddy.
  typedef typename itk::RegistrationBasedCorticalThicknessFilter<InputImageType, ScalarType> NewRegistrationFilterType;
  typename NewRegistrationFilterType::Pointer registrationFilter = NewRegistrationFilterType::New();
  registrationFilter->SetWhitePlusGreyMatterPVMap(clampGMToOneFilter->GetOutput());
  registrationFilter->SetWhiteMatterPVMap(clampWMToOneFilter->GetOutput());        
  registrationFilter->SetThicknessPriorMap(tauImage);                      
  registrationFilter->SetGWI(boundaryFilter->GetOutput());                 
  registrationFilter->SetGreyMask(thresholdedGreyImage->GetOutput());      
  registrationFilter->SetM(args.m);
  registrationFilter->SetLambda(args.lambda);
  registrationFilter->SetUpdateSigma(args.sigmaUpdate);
  registrationFilter->SetDeformationSigma(args.sigmaDeformation);
  registrationFilter->SetEpsilon(args.epsilon);
  registrationFilter->SetAlpha(args.alpha);
  registrationFilter->SetMaxIterations(args.maxIterations);

  // Set base class pointer
  if (args.useOld)
    {
      basePointer = oldRegistrationFilter;
    }
  else
    {
      basePointer = registrationFilter;
    }
  basePointer->UpdateLargestPossibleRegion();
  
  typename GaussianFilterType::Pointer outputGaussianFilter = GaussianFilterType::New();
  outputGaussianFilter->SetInput(basePointer->GetOutput(0));
  outputGaussianFilter->SetVariance(args.sigmaThickness*args.sigmaThickness);
  outputGaussianFilter->SetMaximumKernelWidth(args.kernelWidth);
  
  typename MultiplyFilterType::Pointer multiplyOutputFilter = MultiplyFilterType::New();
  multiplyOutputFilter->SetInput1(outputGaussianFilter->GetOutput());
  multiplyOutputFilter->SetInput2(thresholdedGreyImage->GetOutput());

  typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();
  outputImageWriter->SetFileName(args.outputThicknessImage);

  if (args.blurOutput)
    {
      outputImageWriter->SetInput(multiplyOutputFilter->GetOutput());      
    }
  else
    {
      outputImageWriter->SetInput(basePointer->GetOutput(0));
    }
  
  // And Write the output.  
  outputImageWriter->Update();

  // Optionally write the transformed moving image, so we can check registration.
  if (args.outputTransformedMovingImage.length() > 0)
    {
      outputImageWriter->SetFileName(args.outputTransformedMovingImage);
      outputImageWriter->SetInput(basePointer->GetOutput(1));
      outputImageWriter->Update();      
    }
  
  // Optionall write the displacement image, so we can check vectors
  if (args.outputDisplacementImage.length() > 0)
    {
      typename OutputVectorImageWriterType::Pointer vectorWriter = OutputVectorImageWriterType::New();
      vectorWriter->SetFileName(args.outputDisplacementImage);
      vectorWriter->SetInput(registrationFilter->GetInterfaceDisplacementImage());
      vectorWriter->Update();
    }
  return 0;  
}

/**
 * \brief Implements S. R. Das et. al. NeuroImage 45 (2009) 867-879.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.white = 1;
  args.grey = 2;
  args.csf = 3;  
  args.alpha = 1.0;
  args.sigmaUpdate = 1.5;
  args.sigmaDeformation = 0;
  args.sigmaThickness = 0;
  args.epsilon = 0.0001;
  args.lambda = 1.0;
  args.tau = 6.0;
  args.m = 20;
  args.maxIterations = 100;
  args.resampleToIsotropic = false;
  args.rescalePVMaps = false;
  args.low = 0;
  args.high = 255;
  args.useThicknessPriorImage = false;
  args.useOld = false;
  args.blurOutput = false;
  args.kernelWidth = 10;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.labelImage=argv[++i];
      std::cout << "Set -g=" << args.labelImage << std::endl;
    }    
    else if(strcmp(argv[i], "-gmpv") == 0){
      args.gmpvmapImage=argv[++i];
      std::cout << "Set -g=" << args.gmpvmapImage << std::endl;
    }
    else if(strcmp(argv[i], "-wmpv") == 0){
      args.wmpvmapImage=argv[++i];
      std::cout << "Set -w=" << args.wmpvmapImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputThicknessImage=argv[++i];
      std::cout << "Set -o=" << args.outputThicknessImage << std::endl;
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
    else if(strcmp(argv[i], "-tpi") == 0){
      args.tauImage=argv[++i];
      args.useThicknessPriorImage = true;
      std::cout << "Set -tpi=" << args.tauImage << std::endl;
    }
    else if(strcmp(argv[i], "-tpc") == 0){
      args.tau=atof(argv[++i]);
      std::cout << "Set -tpc=" << niftk::ConvertToString(args.tau) << std::endl;
    }
    else if(strcmp(argv[i], "-mi") == 0){
      args.maxIterations=atoi(argv[++i]);
      std::cout << "Set -mi=" << niftk::ConvertToString(args.maxIterations) << std::endl;
    }
    else if(strcmp(argv[i], "-m") == 0){
      args.m=atoi(argv[++i]);
      std::cout << "Set -m=" << niftk::ConvertToString(args.m) << std::endl;
    }
    else if(strcmp(argv[i], "-l") == 0){
      args.lambda=atof(argv[++i]);
      std::cout << "Set -l=" << niftk::ConvertToString(args.lambda) << std::endl;
    }
    else if(strcmp(argv[i], "-su") == 0){
      args.sigmaUpdate=atof(argv[++i]);
      std::cout << "Set -su=" << niftk::ConvertToString(args.sigmaUpdate) << std::endl;
    }
    else if(strcmp(argv[i], "-sd") == 0){
      args.sigmaDeformation=atof(argv[++i]);
      std::cout << "Set -sd=" << niftk::ConvertToString(args.sigmaDeformation) << std::endl;
    }
    else if(strcmp(argv[i], "-e") == 0){
      args.epsilon=atof(argv[++i]);
      std::cout << "Set -e=" << niftk::ConvertToString(args.epsilon) << std::endl;
    }
    else if(strcmp(argv[i], "-a") == 0){
      args.alpha=atof(argv[++i]);
      std::cout << "Set -a=" << niftk::ConvertToString(args.alpha) << std::endl;
    }    
    else if(strcmp(argv[i], "-iso") == 0){
      args.resampleToIsotropic=true;
      std::cout << "Set -iso=" << niftk::ConvertToString(args.resampleToIsotropic) << std::endl;
    }
    else if(strcmp(argv[i], "-rescale") == 0){
      args.rescalePVMaps=true;
      args.low=atof(argv[++i]);
      args.high=atof(argv[++i]);
      std::cout << "Set -rescale=" << niftk::ConvertToString(args.rescalePVMaps) << " rescaling from low=" << niftk::ConvertToString(args.low) << ", and high=" << niftk::ConvertToString(args.high) << ", to [0-1]" << std::endl;
    }
    else if(strcmp(argv[i], "-old") == 0){
      args.useOld=true;
      std::cout << "Set -old=" << niftk::ConvertToString(args.useOld) << std::endl;
    }    
    else if(strcmp(argv[i], "-resampled") == 0){
      args.outputTransformedMovingImage=argv[++i];
      std::cout << "Set -resampled=" << args.outputTransformedMovingImage << std::endl;
    }
    else if(strcmp(argv[i], "-displacement") == 0){
      args.outputDisplacementImage=argv[++i];
      std::cout << "Set -displacement=" << args.outputDisplacementImage << std::endl;
    }
    else if(strcmp(argv[i], "-st") == 0){
      args.blurOutput = true;
      args.sigmaThickness=atof(argv[++i]);
      std::cout << "Set -st=" << niftk::ConvertToString(args.sigmaThickness) << std::endl;
    }
    else if(strcmp(argv[i], "-sw") == 0){
      args.blurOutput = true;
      args.kernelWidth=atoi(argv[++i]);
      std::cout << "Set -sw=" << niftk::ConvertToString(args.kernelWidth) << std::endl;
    }            
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }        
  }
  
  // Validate command line args
  if (args.labelImage.length() == 0 || args.gmpvmapImage.length() == 0 || args.wmpvmapImage.length() == 0 || args.outputThicknessImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if(args.sigmaUpdate < 0 ){
    std::cerr << argv[0] << "\tThe Gaussian sigma for the update field must be >= 0" << std::endl;
    return -1;
  }

  if(args.sigmaDeformation < 0 ){
    std::cerr << argv[0] << "\tThe Gaussian sigma for the deformation field must be >= 0" << std::endl;
    return -1;
  }

  if(args.sigmaThickness < 0 ){
    std::cerr << argv[0] << "\tThe Gaussian sigma for the thickness image must be >= 0" << std::endl;
    return -1;
  }

  if(args.epsilon < 0 ){
    std::cerr << argv[0] << "\tThe epsilon must be >= 0" << std::endl;
    return -1;
  }

  if(args.lambda < 0 ){
    std::cerr << argv[0] << "\tThe lambda must be >= 0" << std::endl;
    return -1;
  }

  if(args.m < 1 ){
    std::cerr << argv[0] << "\tVariable m must be >= 1" << std::endl;
    return -1;
  }

  if(args.alpha < 0 || args.alpha > 1 ){
    std::cerr << argv[0] << "\tVariable alpha must be 0 <= alpha <= 1" << std::endl;
    return -1;
  }

  if(args.maxIterations < 1 ){
    std::cerr << argv[0] << "\tVariable max iterations must be >= 1" << std::endl;
    return -1;
  }

  int dims = itk::PeekAtImageDimension(args.gmpvmapImage);
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
