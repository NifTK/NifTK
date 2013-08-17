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
#include <itkVector.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkJonesThicknessFilter.h>

/*!
 * \file niftkCTEJones2000.cxx
 * \page niftkCTEJones2000
 * \section niftkCTEJones2000Summary Implements Jones et. al. Human Brain Mapping 11:12-32(2000), with optional Gauss-Seidel optimisation.
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements Jones et. al. Human Brain Mapping 11:12-32(2000), with optional Gauss-Seidel optimisation," << std::endl;
  std::cout << "  and implements Diep et. al ISBI 2007 to cope with anisotropic voxel sizes. " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i     <filename>        Input segmented image, with exactly 3 label values (GM, WM, CSF) " << std::endl;
  std::cout << "    -o     <filename>        Output distance image" << std::endl << std::endl;      
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -w     <float> [1]       Label for white matter" << std::endl;
  std::cout << "    -g     <float> [2]       Label for grey matter" << std::endl;
  std::cout << "    -c     <float> [3]       Label for extra-cerebral matter" << std::endl;  
  std::cout << "    -low   <float> [0]       Low Potential (voltage)" << std::endl;
  std::cout << "    -high  <float> [10000]   High Potential (voltage)" << std::endl;
  std::cout << "    -le    <float> [0.00001] Laplacian relaxation convergence ratio (epsilon)" << std::endl;
  std::cout << "    -li    <int>   [200]     Laplacian relaxation max iterations" << std::endl;
  std::cout << "    -step  <float> [0.1]     Step size for integration" << std::endl;
  std::cout << "    -sigma <float> [0]       Sigma for smoothing of vector normals. Default 0 (i.e. off)." << std::endl;
  std::cout << "    -max   <float> [10]      Max length for integration (so ray casting doesn't continue forever)." << std::endl;
  std::cout << "    -noOpt                   Don't use Gauss-Siedel optimisation" << std::endl; 
  std::cout << "    -label                   When integrating, use the label/segmentation images not the laplacian." << std::endl;
  std::cout << "                             So boundaries are defined in terms of GM/WM/CSF labels." << std::endl;
  std::cout << "                             This means that the label value must be WM < GM < CSF, i.e. the CSF is the high potential surface." << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  double low;
  double high;
  double laplaceRatio;
  short int grey;
  short int white;
  short int csf;
  double step;
  double maxLength;
  double sigma;
  int laplaceIters;
  bool dontUseGaussSeidel;
  bool useLabel;
  bool useSmoothing;
};

template <int Dimension> 
int DoMain(arguments args)
{
  typedef  float ScalarType;
  typedef  float OutputScalarType;
  
  typedef typename itk::Image< ScalarType, Dimension >       InputImageType;
  typedef typename itk::Vector< ScalarType, Dimension >      VectorType;
  typedef typename itk::Image< VectorType, Dimension >       VectorImageType;
  typedef typename itk::Image< OutputScalarType, Dimension > OutputImageType;  
  typedef typename itk::ImageFileReader< InputImageType  >   InputImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >   OutputImageWriterType;
  typedef typename itk::JonesThicknessFilter<OutputImageType, ScalarType, Dimension> JonesThicknessFilterType;
  
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

  typename JonesThicknessFilterType::Pointer filter = JonesThicknessFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  filter->SetLowVoltage(args.low);
  filter->SetHighVoltage(args.high);
  filter->SetLaplaceEpsionRatio(args.laplaceRatio);
  filter->SetLaplaceMaxIterations(args.laplaceIters);
  filter->SetWhiteMatterLabel(args.white);
  filter->SetGreyMatterLabel(args.grey);
  filter->SetCSFMatterLabel(args.csf);
  filter->SetMinimumStepSize(args.step);
  filter->SetMaximumLength(args.maxLength);
  filter->SetSigma(args.sigma);
  filter->SetUseLabels(args.useLabel);
  filter->SetUseSmoothing(args.useSmoothing);
  
  // And Write the output.
  typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
  outputImageWriter->SetFileName(args.outputImage);
  outputImageWriter->SetInput(filter->GetOutput());
  outputImageWriter->Update();

  return 0;    
}

/**
 * \brief Implements Jones et. al. Human Brain Mapping 2005 11:12-32(2000)
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.low = 0;
  args.high = 10000;
  args.laplaceRatio = 0.00001;
  args.laplaceIters = 200;
  args.white = 1;
  args.grey = 2;
  args.csf = 3;
  args.step = 0.1;
  args.dontUseGaussSeidel = false;
  args.maxLength = 10;
  args.sigma = 0;
  args.useLabel = false;
  args.useSmoothing = false;
  

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
    else if(strcmp(argv[i], "-c") == 0){
      args.csf=atoi(argv[++i]);
      std::cout << "Set -c=" << niftk::ConvertToString(args.csf) << std::endl;
    }
    else if(strcmp(argv[i], "-g") == 0){
      args.grey=atoi(argv[++i]);
      std::cout << "Set -g=" << niftk::ConvertToString(args.grey) << std::endl;
    }
    else if(strcmp(argv[i], "-w") == 0){
      args.white=atoi(argv[++i]);
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
    else if(strcmp(argv[i], "-li") == 0){
      args.laplaceIters=atoi(argv[++i]);
      std::cout << "Set -li=" << niftk::ConvertToString(args.laplaceIters) << std::endl;
    }    
    else if(strcmp(argv[i], "-step") == 0){
      args.step=atof(argv[++i]);
      std::cout << "Set -step=" << niftk::ConvertToString(args.step) << std::endl;
    }
    else if(strcmp(argv[i], "-sigma") == 0){
      args.sigma=atof(argv[++i]);
      args.useSmoothing=true;
      std::cout << "Set -sigma=" << niftk::ConvertToString(args.sigma) << std::endl;
    }    
    else if(strcmp(argv[i], "-max") == 0){
      args.maxLength=atof(argv[++i]);
      std::cout << "Set -max=" << niftk::ConvertToString(args.maxLength) << std::endl;
    }
    else if(strcmp(argv[i], "-noOpt") == 0){
      args.dontUseGaussSeidel=true;
      std::cout << "Set -noOpt=" << niftk::ConvertToString(args.dontUseGaussSeidel) << std::endl;
    }
    else if(strcmp(argv[i], "-label") == 0){
      args.useLabel=true;
      std::cout << "Set -label=" << niftk::ConvertToString(args.useLabel) << std::endl;
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
    std::cerr << argv[0] << "\tThe iterations must be >= 1" << std::endl;
    return -1;
  }

  if(args.laplaceRatio < 0 ){
    std::cerr << argv[0] << "\tThe epsilon must be > 0" << std::endl;
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
