/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileReader.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a 2D image, assumed to be a histogram of ints, and evaluates the joint entropy." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i image.nii [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image (histogram)" << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -bins <int>  [64]       Number of bins" << std::endl;
    std::cout << "    -normalise              Normalise data" << std::endl;
  }

struct arguments
{
  std::string inputImage;
  int bins;
  bool normalise;
};

/**
 * \brief Evaluate entropy of histogram.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.bins = 64;
  args.normalise = false;
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -i=" << args.inputImage;
    }
    else if(strcmp(argv[i], "-bins") == 0){
      args.bins=atoi(argv[++i]);
      std::cout << "Set -bins=" << niftk::ConvertToString(args.bins);
    }  
    else if(strcmp(argv[i], "-normalise") == 0){
      args.normalise=true;
      std::cout << "Set -normalise=" << niftk::ConvertToString(args.normalise);
    }        
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.inputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  typedef itk::Image<float,2> ImageType;
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageRegionConstIteratorWithIndex<ImageType> IteratorType;

  ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName(args.inputImage);
  try{
    imageReader->Update();
  }
  catch(itk::ExceptionObject  &err){
    std::cerr<<"Exception caught when reading the input image: "<< args.inputImage <<std::endl;
    std::cerr<<"Error: "<<err<<std::endl;
    return EXIT_FAILURE;
  }

  IteratorType iterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
  
  int bin = args.bins;
  
  double *histo = (double*)calloc(bin*bin,sizeof(double));
  ImageType::IndexType index;
  double voxelNumber=0;
  double voxelTotal=0;
  
  for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator){
    index = iterator.GetIndex();
    histo[index[1]*bin + index[0]] = iterator.Get();
    voxelNumber++;
    voxelTotal += iterator.Get();
  }

  if (args.normalise)
    {
      for(int i=0; i<bin*bin; i++) histo[i] /= voxelTotal;    
    }

  double fEntropy = 0.0;
  double sEntropy = 0.0;
  double jEntropy = 0.0;

  for(int t=0; t<bin; t++){
    double sum=0.0;
    int coord=t*bin;
    for(int r=0; r<bin; r++){
      sum += histo[coord++];
    }
    double logValue=0.0;
    if(sum) logValue = log(sum);
    fEntropy -= sum*logValue;
  }
  for(int r=0; r<bin; r++){
    double sum=0.0;
    int coord=r;
    for(int t=0; t<bin; t++){
      sum += histo[coord];
      coord += bin;
    }
    double logValue=0.0;
    if(sum) logValue = log(sum);
    sEntropy -= sum*logValue;
  }

  for(int tr=0; tr<bin*bin; tr++){
    double jointValue = histo[tr];
    double jointLog = 0.0;
    if(jointValue)  jointLog = log(jointValue);
    jEntropy -= jointValue*jointLog;
  }
  delete[] histo;

  std::cout << jEntropy << std::endl;
  
  return EXIT_SUCCESS;
}
