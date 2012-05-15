/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
 Revision          : $Revision: 7828 $
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
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAddOneLayerOfGreyMatterFilter.h"
#include "itkCastImageFilter.h"
#include "itkJonesThicknessFilter.h"
#include "itkSetGreyBoundaryToWhiteOrCSFFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkBinaryThresholdImageFilter.h"

/*!
 * \file niftkCTEHuttonLayering.cxx
 * \page niftkCTEHuttonLayering
 * \section niftkCTEHuttonLayeringSummary Implements Chloe Hutton's method for detecting sulcal CSF via adding layers of GM and iteratively calculating thickness.
 *
 * See section 'Preserving cortocal topography' as found in Hutton et. al. NeuroImage 2008 paper: doi:10.1016/j.neuroimage.2008.01.027
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements Chloe Hutton's method for detecting sulcal CSF via adding layers of GM and iteratively calculating thickness" << std::endl;
  std::cout << "  See section 'Preserving cortocal topography' as found in Hutton et. al. NeuroImage 2008 paper: doi:10.1016/j.neuroimage.2008.01.027" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i        <filename>        Input segmented image, with exactly 3 label values (GM, WM, CSF) " << std::endl;
  std::cout << "    -o        <filename>        Output segmented image, edited using GM layers, and CSF skeletonisation" << std::endl << std::endl;      
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -w        <float> [1]       Label for white matter" << std::endl;
  std::cout << "    -g        <float> [2]       Label for grey matter" << std::endl;
  std::cout << "    -c        <float> [3]       Label for extra-cerebral matter" << std::endl;  
  std::cout << "    -sigma    <float> [0]       Sigma for smoothing of vector normals." << std::endl;  
  std::cout << "    -maxIters <int>   [10]      Maximum number of layers to iterate" << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  short int grey;
  short int white;
  short int csf;
  unsigned long int maxIterations;
  double sigma;
  bool useSmoothing;
};

template <int Dimension> 
int DoMain(arguments args)
{
  typedef short int                                                                InputPixelType;
  typedef typename itk::Image< InputPixelType, Dimension >                         InputImageType;
  typedef typename InputImageType::Pointer                                         InputImagePointer;
  typedef typename itk::ImageFileReader< InputImageType  >                         InputImageReaderType;
  typedef float                                                                    ThicknessPixelType;
  typedef typename itk::Vector< ThicknessPixelType, Dimension >                    VectorPixelType;
  typedef typename itk::Image<VectorPixelType, Dimension>                          VectorImageType;
  typedef typename VectorImageType::Pointer                                        VectorImagePointer;
  typedef typename itk::Image< ThicknessPixelType, Dimension >                     ThicknessImageType;
  typedef typename itk::ImageFileWriter< InputImageType >                          OutputWriterType;
  typedef typename itk::AddOneLayerOfGreyMatterFilter<InputImageType>              OneLayerFilterType;
  typedef typename itk::CastImageFilter<InputImageType, ThicknessImageType>        CastFilterType;
  typedef typename itk::JonesThicknessFilter<ThicknessImageType, float, Dimension> JonesFilterType;
  typedef typename itk::SetGreyBoundaryToWhiteOrCSFFilter<InputImageType, ThicknessPixelType, Dimension> SetGreyBoundaryFilterType;
  typedef typename itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdFilterType;
  
  typename OneLayerFilterType::Pointer oneLayerFilter = OneLayerFilterType::New();
  typename CastFilterType::Pointer castFilter = CastFilterType::New();
  typename JonesFilterType::Pointer jonesFilter = JonesFilterType::New();
  typename SetGreyBoundaryFilterType::Pointer setBoundaryFilter = SetGreyBoundaryFilterType::New();
  typename BinaryThresholdFilterType::Pointer thresholdCSFFilter = BinaryThresholdFilterType::New();
  
  typename InputImageReaderType::Pointer  imageReader = InputImageReaderType::New();
  imageReader->SetFileName(  args.inputImage );
  

  try 
    { 
      std::cout << "Loading label image:" + args.inputImage << std::endl;
      imageReader->Update();
      std::cout << "Done" << std::endl;
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  typename InputImageType::SpacingType spacing = imageReader->GetOutput()->GetSpacing();
  
  double expectedVoxelSize = 0;
  for (unsigned int i = 0; i < Dimension; i++)
    {
      expectedVoxelSize += spacing[i]*spacing[i];
    }
  expectedVoxelSize = sqrt(expectedVoxelSize);
  std::cout << "Voxel size is " << niftk::ConvertToString(expectedVoxelSize) << std::endl;
  
  InputImagePointer inputImage = imageReader->GetOutput();
  
  // This is the one we modify as we go along, adding layers iteratively.
  InputImagePointer updatingLabelledImage = InputImageType::New();
  updatingLabelledImage->SetRegions(inputImage->GetLargestPossibleRegion());
  updatingLabelledImage->SetSpacing(inputImage->GetSpacing());
  updatingLabelledImage->SetDirection(inputImage->GetDirection());
  updatingLabelledImage->SetOrigin(inputImage->GetOrigin());
  updatingLabelledImage->Allocate();
  updatingLabelledImage->FillBuffer(0);

  // This is for the final output image.
  InputImagePointer labelledImage = InputImageType::New();
  labelledImage->SetRegions(inputImage->GetLargestPossibleRegion());
  labelledImage->SetSpacing(inputImage->GetSpacing());
  labelledImage->SetDirection(inputImage->GetDirection());
  labelledImage->SetOrigin(inputImage->GetOrigin());
  labelledImage->Allocate();
  labelledImage->FillBuffer(0);

  // This is to hold voxels that are marked as CSF.
  InputImagePointer csfImage = InputImageType::New();
  csfImage->SetRegions(inputImage->GetLargestPossibleRegion());
  csfImage->SetSpacing(inputImage->GetSpacing());
  csfImage->SetDirection(inputImage->GetDirection());
  csfImage->SetOrigin(inputImage->GetOrigin());
  csfImage->Allocate();
  csfImage->FillBuffer(0);

  InputPixelType csfTag = args.grey + args.white + args.csf;
  
  typename itk::ImageRegionConstIterator<InputImageType> inputImageIterator(inputImage, inputImage->GetLargestPossibleRegion());
  typename itk::ImageRegionIterator<InputImageType> copyImageIterator(updatingLabelledImage, updatingLabelledImage->GetLargestPossibleRegion());
  for (inputImageIterator.GoToBegin(), copyImageIterator.GoToBegin(); !inputImageIterator.IsAtEnd(); ++inputImageIterator, ++copyImageIterator)
    {
      copyImageIterator.Set(inputImageIterator.Get());
    }
  
  unsigned int numberOfLoopIterations = 0;  
  unsigned long int differenceInNumberOfGreyBetweenIterations = 1;
  
  while (numberOfLoopIterations < args.maxIterations && differenceInNumberOfGreyBetweenIterations > 0)
    {
      oneLayerFilter->SetInput(updatingLabelledImage);
      oneLayerFilter->SetLabelThresholds(args.grey, args.white, args.csf);
      oneLayerFilter->Modified();
      oneLayerFilter->UpdateLargestPossibleRegion();
       
      if (oneLayerFilter->GetNumberOfGreyInBoundaryLayer() <= 0)
        {
          std::cout << "No grey voxels left to check, so finishing" << std::endl;
          break;
        }

      castFilter->SetInput(oneLayerFilter->GetOutput());
      castFilter->Modified();
      
      jonesFilter->SetInput(castFilter->GetOutput());
      jonesFilter->SetLowVoltage(0);
      jonesFilter->SetHighVoltage(10000);
      jonesFilter->SetLaplaceEpsionRatio(0.00001);
      jonesFilter->SetLaplaceMaxIterations(500);
      jonesFilter->SetWhiteMatterLabel(args.white);
      jonesFilter->SetGreyMatterLabel(args.grey);
      jonesFilter->SetCSFMatterLabel(args.csf);
      jonesFilter->SetMinimumStepSize(0.1);
      jonesFilter->SetMaximumLength(10);
      jonesFilter->SetSigma(args.sigma);
      jonesFilter->SetUseLabels(true);
      jonesFilter->SetUseSmoothing(args.useSmoothing);
      jonesFilter->Modified();      
      
      setBoundaryFilter->SetLabelImage(updatingLabelledImage);
      setBoundaryFilter->SetOneLayerImage(oneLayerFilter->GetOutput());
      setBoundaryFilter->SetThicknessImage(jonesFilter->GetOutput());
      setBoundaryFilter->SetLabelThresholds(args.grey, args.white, args.csf);
      setBoundaryFilter->SetExpectedVoxelSize(expectedVoxelSize);      
      setBoundaryFilter->Modified();
      setBoundaryFilter->SetTaggedCSFLabel(csfTag);
      setBoundaryFilter->UpdateLargestPossibleRegion();
      
      differenceInNumberOfGreyBetweenIterations = setBoundaryFilter->GetNumberOfGreyBefore() - setBoundaryFilter->GetNumberOfGreyAfter();
      
      std::cout << "differenceInNumberOfGreyBetweenIterations=" << niftk::ConvertToString((int)differenceInNumberOfGreyBetweenIterations) << std::endl;
      
      typename itk::ImageRegionConstIterator<InputImageType> boundaryFilterIterator(setBoundaryFilter->GetOutput(), setBoundaryFilter->GetOutput()->GetLargestPossibleRegion());
      typename itk::ImageRegionIterator<InputImageType> labelIterator(updatingLabelledImage, updatingLabelledImage->GetLargestPossibleRegion());
      typename itk::ImageRegionIterator<InputImageType> csfIterator(csfImage, csfImage->GetLargestPossibleRegion());
      
      for (boundaryFilterIterator.GoToBegin(), 
           labelIterator.GoToBegin(),
           csfIterator.GoToBegin(); 
           !boundaryFilterIterator.IsAtEnd(); 
           ++csfIterator,
           ++boundaryFilterIterator, 
           ++labelIterator)
        {
          if (boundaryFilterIterator.Get() == csfTag)
            {
              csfIterator.Set(1);
              labelIterator.Set(args.white);
            }
          else
            {
              labelIterator.Set(boundaryFilterIterator.Get());    
            }
        }

      numberOfLoopIterations++;
      
      std::cout << "loopIterations=" << niftk::ConvertToString((int)numberOfLoopIterations) \
          << ", maxIterations=" << niftk::ConvertToString((int)args.maxIterations) << std::endl;
      
    } // end while 

  typename itk::ImageRegionConstIterator<InputImageType> editedIterator(updatingLabelledImage, updatingLabelledImage->GetLargestPossibleRegion());
  typename itk::ImageRegionConstIterator<InputImageType> originalIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
  typename itk::ImageRegionConstIterator<InputImageType> csfIterator(csfImage, csfImage->GetLargestPossibleRegion());
  typename itk::ImageRegionIterator<InputImageType> outputIterator(labelledImage, labelledImage->GetLargestPossibleRegion());

  InputPixelType outputPixel;
  
  // We want the original white, the CSF from the edited volume, and otherwise grey.
  for (editedIterator.GoToBegin(),
       originalIterator.GoToBegin(),
       outputIterator.GoToBegin(),
       csfIterator.GoToBegin();
       !editedIterator.IsAtEnd();
       ++editedIterator,
       ++originalIterator,
       ++outputIterator,
       ++csfIterator)
    {
      if (originalIterator.Get() == args.white)
        {
          outputPixel = args.white;
        }
      else if (editedIterator.Get() == args.csf || csfIterator.Get() == 1)
        {
          outputPixel = args.csf;
        }
      else
        {
          outputPixel = args.grey; 
        }
      outputIterator.Set(outputPixel); 
    }
  
  typename OutputWriterType::Pointer imageWriter = OutputWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(labelledImage);
  
  try
  {
    std::cout << "Saving label image:" + args.outputImage << std::endl;
    imageWriter->Update(); 
    std::cout << "Done" << std::endl;
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  return EXIT_SUCCESS; 
}

/**
 * \brief Implements Chloe Hutton's GM/CSF editing routines.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.white = 1;
  args.grey = 2;
  args.csf = 3;
  args.maxIterations = 10;
  args.sigma = 1.5;
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
    else if(strcmp(argv[i], "-maxIters") == 0){
      args.maxIterations=atoi(argv[++i]);
      std::cout << "Set -maxIters=" << niftk::ConvertToString((int)args.maxIterations) << std::endl;
    }       
    else if(strcmp(argv[i], "-sigma") == 0){
      args.sigma=atof(argv[++i]);
      args.useSmoothing = true;
      std::cout << "Set -sigma=" << niftk::ConvertToString(args.sigma) << std::endl;
    }            
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }        
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
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

