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
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"
#include "vtkStructuredPoints.h"
#include "vtkStructuredPointsWriter.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"

/*!
 * \file niftkGradientVectorField.cxx
 * \page niftkGradientVectorField
 * \section niftkGradientVectorFieldSummary Take the gradient of an image, and outputs a vector image. Mainly used for generating test images.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Take the gradient of an image, and outputs a vector image. Mainly used for generating test images." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i                 : Input scalar image" << std::endl;
    std::cout << "    -o                 : Output vector image" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -method [int]      : Where:" << std::endl;
    std::cout << "                         0 - Default, use Central Differences" << std::endl;
    std::cout << "                         1 - Use ITKs itkGradientImageFilter" << std::endl;
    std::cout << "                         2 - Use ITKs itkGradientRecursiveGaussianImageFilter" << std::endl;
    std::cout << "    -dx <filename>     : Write the gradient in x direction" << std::endl;
    std::cout << "    -dy <filename>     : Write the gradient in y direction" << std::endl;
    std::cout << "    -dz <filename>     : Write the gradient in z direction" << std::endl;
    std::cout << "    -norm              : Normalize. Default off." << std::endl;
    std::cout << "    -sigma <float> [2] : Sigma to smooth vectors with" << std::endl;
    return;
  }

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  std::string dxImage;
  std::string dyImage;
  std::string dzImage;
  int method;
  bool normalize;
  double sigma;
};

template <int Dimension>
int DoMain(arguments args)
{

  typedef float PixelType;
  typedef typename itk::Vector<PixelType, Dimension>                VectorPixelType;
  typedef typename itk::Image<VectorPixelType, Dimension>           VectorImageType;
  typedef typename VectorImageType::Pointer                         VectorImagePointer;
  
  typedef typename itk::Image< PixelType, Dimension >               InputImageType;
  typedef typename itk::ImageFileReader< InputImageType >           InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType >           InputImageWriterType;
  typedef typename itk::ImageFileWriter< VectorImageType >          OutputVectorImageWriterType;
  
  typename InputImageReaderType::Pointer  imageReader = InputImageReaderType::New();
  imageReader->SetFileName( args.inputImage );

  try
  {
    imageReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    return EXIT_FAILURE;
  }
  
  // Copy it into a vector image
  VectorImagePointer finalImage = VectorImageType::New();
  finalImage->SetRegions(imageReader->GetOutput()->GetLargestPossibleRegion());
  finalImage->SetOrigin(imageReader->GetOutput()->GetOrigin());
  finalImage->SetSpacing(imageReader->GetOutput()->GetSpacing());
  finalImage->SetDirection(imageReader->GetOutput()->GetDirection());
  finalImage->Allocate();
  VectorPixelType zero;
  
  zero.Fill(0);
  finalImage->FillBuffer(zero);
  
  typedef typename itk::ScalarImageToNormalizedGradientVectorImageFilter< InputImageType, PixelType> FilterType;
  typename FilterType::Pointer gradientFilter  = FilterType::New();

  gradientFilter->SetInput(imageReader->GetOutput());
  gradientFilter->SetUseMillimetreScaling(true);
  gradientFilter->SetDivideByTwo(true);
  gradientFilter->SetNormalize(args.normalize);
  gradientFilter->SetPadValue(0);
  gradientFilter->SetDerivativeType((typename FilterType::DerivativeType)args.method);
  gradientFilter->SetSigma(args.sigma);
  gradientFilter->Update();

  vtkStructuredPoints *image = vtkStructuredPoints::New();
  vtkFloatArray *vectors = vtkFloatArray::New();
  vtkStructuredPointsWriter *writer = vtkStructuredPointsWriter::New();

  typename itk::ImageRegionConstIterator<VectorImageType> filterOutputIterator(gradientFilter->GetOutput(), gradientFilter->GetOutput()->GetLargestPossibleRegion());
  typename itk::ImageRegionIterator<VectorImageType> vectorIterator(finalImage, finalImage->GetLargestPossibleRegion());
  for (filterOutputIterator.GoToBegin(), vectorIterator.GoToBegin();
       !filterOutputIterator.IsAtEnd();
       ++filterOutputIterator, ++vectorIterator)
    {
        vectorIterator.Set(filterOutputIterator.Get());
    }

  if (args.outputImage.substr(args.outputImage.size()-3, 3) == "vtk")
    {
      typename VectorImageType::SizeType size = finalImage->GetLargestPossibleRegion().GetSize();
      typename VectorImageType::SpacingType spacing = finalImage->GetSpacing();
      typename VectorImageType::PointType origin = finalImage->GetOrigin();
      
      unsigned long int numberOfVectors = 0;
      
      if (Dimension == 2)
        {
          image->SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, 0); 
          image->SetOrigin(origin[0], origin[1], 0);
          image->SetSpacing(spacing[0], spacing[1], 1);
          numberOfVectors = size[0]*size[1];
        }
      else
        {
          image->SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1);
          image->SetOrigin(origin[0], origin[1], origin[2]);
          image->SetSpacing(spacing[0], spacing[1], spacing[2]);
          numberOfVectors = size[0]*size[1]*size[2];
        }
      image->SetScalarTypeToFloat();
      
      vectors->SetNumberOfComponents(3);
      vectors->Allocate(numberOfVectors);
      
      unsigned long int i = 0;
      for (vectorIterator.GoToBegin(); !vectorIterator.IsAtEnd(); ++vectorIterator)
        {
          if (Dimension == 2)
            {
              vectors->InsertComponent(i, 0, vectorIterator.Get()[0]);
              vectors->InsertComponent(i, 1, vectorIterator.Get()[1]);
              vectors->InsertComponent(i, 2, 0);              
            }
          else
            {
              vectors->SetTuple3(i, vectorIterator.Get()[0], vectorIterator.Get()[1], vectorIterator.Get()[2]);
            }
          i++;
        }

      image->GetPointData()->SetVectors(vectors);
      
      writer->SetFileName(args.outputImage.c_str());
      writer->SetInput(image);
      writer->SetFileTypeToASCII();
      writer->Update();
    }
  else
    {
      typename OutputVectorImageWriterType::Pointer imageVectorWriter = OutputVectorImageWriterType::New();
      imageVectorWriter->SetFileName( args.outputImage );
      imageVectorWriter->SetInput(finalImage);
      try
      {
        imageVectorWriter->Update();
      }
      catch( itk::ExceptionObject & err )
      {
        std::cerr << "Failed: " << err << std::endl;
        return EXIT_FAILURE;
      }      
    }

  

  std::cout << "Done main volume" << std::endl;
  
  if (args.dxImage.length() > 0)
    {
      
      typename InputImageType::Pointer tmpImage = InputImageType::New();
      tmpImage->SetRegions(imageReader->GetOutput()->GetLargestPossibleRegion());
      tmpImage->SetOrigin(imageReader->GetOutput()->GetOrigin());
      tmpImage->SetSpacing(imageReader->GetOutput()->GetSpacing());
      tmpImage->SetDirection(imageReader->GetOutput()->GetDirection());
      tmpImage->Allocate();
      tmpImage->FillBuffer(0);
      
      typename itk::ImageRegionIterator<VectorImageType> vectorIterator(finalImage, finalImage->GetLargestPossibleRegion());
      typename itk::ImageRegionIterator<InputImageType> scalarIterator(tmpImage, tmpImage->GetLargestPossibleRegion());
      for (vectorIterator.GoToBegin(), scalarIterator.GoToBegin(); !vectorIterator.IsAtEnd(); ++vectorIterator, ++scalarIterator)
        {
          scalarIterator.Set(vectorIterator.Get()[0]);
        }
      typename InputImageWriterType::Pointer scalarWriter = InputImageWriterType::New();
      scalarWriter->SetInput(tmpImage);
      scalarWriter->SetFileName(args.dxImage);
      scalarWriter->Update();
      
      std::cout << "Done dx volume" << std::endl;
    }

  if (args.dyImage.length() > 0)
    {
      
      typename InputImageType::Pointer tmpImage = InputImageType::New();
      tmpImage->SetRegions(imageReader->GetOutput()->GetLargestPossibleRegion());
      tmpImage->SetOrigin(imageReader->GetOutput()->GetOrigin());
      tmpImage->SetSpacing(imageReader->GetOutput()->GetSpacing());
      tmpImage->SetDirection(imageReader->GetOutput()->GetDirection());
      tmpImage->Allocate();
      tmpImage->FillBuffer(0);
      
      typename itk::ImageRegionIterator<VectorImageType> vectorIterator(finalImage, finalImage->GetLargestPossibleRegion());
      typename itk::ImageRegionIterator<InputImageType> scalarIterator(tmpImage, tmpImage->GetLargestPossibleRegion());
      for (vectorIterator.GoToBegin(), scalarIterator.GoToBegin(); !vectorIterator.IsAtEnd(); ++vectorIterator, ++scalarIterator)
        {
          scalarIterator.Set(vectorIterator.Get()[1]);
        }
      typename InputImageWriterType::Pointer scalarWriter = InputImageWriterType::New();
      scalarWriter->SetInput(tmpImage);
      scalarWriter->SetFileName(args.dyImage);
      scalarWriter->Update();
      
      std::cout << "Done dy volume" << std::endl;
    }

  if (args.dzImage.length() > 0)
    {
      
      typename InputImageType::Pointer tmpImage = InputImageType::New();
      tmpImage->SetRegions(imageReader->GetOutput()->GetLargestPossibleRegion());
      tmpImage->SetOrigin(imageReader->GetOutput()->GetOrigin());
      tmpImage->SetSpacing(imageReader->GetOutput()->GetSpacing());
      tmpImage->SetDirection(imageReader->GetOutput()->GetDirection());
      tmpImage->Allocate();
      tmpImage->FillBuffer(0);
      
      typename itk::ImageRegionIterator<VectorImageType> vectorIterator(finalImage, finalImage->GetLargestPossibleRegion());
      typename itk::ImageRegionIterator<InputImageType> scalarIterator(tmpImage, tmpImage->GetLargestPossibleRegion());
      for (vectorIterator.GoToBegin(), scalarIterator.GoToBegin(); !vectorIterator.IsAtEnd(); ++vectorIterator, ++scalarIterator)
        {
          scalarIterator.Set(vectorIterator.Get()[2]);
        }
      typename InputImageWriterType::Pointer scalarWriter = InputImageWriterType::New();
      scalarWriter->SetInput(tmpImage);
      scalarWriter->SetFileName(args.dzImage);
      scalarWriter->Update();
      
      std::cout << "Done dz volume" << std::endl;
    }

  return EXIT_SUCCESS;
}

/**
 * \brief Take the abs value of an image, for displaying the image properly in Midas.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.method = 0;
  args.normalize = false;
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
    else if(strcmp(argv[i], "-dx") == 0){
      args.dxImage=argv[++i];
      std::cout << "Set -dx=" << args.dxImage << std::endl;
    }
    else if(strcmp(argv[i], "-dy") == 0){
      args.dyImage=argv[++i];
      std::cout << "Set -dy=" << args.dyImage << std::endl;
    }
    else if(strcmp(argv[i], "-dz") == 0){
      args.dzImage=argv[++i];
      std::cout << "Set -dz=" << args.dzImage << std::endl;
    }            
    else if(strcmp(argv[i], "-method") == 0){
      args.method=atoi(argv[++i]);
      std::cout << "Set -method=" << niftk::ConvertToString(args.method) << std::endl;
    }
    else if(strcmp(argv[i], "-sigma") == 0){
      args.sigma=atof(argv[++i]);
      std::cout << "Set -sigma=" << niftk::ConvertToString(args.sigma) << std::endl;
    }      
    else if(strcmp(argv[i], "-norm") == 0){
      args.normalize = true;
      std::cout << "Set -norm=" << niftk::ConvertToString(args.normalize) << std::endl;
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

  if (args.method < 0 || args.method > 2)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;      
    }
  
  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

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
