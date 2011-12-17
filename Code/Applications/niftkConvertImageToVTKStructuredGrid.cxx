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
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vtkStructuredGrid.h"
#include "vtkStructuredGridWriter.h"
#include "vtkPoints.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkIndent.h"

void Usage(char *exec)
  {
	niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Transform's an image, as read by ITK, into a VTK structured grid." << std::endl;
    std::cout << "  This program was written because ITK only writes VTK structured points, which doesn't include direction cosines." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputImage.nii -o outputImage.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image, any format read by ITK, we recommend Nifti." << std::endl;
    std::cout << "    -o    <filename>        Output image, specifically in VTK structured grid format." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputFile;
  std::string outputFile;
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::Image< PixelType, 3> ImageType;
  
  try
  {
    
    typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
    imageReader->SetFileName(args.inputFile);
    imageReader->Update();
    
    typename InputImageType::Pointer inputImage = imageReader->GetOutput();
    typename InputImageType::SizeType inputSize = inputImage->GetLargestPossibleRegion().GetSize();
    typename InputImageType::SpacingType inputSpacing = inputImage->GetSpacing();
    typename InputImageType::PointType inputOrigin = inputImage->GetOrigin();
    typename InputImageType::DirectionType inputDirection = inputImage->GetDirection();
    typename InputImageType::IndexType inputIndex = inputImage->GetLargestPossibleRegion().GetIndex();
    
    // Force copy to 3D image 
    typename ImageType::SizeType outputSize;
    typename ImageType::SpacingType outputSpacing;
    typename ImageType::PointType outputOrigin;
    typename ImageType::DirectionType outputDirection;
    typename ImageType::IndexType outputIndex;
    
    for (int i = 0; i < Dimension; i++)
      {
        outputSize[i] = inputSize[i];
        outputSpacing[i] = inputSpacing[i];
        outputOrigin[i] = inputOrigin[i];
        outputIndex[i] = inputIndex[i];
        for (int j = 0; j < Dimension; j++)
          {
            outputDirection[i][j] = inputDirection[i][j];
          }
      }
    
    if (Dimension == 2)
      {
        outputSize[Dimension] = 1;
        outputSpacing[Dimension] = 1;
        outputOrigin[Dimension] = 0;
        outputIndex[Dimension] = 0;
        for (int i = 0; i < Dimension; i++)
          {
            outputDirection[i][Dimension] = 0;
          }
        outputDirection[Dimension][Dimension] = 1;
      }
    
    std::cout << "Input image size=" << inputSize << ", spacing=" << inputSpacing << ", origin=" << inputOrigin << ", index=" << inputIndex << ", inputDirection=\n" << inputDirection << std::endl;
    std::cout << "Output image size=" << outputSize << ", spacing=" << outputSpacing << ", origin=" << outputOrigin << ", index=" << outputIndex << ", outputDirection=\n" << outputDirection << std::endl;
    
    typename ImageType::RegionType outputRegion;
    outputRegion.SetSize(outputSize);
    outputRegion.SetIndex(outputIndex);
    
    typename ImageType::Pointer outputImage = ImageType::New();
    outputImage->SetRegions(outputRegion);
    outputImage->SetSpacing(outputSpacing);
    outputImage->SetOrigin(outputOrigin);
    outputImage->SetDirection(outputDirection);
    outputImage->Allocate();
    outputImage->FillBuffer(0);
    
    // Copy data to output image.
    itk::ImageRegionConstIterator<InputImageType> inputIterator(inputImage, inputImage->GetLargestPossibleRegion());
    itk::ImageRegionIteratorWithIndex<ImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
    
    for (inputIterator.GoToBegin(),
         outputIterator.GoToBegin();
         !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd();
         ++inputIterator,
         ++outputIterator)
      {
        outputIterator.Set(inputIterator.Get());
      }
    
    // Create VTK Structured Grid
    vtkStructuredGrid *grid = vtkStructuredGrid::New();
    grid->SetDimensions(outputSize[0], outputSize[1], outputSize[2]);
    
    vtkPoints *points = vtkPoints::New();
    points->SetDataTypeToFloat();
    points->Allocate(outputSize[0] * outputSize[1] * outputSize[2]);
    
    vtkFloatArray *scalars = vtkFloatArray::New();
    scalars->SetNumberOfComponents(1);
    scalars->SetNumberOfValues(outputSize[0] * outputSize[1] * outputSize[2]);
    
    typename ImageType::PointType point;
    typename ImageType::IndexType index;
    PixelType value;
    float vtkPoint[3];
    unsigned long int counter = 0;
    
    for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
      {
        value = outputIterator.Get();
        index = outputIterator.GetIndex();
        
        outputImage->TransformIndexToPhysicalPoint(index, point);
        
        vtkPoint[0] = point[0];
        vtkPoint[1] = point[1];
        vtkPoint[2] = point[2];
        
        points->InsertPoint(counter, vtkPoint);
        scalars->SetValue(counter, value);
        counter++;
      }
    
    grid->SetPoints(points);  
    grid->GetPointData()->SetScalars(scalars);
    
    // Write VTK Structured Grid
    vtkStructuredGridWriter *writer = vtkStructuredGridWriter::New();
    writer->SetInput(grid);
    writer->SetFileName(args.outputFile.c_str());
    writer->SetFileTypeToASCII();
    writer->Update();
    
    // Clean up
    points->Delete();
    scalars->Delete();
    grid->Delete();

  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  return EXIT_SUCCESS;
}

/**
 * \brief Transform's VTK poly data file by any number of affine transformations.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputFile=argv[++i];
      std::cout << "Set -i=" << args.inputFile<< std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputFile=argv[++i];
      std::cout << "Set -o=" << args.outputFile<< std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }      
  }

  // Validate command line args
  if (args.inputFile.length() == 0 || args.outputFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputFile);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputFile))
    {
    case itk::ImageIOBase::UCHAR:
      if (dims == 2)
        {
          result = DoMain<2, unsigned char>(args);  
        }
      else
        {
          result = DoMain<3, unsigned char>(args);
        }
      break;
    case itk::ImageIOBase::CHAR:
      if (dims == 2)
        {
          result = DoMain<2, char>(args);  
        }
      else
        {
          result = DoMain<3, char>(args);
        }
      break;
    case itk::ImageIOBase::USHORT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned short>(args);  
        }
      else
        {
          result = DoMain<3, unsigned short>(args);
        }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        {
          result = DoMain<2, short>(args);  
        }
      else
        {
          result = DoMain<3, short>(args);
        }
      break;
    case itk::ImageIOBase::UINT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned int>(args);  
        }
      else
        {
          result = DoMain<3, unsigned int>(args);
        }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 2)
        {
          result = DoMain<2, int>(args);  
        }
      else
        {
          result = DoMain<3, int>(args);
        }
      break;
    case itk::ImageIOBase::ULONG:
      if (dims == 2)
        {
          result = DoMain<2, unsigned long>(args);  
        }
      else
        {
          result = DoMain<3, unsigned long>(args);
        }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 2)
        {
          result = DoMain<2, long>(args);  
        }
      else
        {
          result = DoMain<3, long>(args);
        }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        {
          result = DoMain<2, float>(args);  
        }
      else
        {
          result = DoMain<3, float>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = DoMain<2, double>(args);  
        }
      else
        {
          result = DoMain<3, double>(args);
        }
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
  
} // end main
