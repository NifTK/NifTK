/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-10-29 13:28:32 +0100 (Fri, 29 Oct 2010) $
 Revision          : $Revision: 4088 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVTKImageIO.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkErrorCode.h"
#include "itkRecursiveGaussianImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkGrayscaleDilateImageFilter.h"

int main(int argc, char** argv)
{
  vtkPolyDataReader* polyDataReader = vtkPolyDataReader::New();
  
  polyDataReader->SetFileName(argv[1]); 
  polyDataReader->Update(); 
  if (polyDataReader->GetErrorCode() != 0)
  {
    std::cout << vtkErrorCode::GetStringFromErrorCode(polyDataReader->GetErrorCode()) << std::endl; 
    return EXIT_FAILURE; 
  }
  vtkPolyData* polyData = polyDataReader->GetOutput(); 
      
  vtkIdType numberOfPoints = polyData->GetNumberOfPoints(); 
  std::cout << "# of points=" << numberOfPoints << std::endl; 
  int numberOfArrays = polyData->GetPointData()->GetNumberOfArrays(); 
  std::cout << "Number of arrays=" << numberOfArrays << std::endl; 
  for (int j = 0; j < numberOfArrays; j++)
    std::cout << "Array name=" << polyData->GetPointData()->GetArrayName(j) << std::endl; 
  
  typedef float TScalar; 
  const int TDimension = 3; 
  typedef itk::Image<TScalar,TDimension> ImageType;
  ImageType::Pointer inputImage; 
  itk::ImageFileReader<ImageType>::Pointer reader;
  
  // Reading it in. 
  try
  {
    reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(argv[2]);
    reader->Update();
    inputImage = reader->GetOutput(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cout << "ExceptionObject caught !" << std::endl; 
    std::cout << err << std::endl; 
    return EXIT_FAILURE;
  } 
  
  ImageType::PointType origin; 
  ImageType::DirectionType direction; 
  ImageType::RegionType::SizeType regionSize = inputImage->GetLargestPossibleRegion().GetSize(); 
  
  // VTK points are in a coordinate system with origin (0,0,0). 
  origin = inputImage->GetOrigin(); 
  std::cout << "original origin=" << origin[0] << "," << origin[1] << "," << origin[2] << std::endl; 
  origin[0] = 0.0; 
  origin[1] = 0.0; 
  origin[2] = 0.0; 
  inputImage->SetOrigin(origin); 
  
  // VTK points are in a coordinate system with the following orientations. 
  direction = inputImage->GetDirection(); 
  std::cout << "original direction=" << std::endl << direction; 
  if (argc >= 5)
  {
    if (strcmp("SNT",argv[4]) == 0)
    {
      std::cout << "Map to SNT mask" << std::endl; 
      direction[0][0] = 1;
      direction[0][1] = 0;
      direction[0][2] = 0;
      direction[1][0] = 0;
      direction[1][1] = 1;
      direction[1][2] = 0;
      direction[2][0] = 0;
      direction[2][1] = 0;
      direction[2][2] = 1;
      inputImage->SetDirection(direction); 
    }
    else if (strcmp("T1",argv[4]) == 0)
    {
      std::cout << "Map to T1" << std::endl; 
      direction[0][0] = 0;
      direction[0][1] = 0;
      direction[0][2] = -1;
      direction[1][0] = 0;
      direction[1][1] = -1;
      direction[1][2] = 0;
      direction[2][0] = -1;
      direction[2][1] = 0;
      direction[2][2] = 0;
      inputImage->SetDirection(direction); 
    }
  }
  direction = inputImage->GetDirection(); 
  std::cout << "new direction=" << std::endl << direction; 
  
  inputImage->FillBuffer(0.0); 
  
  // Threhsold for p-values. 
  double threshold = atof(argv[5]); 
  std::cout << "p-value threshold=" << threshold << std::endl;
  
  // Which array? 
  const char* arrayName = argv[6]; 
  std::cout << "array name=" << arrayName << std::endl;
  
  // Weight file. 
  std::string weightFilename; 
  std::fstream weightFile; 
  std::string blurOption; 
  std::string blurFlag; 
  
  if (argc >= 8) 
  {
    weightFilename = argv[7]; 
    weightFile.open(weightFilename.c_str()); 
  }
  if (argc >= 10)
  {
    blurOption = argv[8]; 
    blurFlag = argv[9];
  }
  
  for (vtkIdType i = 0; i < numberOfPoints; i++)
  {
    double* coords = polyData->GetPoints()->GetPoint(i); 
    ImageType::PointType point; 
    ImageType::IndexType index; 
    
    // std::cout << "point=" << coords[0] << "," << coords[1] << "," << coords[2] << std::endl; 
    point[0] = coords[0]; 
    point[1] = coords[1]; 
    point[2] = coords[2]; 
    inputImage->TransformPhysicalPointToIndex(point, index); 
    index[0] = (index[0]+regionSize[0])%regionSize[0]; 
    index[1] = (index[1]+regionSize[1])%regionSize[1]; 
    index[2] = (index[2]+regionSize[2])%regionSize[2]; 
    // std::cout << "index=" << index[0] << "," << index[1] << "," << index[2] << std::endl; 
    
    if (numberOfArrays > 0)
    {
      if (polyData->GetPointData()->GetArray(arrayName)->GetComponent(i, 0) <= threshold)
        inputImage->SetPixel(index, 9999.0); 
    }
    else
    {
      // std::cout << "scalar=" << polyData->GetPointData()->GetScalars()->GetComponent(i, 0) << std::endl; 
      if (weightFilename.length() <= 0)
      {
        if (polyData->GetPointData()->GetScalars()->GetComponent(i, 0) <= threshold)
          inputImage->SetPixel(index, 9999.0); 
      }
      else
      {
        float weight; 
        
        weightFile >> weight; 
        inputImage->SetPixel(index, weight); 
        //std::cout << "weight=" << weight << std::endl; 
      }
    }
  }
  
  ImageType::Pointer outputImage = inputImage; 
  
  if (weightFilename.length() >= 0)
  {
    typedef itk::ImageRegionIterator<ImageType> IteratorType; 
    typedef itk::BinaryCrossStructuringElement<TScalar, TDimension> StructuringElementType;
    typedef itk::GrayscaleDilateImageFilter<ImageType, ImageType, StructuringElementType> GrayscaleDilateImageFilterType; 
    GrayscaleDilateImageFilterType::Pointer grayscaleDilateImageFilter = GrayscaleDilateImageFilterType::New(); 
    StructuringElementType structuringElement;
    
    if (blurOption == "dilation")
    {
      std::cout << "Performing dilation..." << std::endl; 
      structuringElement.SetRadius(atoi(blurFlag.c_str())); // 3x3 structuring element
      structuringElement.CreateStructuringElement();      
      grayscaleDilateImageFilter->SetInput(inputImage); 
      grayscaleDilateImageFilter->SetKernel(structuringElement); 
      grayscaleDilateImageFilter->Update(); 
      outputImage = grayscaleDilateImageFilter->GetOutput(); 
      
      // Keep the original weights. 
      IteratorType inputIt(inputImage, inputImage->GetLargestPossibleRegion()); 
      IteratorType outputIt(outputImage, outputImage->GetLargestPossibleRegion()); 
      
      for (inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++outputIt)
      {
        if (inputIt.Get() > 1.0e-10)
        {
          outputIt.Set(inputIt.Get()); 
        }
      }
      
    }
    else if (blurOption == "gaussian")
    {
      std::cout << "Performing gaussian blurring..." << std::endl; 
      typedef itk::RecursiveGaussianImageFilter<ImageType, ImageType> GaussianBlurFilterType; 
      GaussianBlurFilterType::Pointer gaussianBlurFilter = GaussianBlurFilterType::New(); 
      
      gaussianBlurFilter->SetInput(inputImage); 
      gaussianBlurFilter->SetSigma(atof(blurFlag.c_str())); 
      gaussianBlurFilter->Update(); 
      outputImage = gaussianBlurFilter->GetOutput(); 
      
      structuringElement.SetRadius(1); // 3x3 structuring element
      structuringElement.CreateStructuringElement();      
      grayscaleDilateImageFilter->SetInput(inputImage); 
      grayscaleDilateImageFilter->SetKernel(structuringElement); 
      grayscaleDilateImageFilter->Update(); 
      
      // Don't let it spreadout too much. 
      IteratorType outputIt(outputImage, outputImage->GetLargestPossibleRegion()); 
      IteratorType maskIt(grayscaleDilateImageFilter->GetOutput(), grayscaleDilateImageFilter->GetOutput()->GetLargestPossibleRegion()); 
      
      for (outputIt.GoToBegin(), maskIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, ++maskIt)
      {
        if (maskIt.Get() < 1e-10)
        {
          outputIt.Set(0.0); 
        }
        
      }
    }
    
    // Normalise the weight. 
    IteratorType it(outputImage, outputImage->GetLargestPossibleRegion()); 
    double mean = 0.0; 
    double voxelCount = 0.0;
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      if (it.Get() > 1e-10)
      {
        mean += it.Get(); 
        voxelCount++; 
      }
    }
    mean /= voxelCount; 
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      if (it.Get() > 1e-10)
        it.Set(it.Get()/mean); 
      else
        it.Set(0.0); 
    }
  }  
  
  
  
  itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
  writer->SetInput(outputImage); 
  writer->SetFileName(argv[3]); 
  writer->Update(); 
  
  polyDataReader->Delete(); 
  return 0; 
}





