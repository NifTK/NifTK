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

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkEulerAffineTransform.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkThresholdImageFilter.h"

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef short PixelType;
  typedef itk::Image< PixelType, Dimension > FixedImageType;
  typedef itk::ImageFileReader< FixedImageType > FixedImageReaderType;
  typedef itk::LinearInterpolateImageFunction< FixedImageType, double> InterpolatorType;
  typedef itk::EulerAffineTransform< double,Dimension,Dimension > TransformType;
  
  // Read in the fixed image and its mask. 
  FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
  FixedImageType::SizeType regionSize; 
  
  fixedImageReader->SetFileName(argv[1]);
  fixedImageReader->Update();
  regionSize = fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize(); 
  FixedImageType::IndexType centerIndex; 
  centerIndex[0] = regionSize[0]/2; 
  centerIndex[1] = regionSize[1]/2; 
  centerIndex[2] = regionSize[2]/2; 
  FixedImageType::PointType centerPoint; 
  fixedImageReader->GetOutput()->TransformIndexToPhysicalPoint(centerIndex, centerPoint);
  
  char* outputFilename = argv[2];
  
  TransformType::Pointer transform = TransformType::New();
  TransformType::ParametersType parameters; 
    
  transform->SetRigid();
  transform->SetIdentity();
  transform->SetCenter(centerPoint);
  
  char* dofinFilename = argv[3]; 
  std::ifstream dofFile(dofinFilename);
  std::string line;
  unsigned int parameterIndex = 0;
  if (dofFile.is_open())
  {
    while (!dofFile.eof())
    {
      getline(dofFile, line);
      if (!dofFile.eof())
      {
        if (parameterIndex < parameters.Size())
        {
          parameters[parameterIndex] = atof(line.c_str());
          parameterIndex++;
        }
      }
    }
    dofFile.close();
  }
    
  typedef itk::ResampleImageFilter< FixedImageType, FixedImageType > ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  typedef FixedImageType OutputImageType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  typedef itk::ImageRegistrationFactory< FixedImageType, Dimension, double >       FactoryType;  
  FactoryType::Pointer factory = FactoryType::New();
  typedef itk::ThresholdImageFilter< FixedImageType > ThresholdImageFilterType; 
  ThresholdImageFilterType::Pointer thresholdImageFilter = ThresholdImageFilterType::New(); 
  
  int interpolationMode = atoi(argv[4]);
  resampler->SetInput(fixedImageReader->GetOutput());
  resampler->SetTransform(transform);
  resampler->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)interpolationMode));
  resampler->SetDefaultPixelValue(0);
  resampler->SetReferenceImage(fixedImageReader->GetOutput());
  resampler->UseReferenceImageOn();
  resampler->Update(); 
  const FixedImageType* outputImage = resampler->GetOutput(); 
  
  if (argc > 5)
  {
    int thresholdValue = atoi(argv[5]); 
    thresholdImageFilter->SetInput(outputImage); 
    thresholdImageFilter->ThresholdBelow(thresholdValue);
    thresholdImageFilter->SetOutsideValue(0);
    thresholdImageFilter->Update(); 
    outputImage = thresholdImageFilter->GetOutput(); 
  } 
  
  writer->SetInput(outputImage);
  writer->SetFileName(outputFilename);
  writer->Update();
  
  return EXIT_SUCCESS;    
  
}
