/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <iostream>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkSignedDanielssonDistanceMapImageFilter.h>
#include <itkWatershedImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkCastImageFilter.h>


int main(int argc, char* argv[])
{
  const unsigned int Dimensions = 3; 
  typedef itk::Image< short, Dimensions > ImageType;
	typedef itk::Image< float, Dimensions > FloatImageType; 
	typedef itk::Image< unsigned long, Dimensions > LongImageType; 
	
	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	ImageReaderType::Pointer reader = ImageReaderType::New(); 
	
	typedef itk::SignedDanielssonDistanceMapImageFilter<ImageType, FloatImageType> DistanceMapFilterType;
	DistanceMapFilterType::Pointer distanceMapFilter = DistanceMapFilterType::New(); 
	
	reader->SetFileName(argv[1]); 
	reader->Update(); 
	
	distanceMapFilter->SetInput(reader->GetOutput()); 
  distanceMapFilter->SetUseImageSpacing(true); 
	distanceMapFilter->Update(); 
	
	typedef itk::ImageFileWriter<FloatImageType> FloatImageWriterType;
	FloatImageWriterType::Pointer writer = FloatImageWriterType::New(); 
	
	writer->SetFileName(argv[2]); 
	writer->SetInput(distanceMapFilter->GetOutput()); 
	writer->Update(); 
	
	typedef itk::WatershedImageFilter<FloatImageType> WatershedImageFilterType;
	WatershedImageFilterType::Pointer watershedFilter = WatershedImageFilterType::New(); 
	
	watershedFilter->SetInput(distanceMapFilter->GetOutput());
	watershedFilter->SetThreshold(atof(argv[4])); 
	watershedFilter->SetLevel(atof(argv[5])); 
  
  typedef itk::CastImageFilter<WatershedImageFilterType::OutputImageType, ImageType> CastImageFilterType; 
  CastImageFilterType::Pointer castImageFilter = CastImageFilterType::New(); 
  castImageFilter->SetInput(watershedFilter->GetOutput());
  
  typedef itk::ImageFileWriter<ImageType> IntFileWriterType;
  IntFileWriterType::Pointer intWriter = IntFileWriterType::New();
  intWriter->SetInput(castImageFilter->GetOutput());
  intWriter->SetFileName(argv[3]);
  intWriter->Update(); 

  return 0;
}

