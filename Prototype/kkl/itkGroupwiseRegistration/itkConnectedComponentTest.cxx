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
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIterator.h"

int main(int argc, char* argv[] )
{
  if( argc < 5 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " inputImage  maskImage outputImage threshold_low threshold_hi fully_connected minimum_object_size" << std::endl;
    return EXIT_FAILURE;
    }

  typedef   short  InternalPixelType;
  typedef   bool  MaskPixelType;
  const     unsigned int    Dimension = 3;
  
  typedef itk::Image< InternalPixelType, Dimension >  InternalImageType;
  typedef itk::Image< MaskPixelType, Dimension >  MaskImageType;
  typedef itk::Image<short,Dimension> OutputImageType;

  typedef itk::ImageFileReader< InternalImageType > ReaderType;
  typedef itk::ImageFileReader< MaskImageType > MaskReaderType;
  typedef itk::ImageFileWriter<  OutputImageType  > WriterType;
  
  typedef itk::BinaryThresholdImageFilter< InternalImageType, InternalImageType > ThresholdFilterType;
  typedef itk::ConnectedComponentImageFilter< InternalImageType, OutputImageType, MaskImageType > FilterType;
  typedef itk::RelabelComponentImageFilter< OutputImageType, OutputImageType > RelabelType;

  
  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  ThresholdFilterType::Pointer threshold = ThresholdFilterType::New();
  FilterType::Pointer filter = FilterType::New();
  RelabelType::Pointer relabel = RelabelType::New();
  MaskReaderType::Pointer maskReader = MaskReaderType::New(); 
  
  reader->SetFileName( argv[1] );
  reader->Update(); 
  maskReader->SetFileName(argv[2]); 
  maskReader->Update(); 
  writer->SetFileName(argv[3]); 

  InternalPixelType threshold_low, threshold_hi;
  threshold_low = atoi( argv[4]);
  threshold_hi = atoi( argv[5]);

  threshold->SetInput (reader->GetOutput());
  threshold->SetInsideValue(itk::NumericTraits<InternalPixelType>::One);
  threshold->SetOutsideValue(itk::NumericTraits<InternalPixelType>::Zero);
  threshold->SetLowerThreshold(threshold_low);
  threshold->SetUpperThreshold(threshold_hi);
  threshold->Update();
  
  filter->SetInput (threshold->GetOutput());
  //filter->SetMaskImage (maskReader->GetOutput());
  //filter->SetInput (reader->GetOutput());

  if (argc > 6)
  {
    int fullyConnected = atoi( argv[6] );
    filter->SetFullyConnected( fullyConnected );
  }
  relabel->SetInput( filter->GetOutput() );
  if (argc > 7)
  {
    int minSize = atoi( argv[7] );
    relabel->SetMinimumObjectSize( minSize );
    std::cerr << "minSize: " << minSize << std::endl;
  }
  
  try
  {
    filter->Update(); 
    relabel->Update();
    writer->SetInput(relabel->GetOutput()); 
    writer->Update(); 
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Relabel: exception caught !" << std::endl;
    std::cerr << excep << std::endl;
  }

  unsigned short numObjects = relabel->GetNumberOfObjects();
  unsigned int largestSize = 0; 
  unsigned short largetSizeIndex = 0;
  
  for (unsigned short i = 0; i < numObjects; i++)
  {
    if (relabel->GetSizeOfObjectInPixels(i) > largestSize)
    {
      largestSize = relabel->GetSizeOfObjectInPixels(i);
      largetSizeIndex = i; 
    }
  }
  std::cout << "largestSize=" << largestSize << ",largetSizeIndex=" << largetSizeIndex << std::endl;
  std::cout << largetSizeIndex << std::endl;

  return EXIT_SUCCESS;
}





