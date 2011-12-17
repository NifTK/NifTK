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

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSegmentationReliabilityCalculator.h"
#include "itkIndent.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
  if (argc < 5)
  {
    std::cerr << "Segmentation reliablility" << std::endl << std::endl; 
    std::cerr << "Usage: " << argv[0] << std::endl;
    std::cerr << "         <baseline mask for BSI>" << std::endl; 
    std::cerr << "         <repeat mask for BSI>" << std::endl;
    std::cerr << "         <number of erosion>" << std::endl;
    std::cerr << "         <number of dilation>" << std::endl;
    return EXIT_FAILURE;
  }
  
  try
  {
    typedef itk::Image<int, 3> IntImageType;

    typedef itk::ImageFileReader<IntImageType> IntReaderType;
    typedef itk::ImageFileWriter<IntImageType> WriterType;
    typedef itk::SegmentationReliabilityCalculator<IntImageType,IntImageType,IntImageType> SegmentationReliabilityFilterType;

    SegmentationReliabilityFilterType::Pointer bsiFilter = SegmentationReliabilityFilterType::New();
    WriterType::Pointer writer = WriterType::New();
    IntReaderType::Pointer baselineBSIMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatBSIMaskReader = IntReaderType::New();

    baselineBSIMaskReader->SetFileName(argv[1]);
    repeatBSIMaskReader->SetFileName(argv[2]);

    bsiFilter->SetBaselineMask(baselineBSIMaskReader->GetOutput());
    bsiFilter->SetRepeatMask(repeatBSIMaskReader->GetOutput());
    bsiFilter->SetNumberOfErosion(atoi(argv[3]));
    bsiFilter->SetNumberOfDilation(atoi(argv[4]));
    bsiFilter->Compute();
    //std::cout << "BSI=" << bsiFilter->GetSegmentationReliability() << std::endl;
    std::cout << bsiFilter->GetSegmentationReliability() << std::endl;
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}


