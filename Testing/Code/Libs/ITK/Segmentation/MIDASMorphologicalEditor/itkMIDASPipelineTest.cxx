/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include <math.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <MorphologicalSegmentorPipeline.h>
#include <MorphologicalSegmentorPipelineParams.h>

/**
 * Basic tests for the base morpological (brain editor) without edits.
 */
int itkMIDASPipelineTest(int argc, char * argv[])
{
  for (int i = 0; i < argc; i++)
  {
    std::cerr << "argv[" << i << "]=" << argv[i] << std::endl;
  }

  if (argc != 13)
  {
    std::cerr << "Usage: itkMIDASPipelineTest inImage.img outImage.img stage lowThreshold upperThreshold axialCutOff upperErosionThreshold numberErosions lowDilationsPercentage highDilationsPercentage numberDilations rethresholdingBox" << std::endl;
    return EXIT_FAILURE;
  }

  std::string inputFileName = argv[1];
  std::string outputFileName = argv[2];
  int stage = atoi(argv[3]);
  int lowThreshold = atoi(argv[4]);
  int upperThreshold = atoi(argv[5]);
  int axialCutOff = atoi(argv[6]);
  int upperErosions = atoi(argv[7]);
  int numberErosions = atoi(argv[8]);
  int lowDilationsPercentage = atoi(argv[9]);
  int highDilationsPercentage = atoi(argv[10]);
  int numberDilations = atoi(argv[11]);
  int rethresholdingBoxSize = atoi(argv[12]);

  // Declare the types of the images
  const unsigned int Dimension = 3;
  typedef short InputPixelType;
  typedef unsigned char OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::ImageFileReader<InputImageType>  ImageFileReaderType;

  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::ImageFileWriter<OutputImageType>  ImageFileWriterType;

  ImageFileReaderType::Pointer reader = ImageFileReaderType::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  OutputImageType::Pointer segmentationInput;
  OutputImageType::Pointer thresholdingMask;

  // Not used in this test. This test is only for when not editing.
  std::vector<bool> editingFlags;
  editingFlags.push_back(false);
  editingFlags.push_back(false);
  editingFlags.push_back(false);
  editingFlags.push_back(false);

  // Not used in this test. This test is only for when not editing.
  std::vector<int> editingRegion;
  editingRegion.push_back(0);
  editingRegion.push_back(0);
  editingRegion.push_back(0);
  editingRegion.push_back(1);
  editingRegion.push_back(1);
  editingRegion.push_back(1);

  // Not used in this test, but we must create memory.
  OutputImageType::Pointer erosionsSubtractions = OutputImageType::New();
  erosionsSubtractions->SetOrigin(reader->GetOutput()->GetOrigin());
  erosionsSubtractions->SetDirection(reader->GetOutput()->GetDirection());
  erosionsSubtractions->SetSpacing(reader->GetOutput()->GetSpacing());
  erosionsSubtractions->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  erosionsSubtractions->Allocate();
  erosionsSubtractions->FillBuffer(0);

  OutputImageType::Pointer erosionsAdditions = OutputImageType::New();
  erosionsAdditions->SetOrigin(reader->GetOutput()->GetOrigin());
  erosionsAdditions->SetDirection(reader->GetOutput()->GetDirection());
  erosionsAdditions->SetSpacing(reader->GetOutput()->GetSpacing());
  erosionsAdditions->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  erosionsAdditions->Allocate();
  erosionsAdditions->FillBuffer(0);

  OutputImageType::Pointer dilationSubtractions = OutputImageType::New();
  dilationSubtractions->SetOrigin(reader->GetOutput()->GetOrigin());
  dilationSubtractions->SetDirection(reader->GetOutput()->GetDirection());
  dilationSubtractions->SetSpacing(reader->GetOutput()->GetSpacing());
  dilationSubtractions->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  dilationSubtractions->Allocate();
  dilationSubtractions->FillBuffer(0);

  OutputImageType::Pointer dilationsAdditions = OutputImageType::New();
  dilationsAdditions->SetOrigin(reader->GetOutput()->GetOrigin());
  dilationsAdditions->SetDirection(reader->GetOutput()->GetDirection());
  dilationsAdditions->SetSpacing(reader->GetOutput()->GetSpacing());
  dilationsAdditions->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  dilationsAdditions->Allocate();
  dilationsAdditions->FillBuffer(0);

  MorphologicalSegmentorPipelineParams params;
  params.m_LowerIntensityThreshold = lowThreshold;
  params.m_UpperIntensityThreshold = upperThreshold;
  params.m_AxialCutOffSlice = axialCutOff;
  params.m_UpperErosionsThreshold = upperErosions;
  params.m_NumberOfErosions = numberErosions;
  params.m_LowerPercentageThresholdForDilations = lowDilationsPercentage;
  params.m_UpperPercentageThresholdForDilations = highDilationsPercentage;
  params.m_NumberOfDilations = numberDilations;
  params.m_BoxSize = rethresholdingBoxSize;

  typedef MorphologicalSegmentorPipeline<InputPixelType, Dimension> PipelineType;
  PipelineType *pipeline = new PipelineType();
  pipeline->SetForegroundValue((unsigned char)255);
  pipeline->SetBackgroundValue((unsigned char)0);

  for (int i = 0; i <= stage; i++)
  {
    params.m_Stage = i;
    pipeline->SetParams(reader->GetOutput(),
                        erosionsAdditions,
                        erosionsSubtractions,
                        dilationsAdditions,
                        dilationSubtractions,
                        segmentationInput,
                        thresholdingMask,
                        params);
    pipeline->Update(editingFlags, editingRegion);
  }

  ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
  writer->SetInput(pipeline->GetOutput().GetPointer());
  writer->SetFileName(outputFileName);
  writer->Update();

  return EXIT_SUCCESS;
}
