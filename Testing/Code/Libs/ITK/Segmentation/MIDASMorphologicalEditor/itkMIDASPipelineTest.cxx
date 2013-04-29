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
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "MorphologicalSegmentorPipeline.h"
#include "MorphologicalSegmentorPipelineParams.h"

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
    std::cerr << "Usage: itkMIDASPipelineTest inImage.img outImage.img stage lowThreshold upperThreshold axialCutoff upperErosionThreshold numberErosions lowDilationsPercentage highDilationsPercentage numberDilations rethresholdingBox" << std::endl;
    return EXIT_FAILURE;
  }

  std::string inputFileName = argv[1];
  std::string outputFileName = argv[2];
  int stage = atoi(argv[3]);
  int lowThreshold = atoi(argv[4]);
  int upperThreshold = atoi(argv[5]);
  int axialCutoff = atoi(argv[6]);
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
  OutputImageType::Pointer erodeEdits = OutputImageType::New();
  erodeEdits->SetOrigin(reader->GetOutput()->GetOrigin());
  erodeEdits->SetDirection(reader->GetOutput()->GetDirection());
  erodeEdits->SetSpacing(reader->GetOutput()->GetSpacing());
  erodeEdits->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  erodeEdits->Allocate();
  erodeEdits->FillBuffer(0);

  OutputImageType::Pointer erodeAdds = OutputImageType::New();
  erodeAdds->SetOrigin(reader->GetOutput()->GetOrigin());
  erodeAdds->SetDirection(reader->GetOutput()->GetDirection());
  erodeAdds->SetSpacing(reader->GetOutput()->GetSpacing());
  erodeAdds->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  erodeAdds->Allocate();
  erodeAdds->FillBuffer(0);

  OutputImageType::Pointer dilateEdits = OutputImageType::New();
  dilateEdits->SetOrigin(reader->GetOutput()->GetOrigin());
  dilateEdits->SetDirection(reader->GetOutput()->GetDirection());
  dilateEdits->SetSpacing(reader->GetOutput()->GetSpacing());
  dilateEdits->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  dilateEdits->Allocate();
  dilateEdits->FillBuffer(0);

  OutputImageType::Pointer dilateAdds = OutputImageType::New();
  dilateAdds->SetOrigin(reader->GetOutput()->GetOrigin());
  dilateAdds->SetDirection(reader->GetOutput()->GetDirection());
  dilateAdds->SetSpacing(reader->GetOutput()->GetSpacing());
  dilateAdds->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  dilateAdds->Allocate();
  dilateAdds->FillBuffer(0);

  MorphologicalSegmentorPipelineParams params;
  params.m_LowerIntensityThreshold = lowThreshold;
  params.m_UpperIntensityThreshold = upperThreshold;
  params.m_AxialCutoffSlice = axialCutoff;
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
    pipeline->SetParam(reader->GetOutput(),
                       erodeAdds,
                       erodeEdits,
                       dilateAdds,
                       dilateEdits,
                       params);
    pipeline->Update(editingFlags, editingRegion);
  }

  ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
  writer->SetInput(pipeline->GetOutput(editingFlags).GetPointer());
  writer->SetFileName(outputFileName);
  writer->Update();

  return EXIT_SUCCESS;
}
