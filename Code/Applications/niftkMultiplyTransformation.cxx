/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include <itkImage.h>
#include <itkImageRegistrationFactory.h>
#include <itkImageRegistrationFilter.h>
#include <itkTransformFileWriter.h>
#include <itkImageFileReader.h>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_STRING|OPT_REQ, "i1",  "filename", "Input transformation 1."},

  {OPT_STRING|OPT_REQ, "i2", "filename", "Input transformation 2."},
  
  {OPT_STRING, "o1", "filename", "Output transformation."},
  
  {OPT_STRING, "air_target_image", "filename", "Convert AIR transformation which uses the specified image as the target."},
  
  {OPT_STRING, "air_source_image", "filename", "Convert AIR transformation which uses the specified image as the source."},
  
  {OPT_DONE, NULL, NULL, "Multiply the two input transformations."}
   
};


enum {
  O_INPUT_FILE_1=0,

  O_INPUT_FILE_2, 
  
  O_OUTPUT_FILE_1,
   
  O_AIR_TARGET_FILE, 
  
  O_AIR_SOURCE_FILE
};

int main(int argc, char** argv)
{
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);
  std::string inputFilename1;
  std::string inputFilename2;
  std::string outputFilename; 
  std::string airTargetFilename; 
  std::string airSourceFilename; 
  
  CommandLineOptions.GetArgument(O_INPUT_FILE_1, inputFilename1);
  CommandLineOptions.GetArgument(O_INPUT_FILE_2, inputFilename2);
  CommandLineOptions.GetArgument(O_OUTPUT_FILE_1, outputFilename);
  CommandLineOptions.GetArgument(O_AIR_TARGET_FILE, airTargetFilename);
  CommandLineOptions.GetArgument(O_AIR_SOURCE_FILE, airSourceFilename);
  
  typedef short PixelType; 
  const int Dimension = 3; 
  typedef itk::Image<PixelType, Dimension>  InputImageType; 
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  typedef itk::EulerAffineTransform<double, Dimension, Dimension> AffineTransformType; 
  typedef itk::ImageFileReader<InputImageType> ImageReaderType;
  ImageReaderType::Pointer airTargetReader = ImageReaderType::New(); 
  ImageReaderType::Pointer airSourceReader = ImageReaderType::New(); 
  
  AffineTransformType::Pointer inputTransform1; 
  AffineTransformType::Pointer inputTransform2; 
  
  try
  {
    FactoryType::Pointer factory = FactoryType::New();
    
    inputTransform1 = dynamic_cast<AffineTransformType*>(factory->CreateTransform(inputFilename1).GetPointer());
    std::cout << inputTransform1->GetFullAffineMatrix() << std::endl; 
    
    inputTransform2 = dynamic_cast<AffineTransformType*>(factory->CreateTransform(inputFilename2).GetPointer());
    std::cout << inputTransform2->GetFullAffineMatrix() << std::endl; 
    
    AffineTransformType::FullAffineMatrixType targetMatrix; 
    targetMatrix.SetIdentity(); 
    AffineTransformType::FullAffineMatrixType sourceMatrix; 
    sourceMatrix.SetIdentity(); 
    AffineTransformType::FullAffineMatrixType sourceMatrixInverse; 
    sourceMatrixInverse.SetIdentity(); 
    
    // AIR does not seem to take into account the direction matrix in the images. 
    // When converting to ITK dof, we need to mulitply the AIR matrix 
    // by the target direction matrix on the left, and 
    // by the inverse of the source direction matrix on the right. 
    if (airTargetFilename.length() > 0)
    {
      airTargetReader->SetFileName(airTargetFilename); 
      airTargetReader->Update(); 
      InputImageType::DirectionType direction = airTargetReader->GetOutput()->GetDirection(); 
      for (int i = 0; i < Dimension; i++)
      {
        for (int j = 0; j < Dimension; j++)
        {
          targetMatrix(i, j) = direction(i, j); 
        }
      }
      std::cout << "targetMatrix=" << std::endl << targetMatrix << std::endl; 
    }
    
    if (airSourceFilename.length() > 0)
    {
      airSourceReader->SetFileName(airSourceFilename); 
      airSourceReader->Update(); 
      InputImageType::DirectionType direction = airSourceReader->GetOutput()->GetDirection(); 
      for (int i = 0; i < Dimension; i++)
      {
        for (int j = 0; j < Dimension; j++)
        {
          sourceMatrix(i, j) = direction(i, j); 
        }
      }
      std::cout << "sourceMatrix=" << std::endl << sourceMatrix << std::endl; 
      sourceMatrixInverse = sourceMatrix.GetInverse(); 
      std::cout << "sourceMatrixInverse=" << std::endl << sourceMatrixInverse << std::endl; 
    }
    
    std::cout << "matrix=" << std::endl << targetMatrix*inputTransform1->GetFullAffineMatrix()*inputTransform2->GetFullAffineMatrix()*sourceMatrixInverse << std::endl;  
    
    inputTransform1->SetFullAffineMatrix(targetMatrix*inputTransform1->GetFullAffineMatrix()*inputTransform2->GetFullAffineMatrix()*sourceMatrixInverse); 
    inputTransform1->SetParametersFromTransform(inputTransform1->GetFullAffineTransform()); 
    std::cout << inputTransform1->GetFullAffineMatrix() << std::endl; 
    
    if (outputFilename.length() > 0)
    {
#if 0
      itk::EulerAffineTransform<double, Dimension, Dimension>::TranslationType translation = inputTransform1->GetTranslation(); 
      translation[2] = -translation[2]; 
      inputTransform1->SetTranslation(translation); 
      
      itk::EulerAffineTransform<double, Dimension, Dimension>::RotationType rotation = inputTransform1->GetRotation(); 
      rotation[2] = -rotation[2]; 
      inputTransform1->SetRotation(rotation);   
#endif      
      
      itk::TransformFileWriter::Pointer transformWriter;
      transformWriter = itk::TransformFileWriter::New();
      transformWriter->SetFileName(outputFilename);
      transformWriter->SetInput(inputTransform1);
      transformWriter->Update();
    }
    
  }  
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cerr << "Failed to load global tranform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  return 0; 
  
}
  
  
  
  
  
  
  
