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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegistrationFactory.h>
#include <itkImageRegistrationFilter.h>
#include <itkImageRegistrationFactory.h>
#include <itkGradientDescentOptimizer.h>
#include <itkUCLSimplexOptimizer.h>
#include <itkUCLRegularStepGradientDescentOptimizer.h>
#include <itkSingleResolutionImageRegistrationBuilder.h>
#include <itkMaskedImageRegistrationMethod.h>
#include <itkTransformFileWriter.h>
#include <itkEulerAffineTransform.h>
#include <itkIdentityTransform.h>
#include <fstream>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_STRING|OPT_REQ, "i",  "filename", "Input NIFTK affine transformation file."},

  {OPT_STRING|OPT_REQ, "t", "filename", "Input image with the target coordinate system."},

  {OPT_STRING|OPT_REQ, "o", "filename", "Output NiftiReg transformation file."},

  {OPT_DONE, NULL, NULL,
   "Convert NIFTK affine transformation file format to NiftiReg format."}
   
};


enum {
  O_INPUT_FILE=0,

  O_INPUT_TARGET_FILE=1,

  O_OUTPUT_FILE

};


int main(int argc, char** argv)
{
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);
  std::string inputFilename;
  std::string inputTargetImageFilename;
  std::string outputFilename;
  
  CommandLineOptions.GetArgument(O_INPUT_FILE, inputFilename);
  CommandLineOptions.GetArgument(O_INPUT_TARGET_FILE, inputTargetImageFilename);
  CommandLineOptions.GetArgument(O_OUTPUT_FILE, outputFilename);

  typedef short PixelType; 
  const int Dimension = 3; 
  typedef itk::Image< PixelType, Dimension >  InputImageType; 
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  FactoryType::Pointer factory = FactoryType::New();
  FactoryType::TransformType::Pointer globalTransform; 
  typedef itk::ImageFileReader<InputImageType> ImageReaderType;
  typedef itk::ImageFileWriter<InputImageType> ImageWriterType;
  InputImageType::PointType origin;
  
  try
  {
    std::cout << "Creating global transform from:" << inputFilename << std::endl; 
    globalTransform = factory->CreateTransform(inputFilename);
    std::cout << "Done" << std::endl; 

    std::cout << "Reading image: " << inputTargetImageFilename << std::endl;
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName(inputTargetImageFilename);
    reader->Update();
    origin = reader->GetOutput()->GetOrigin();
    std::cout << "origin=" << origin << std::endl;
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cerr << "Failed to load global tranform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }

  itk::EulerAffineTransform<double, Dimension, Dimension>* affineTransform = dynamic_cast<itk::EulerAffineTransform<double, Dimension, Dimension>*>(globalTransform.GetPointer()); 
  
  if (affineTransform == NULL)
  {
    std::cerr << "Input file format is incorrect." << std::endl;
    return EXIT_FAILURE; 
  }
  
  // It appears that ITK flips the x and y axes, so they are flipped back to make it work with NiftiReg. 
  itk::EulerAffineTransform<double, Dimension, Dimension>::InputPointType center = affineTransform->GetCenter(); 
  center[0] = -origin[0];
  center[1] = -origin[1];
  center[2] = origin[2];
  affineTransform->SetCenter(center);
  
  itk::EulerAffineTransform<double, Dimension, Dimension>::TranslationType translation = affineTransform->GetTranslation(); 
  translation[0] = -translation[0];
  translation[1] = -translation[1];
  translation[2] = -translation[2];
  affineTransform->SetTranslation(translation);
  
  itk::EulerAffineTransform<double, Dimension, Dimension>::RotationType rotation = affineTransform->GetRotation(); 
  rotation[0] = rotation[0];
  rotation[1] = rotation[1];
  rotation[2] = rotation[2];
  affineTransform->SetRotation(rotation);

  itk::EulerAffineTransform<double, Dimension, Dimension>::ScaleType scale = affineTransform->GetScale();
  scale[0] = scale[0];
  scale[1] = scale[1];
  scale[2] = scale[2];
  affineTransform->SetScale(scale);

  itk::EulerAffineTransform<double, Dimension, Dimension>::SkewType skew = affineTransform->GetSkew();
  skew[0] = skew[0];
  skew[1] = skew[1];
  skew[2] = skew[2];
  affineTransform->SetSkew(skew);
  
  itk::EulerAffineTransform<double, Dimension, Dimension>::FullAffineMatrixType matrix = affineTransform->GetFullAffineMatrix(); 
      
  std::ofstream outputFile(outputFilename.c_str()); 
  outputFile << matrix; 
  outputFile.close(); 
  
  std::ofstream outputFile1("inverse.mat");
  outputFile1 << matrix.GetInverse();
  outputFile1.close();


  return 0;   
  
}




