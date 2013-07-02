/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <ConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegistrationFactory.h>
#include <itkCastImageFilter.h>

/*!
 * \file niftkConvertMidasStrToNii.cxx
 * \page niftkConvertMidasStrToNii
 * \section niftkConvertMidasStrToNiiSummary Converts a Jacobian file in MIDAS str format to nii format.
 *
 * This program converts a Jacobian file in Midas str format to nii format.
 * \li Dimensions: 3
 */
void Usage(char *name)
{
  std::cout << "  " << std::endl;
  std::cout << "  Convert a Jacobian file in MIDAS str format to nii format." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -ti inputImage inputStr -onii outputNiiJacobian" << std::endl;
  std::cout << "  " << std::endl;  
}


/**
 * \brief Convert from Midas str file format. 
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef int        PixelType;
  typedef double     ScalarType;
  
  // Define command line args
  std::string imageName; 
  std::string strName; 
  std::string outputJacobianNiiName; 

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-ti") == 0){
      imageName=argv[++i];
      strName=argv[++i];
      std::cout << "Set -ti=" << imageName << " " << strName << std::endl;
    }
    else if (strcmp(argv[i], "-onii") == 0) {
      outputJacobianNiiName = argv[++i]; 
      std::cout << "Set -onii=" << outputJacobianNiiName << std::endl; 
    }
    else{
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }    
  }

  // Validate command line args
  if (strName.length() == 0 || outputJacobianNiiName.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  typedef itk::Image< PixelType, Dimension > InputImageType; 
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, ScalarType> FactoryType;
  typedef itk::DeformableTransform< InputImageType, double, 3, double > DoubleDeformableTransformType; 
  typedef itk::DeformableTransform< InputImageType, double, 3, float > FloatDeformableTransformType; 
  typedef itk::FluidDeformableTransform< InputImageType, double, 3, float > FluidFloatDeformableTransformType; 
  
  FluidFloatDeformableTransformType::Pointer fluidDeformableTransform = FluidFloatDeformableTransformType::New(); 
  
  try
  {
    std::cout << "Trying to load it as a deformation field." << std::endl;
    typedef itk::ImageFileReader<InputImageType> ReaderType;     
    ReaderType::Pointer reader = ReaderType::New(); 
    
    reader->SetFileName(imageName); 
    reader->Update(); 
    FluidFloatDeformableTransformType::DeformableParameterType::RegionType deformationImageRegion = reader->GetOutput()->GetLargestPossibleRegion(); 
    
    InputImageType::Pointer image = InputImageType::New(); 
    InputImageType::RegionType region; 
    for (unsigned int i = 0; i < Dimension; i++)
      region.SetSize(i, deformationImageRegion.GetSize(i)); 
    image->SetRegions(region); 
    image->Allocate(); 
    
    fluidDeformableTransform = FluidFloatDeformableTransformType::New(); 
    fluidDeformableTransform->Initialize(image.GetPointer()); 
    
    // And write it.
    typedef float OutputPixelType;
    typedef itk::Image<OutputPixelType, Dimension> OutputImageType;  
    typedef itk::Image<double, Dimension> JacobianImageType;
    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    typedef itk::CastImageFilter<JacobianImageType, OutputImageType> CastFilterType;
    
    typedef itk::ImageFileReader<JacobianImageType> JacobianImageReaderType; 
    JacobianImageReaderType::Pointer jacobianImageReader = JacobianImageReaderType::New(); 
    
    jacobianImageReader->SetFileName(imageName); 
    jacobianImageReader->Update(); 
    
    int origin[Dimension]; 
    FluidFloatDeformableTransformType::DeformationFieldType::RegionType finalRegion = region; 
    fluidDeformableTransform->ReadMidasStrImage(strName, origin, finalRegion, jacobianImageReader->GetOutput()); 
    
    CastFilterType::Pointer caster = CastFilterType::New();
    WriterType::Pointer writer = WriterType::New();
    
    caster->SetInput(jacobianImageReader->GetOutput());
    
    writer->SetFileName(outputJacobianNiiName);
    writer->SetInput(caster->GetOutput());
    writer->Update();
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed:" << exceptionObject << std::endl;
    return -1; 
  }
  
  
  return EXIT_SUCCESS;   
}


