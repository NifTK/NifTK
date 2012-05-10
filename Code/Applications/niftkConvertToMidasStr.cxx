/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-14 09:36:52 +0000 (Mon, 14 Nov 2011) $
 Revision          : $Revision: 7770 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
// #include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationFactory.h"

/*!
 * \file niftkConvertToMidasStr.cxx
 * \page niftkConvertToMidasStr
 * \section niftkConvertToMidasStrSummary Output a Jacobian file in Midas str format. 
 *
 * This program convert a Jacobian file in Midas str format to nii format.
 * \li Dimensions: 3
 * \li Pixel type: Vectors only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float, double
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Output a Jacobian file in Midas str format." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -ti input -oi outputJacobian -ov outputVector -onii outputNiiJacobian" << std::endl;
  std::cout << "  " << std::endl;  
}


/**
 * \brief Convert to Midas str file format. 
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef int        PixelType;
  typedef double     ScalarType;
  
  // Define command line args
  std::string transformName; 
  std::string outputName; 
  std::string outputVectorName;
  std::string outputJacobianNiiName; 
  int imageSize[Dimension] = {0, 0, 0};
  bool isResize = false; 

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-ti") == 0){
      transformName=argv[++i];
      std::cout << "Set -ti=" << transformName << std::endl;
    }
    else if(strcmp(argv[i], "-oi") == 0){
      outputName=argv[++i];
      std::cout << "Set -oi=" << outputName<< std::endl;
    }    
    else if(strcmp(argv[i], "-ov") == 0){
      outputVectorName=argv[++i];
      std::cout << "Set -ov=" << outputVectorName<< std::endl;
    }
    else if(strcmp(argv[i], "-size") == 0){
      for (unsigned int j = 0; j < Dimension; j++)
      {
        imageSize[j] = atoi(argv[++i]);
        std::cout << "Set -size=" << argv[i]<< std::endl;
      }
     isResize = true; 
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
  if (transformName.length() == 0 || outputName.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  typedef itk::Image< PixelType, Dimension > InputImageType; 
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, ScalarType> FactoryType;
  typedef itk::DeformableTransform< InputImageType, double, 3, double > DoubleDeformableTransformType; 
  typedef itk::DeformableTransform< InputImageType, double, 3, float > FloatDeformableTransformType; 
  typedef itk::FluidDeformableTransform< InputImageType, double, 3, float > FluidFloatDeformableTransformType; 
  
  // The factory.
  FactoryType::Pointer factory = FactoryType::New();
  FactoryType::TransformType::Pointer deformableTransform; 
  
  FluidFloatDeformableTransformType::Pointer fluidDeformableTransform = FluidFloatDeformableTransformType::New(); 
  
  try
  {
    std::cout << "Trying to load it as a deformation field." << std::endl;
    typedef itk::ImageFileReader<FluidFloatDeformableTransformType::DeformableParameterType> DeformationFileReaderType;     
    DeformationFileReaderType::Pointer reader = DeformationFileReaderType::New(); 
    
    reader->SetFileName(transformName); 
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
    fluidDeformableTransform->SetDeformableParameters(reader->GetOutput()); 
    
    deformableTransform = fluidDeformableTransform.GetPointer(); 
    
    if (outputJacobianNiiName.length() > 0)
    {
      fluidDeformableTransform->ComputeMinJacobian(); 
      fluidDeformableTransform->WriteJacobianImage(outputJacobianNiiName); 
    }
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed to load it as a deformation field." << exceptionObject << std::endl;
    std::cout << "Trying it to load it as a itk::deformableTransform." << exceptionObject << std::endl;
  }
  
  try
  {
    if (deformableTransform.IsNull())
      deformableTransform = factory->CreateTransform(transformName);
  }  
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed to load it as itk::deformableTransform tranform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  DoubleDeformableTransformType* doubleDeformableTransform = dynamic_cast<DoubleDeformableTransformType*>(deformableTransform.GetPointer()); 
  FloatDeformableTransformType* floatDeformableTransform = dynamic_cast<FloatDeformableTransformType*>(deformableTransform.GetPointer()); 
  int origin[Dimension]; 
  
  for (unsigned int i = 0; i < Dimension; i++)
  	origin[i] = 0; 
  
  if (doubleDeformableTransform)
  {
    DoubleDeformableTransformType::DeformationFieldType::RegionType region = doubleDeformableTransform->GetDeformationField()->GetLargestPossibleRegion(); 
    DoubleDeformableTransformType::DeformationFieldType::RegionType finalRegion = region; 
    
    if (isResize)
    {
      for (unsigned int i = 0; i < Dimension; i++)
      {
        finalRegion.SetIndex(i, (region.GetSize()[i]-imageSize[i])/2); 
        finalRegion.SetSize(i, imageSize[i]); 
      }
    }
    
    if (outputName.length() > 0)
      doubleDeformableTransform->WriteMidasStrImage(outputName, origin, finalRegion, NULL); 
    if (outputVectorName.length() > 0)
      doubleDeformableTransform->WriteMidasVecImage(outputVectorName, origin, finalRegion); 
  }
  else if (floatDeformableTransform)
  {
    FloatDeformableTransformType::DeformationFieldType::RegionType region = floatDeformableTransform->GetDeformationField()->GetLargestPossibleRegion(); 
    FloatDeformableTransformType::DeformationFieldType::RegionType finalRegion = region; 
    
    if (isResize)
    {
      for (unsigned int i = 0; i < Dimension; i++)
      {
        finalRegion.SetIndex(i, (region.GetSize()[i]-imageSize[i])/2); 
        finalRegion.SetSize(i, imageSize[i]); 
      }
    }
    
    if (outputName.length() > 0)
      floatDeformableTransform->WriteMidasStrImage(outputName, origin, finalRegion, NULL); 
    if (outputVectorName.length() > 0)
      floatDeformableTransform->WriteMidasVecImage(outputVectorName, origin, finalRegion); 
  }
  
  
  return EXIT_SUCCESS;   
}


