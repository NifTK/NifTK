/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 17:44:42 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7864 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkGradientDescentOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkMaskedImageRegistrationMethod.h"
#include "itkTransformFileWriter.h"
#include "itkEulerAffineTransform.h"
#include "itkIdentityTransform.h"
#include "itkShiftScaleImageFilter.h"

/*!
 * \file niftkTransformation.cxx
 * \page niftkTransformation
 * \section niftkTransformationSummary Transforms an image by a transformation.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Transforms an image by a transformation." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -ti inputFixedImage -si inputMovingImage -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -ti   <filename>        Input Target/Fixed image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -si    <filename>       Input Source/Moving image. If this isn't specified, we just read the targetImage twice. " << std::endl;
    std::cout << "    -j    [int] [3]         Choose reslicing interpolator" << std::endl;
    std::cout << "                              1. Nearest Neighbour"  << std::endl;
    std::cout << "                              2. Linear"  << std::endl;
    std::cout << "                              3. BSpline"  << std::endl;
    std::cout << "                              4. Sinc"  << std::endl;
    std::cout << "    -default <int> [0]      Default pixel value for padding" << std::endl;
    std::cout << "    -identity               Ignore all transformations, and resample with identity transform" << std::endl;
    std::cout << "    -iso  <float> [1]       Resample to iso-tropic voxels of the specified size" << std::endl << std::endl;
    std::cout << "    -half                   Half the affine and non-linear transformation" << std::endl;
    std::cout << "    -invert                 Invert the affine and non-linear transformation" << std::endl << std::endl;
    std::cout << "    -g    <filename>        Global (Affine) ITK transform file" << std::endl;
    std::cout << "    -halfAffine             Half the affine transformation" << std::endl;
    std::cout << "    -invertAffine           Invert the affine transformation" << std::endl << std::endl;
    std::cout << "    -df   <filename>        Deformation dof file name " << std::endl;
    std::cout << "    -di   <filename>        Deformation vector image file name " << std::endl;
    std::cout << "    -halfNonLinear          Half the non-linear transformation" << std::endl;
    std::cout << "    -invertNonLinear        Invert the non-linear transformation" << std::endl;
    std::cout << "    -ajc                    Affine Jacobian intensity correction" << std::endl; 
    std::cout << "    -sym_midway <fixed> <moving> This option is used to handle dof files from the midway registration (-sym_midway)" << std::endl; 
    std::cout << "  " << std::endl;
    std::cout << " TODO:The resampling to isotropic doesn't smooth when downsampling, and doesn't sharpen when upsampling" << std::endl;
  }

struct arguments
{
  std::string fixedImage;
  std::string movingImage;
  std::string outputImage;
  std::string globalTransformName;
  std::string deformableTransformName;
  std::string deformationFieldName;
  int finalInterp;
  bool halfAffineTransformation;  
  bool invertAffineTransformation; 
  bool halfNonLinearTransformation;  
  bool invertNonLinearTransformation;
  bool isIdentity;
  float isoVoxelSize;
  bool doIsotropicVoxels;
  int defaultPixelValue;
  bool affineJacobianIntensityCorrection; 
  bool isSymMidway; 
  int numberOfSymImages; 
  std::string* symImages; 
};


template <int Dimension, class PixelType> 
int DoMain(arguments args)
{ 
  typedef typename itk::Image< PixelType, Dimension >  InputImageType; 
  typedef typename itk::Image< PixelType, Dimension >  OutputImageType;  
  typedef typename itk::ImageFileReader< InputImageType >   FixedImageReaderType;
  typedef typename itk::ImageFileReader< InputImageType >   MovingImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType >   OutputImageWriterType;
  typedef typename itk::Vector<float, Dimension>            VectorPixelType;
  typedef typename itk::Image<VectorPixelType, Dimension>   VectorImageType;
  typedef typename itk::ImageFileReader < VectorImageType > VectorImageReaderType;
  
  typename FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  typename MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  typename VectorImageReaderType::Pointer vectorImageReader = VectorImageReaderType::New();
  
  double affineScalingFactor = 1.0; 
  
  fixedImageReader->SetFileName(  args.fixedImage );
  movingImageReader->SetFileName( args.movingImage );
  

  // Load both images.
  try 
  { 
    std::cout << "Reading fixedImage:" <<  args.fixedImage << std::endl;
    fixedImageReader->Update();
    std::cout << "Done" << std::endl;
    
    std::cout << "Reading movingImage:" << args.movingImage << std::endl;
    movingImageReader->Update();
    std::cout << "Done" << std::endl;
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed to load input images: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  
  // Setup objects to build registration.
  typedef typename itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  
  // The factory.
  typename FactoryType::Pointer factory = FactoryType::New();
  typename FactoryType::TransformType::Pointer globalTransform; 
  typename FactoryType::TransformType::Pointer deformableTransform; 
  typename FactoryType::FluidDeformableTransformType::Pointer fluidTransform;
  typename FactoryType::InterpolatorType::Pointer interpolator; 
  typename VectorImageType::Pointer vectorImage;
  
  interpolator = factory->CreateInterpolator((itk::InterpolationTypeEnum)args.finalInterp);
  
  if (args.isIdentity)
    { 
      globalTransform = itk::IdentityTransform<double, Dimension>::New();
    }
  
  if (args.globalTransformName.length() != 0)
  {
    try
    {
      std::cout << "Creating global transform from: " << args.globalTransformName << std::endl;
      globalTransform = factory->CreateTransform(args.globalTransformName);
      std::cout << "Done" << std::endl;
    }  
    catch (itk::ExceptionObject& exceptionObject)
    {
      std::cerr << "Failed to load global tranform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }
    
    itk::EulerAffineTransform<double, Dimension, Dimension>* affineTransform = dynamic_cast<itk::EulerAffineTransform<double, Dimension, Dimension>*>(globalTransform.GetPointer()); 
    std::cout << affineTransform->GetFullAffineMatrix() << std::endl; 
    
    if (args.halfAffineTransformation)
    {
      affineTransform->HalfTransformationMatrix(); 
      std::cout << "half:" << std::endl << affineTransform->GetFullAffineMatrix() << std::endl; 
    }
    if (args.invertAffineTransformation)
    {
      affineTransform->InvertTransformationMatrix(); 
      std::cout << "inverted:" << std::endl << affineTransform->GetFullAffineMatrix() << std::endl; 
    }
    for (int i = 0; i < Dimension; i++)
      affineScalingFactor *= affineTransform->GetScale()[i]; 
    std::cout << "Scaling factor = " << affineScalingFactor << std::endl; 
  }

  if (args.deformationFieldName.length() != 0)
    {
      try
      {
        std::cerr << "Reading vector image from:" + args.deformationFieldName << std::endl; 
        vectorImageReader->SetFileName(args.deformationFieldName);
        vectorImageReader->SetDebug(true);
        vectorImageReader->Update();
        std::cerr << "Done" << std::endl; 
      }  
      catch (itk::ExceptionObject& exceptionObject)
      {
        std::cerr << "Failed to load vector image of deformation field:" << exceptionObject << std::endl;
        return EXIT_FAILURE; 
      }
      
      if (vectorImageReader->GetOutput()->GetLargestPossibleRegion().GetSize() !=
        fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize())
        {
          std::cerr << "Vector image of deformation field, must be the same size as the fixed image" << std::endl;
          return EXIT_FAILURE; 
        }
      
      fluidTransform = FactoryType::FluidDeformableTransformType::New();
      fluidTransform->SetDeformableParameters(vectorImageReader->GetOutput());
      deformableTransform = fluidTransform;          
    }

  if (args.deformableTransformName.length() != 0)
  {
    try
    {
      std::cout << "Creating deformable transform from:" << args.deformableTransformName << std::endl;
      deformableTransform = factory->CreateTransform(args.deformableTransformName);
      std::cout << "Done" << std::endl;
    }  
    catch (itk::ExceptionObject& exceptionObject)
    {
      std::cerr << "Failed to load deformableTransform tranform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }
  }
  
  if (deformableTransform.IsNotNull()) 
  {
    if (strcmp(deformableTransform->GetNameOfClass(),"BSplineTransform") == 0)
    {
      typename FactoryType::BSplineDeformableTransformType* bSplineTransform = dynamic_cast<typename FactoryType::BSplineDeformableTransformType*>(deformableTransform.GetPointer()); 
      bSplineTransform->SetGlobalTransform(globalTransform); 
      if (args.halfNonLinearTransformation)
      {
        typename FactoryType::BSplineDeformableTransformType::ParametersType parameters = bSplineTransform->GetParameters(); 
        for (unsigned int i = 0; i < parameters.GetSize(); i++)
        {
          parameters.SetElement(i, parameters.GetElement(i)/2.0); 
        }
        bSplineTransform->SetParameters(parameters); 
      }
      if (args.invertNonLinearTransformation)
      {
        typedef itk::BSplineTransform<InputImageType, double, Dimension, float> BSplineTransformType;
        typename BSplineTransformType::Pointer inverseTransform = BSplineTransformType::New();
        inverseTransform->Initialize(fixedImageReader->GetOutput(), 1.0, 1); 
        bSplineTransform->SetInverseSearchRadius(5); 
        bSplineTransform->GetInverse(inverseTransform); 
        deformableTransform = inverseTransform; 
      }
      
    }
    else if (strcmp(deformableTransform->GetNameOfClass(),"FluidDeformableTransform") == 0)
    {
      typename FactoryType::FluidDeformableTransformType* fluidTransform = dynamic_cast<typename FactoryType::FluidDeformableTransformType*>(deformableTransform.GetPointer()); 
      fluidTransform->SetGlobalTransform(globalTransform); 
      if (args.halfNonLinearTransformation)
      {
        typename FactoryType::FluidDeformableTransformType::ParametersType parameters = fluidTransform->GetParameters(); 
        for (unsigned int i = 0; i < parameters.GetSize(); i++)
        {
          parameters.SetElement(i, parameters.GetElement(i)/2.0); 
        }
        fluidTransform->SetParameters(parameters); 
      }
      if (args.invertNonLinearTransformation)
      {
        typedef itk::FluidDeformableTransform<InputImageType, double, Dimension, float > FluidDeformableTransformType;
        typename FluidDeformableTransformType::Pointer inverseTransform = FluidDeformableTransformType::New();
        inverseTransform->Initialize(fixedImageReader->GetOutput()); 
        fluidTransform->SetInverseSearchRadius(5); 
        std::cout << fluidTransform->GetDeformationField()->GetDirection() << std::endl; 
        std::cout << inverseTransform->GetDeformationField()->GetDirection() << std::endl; 
        fluidTransform->GetInverse(inverseTransform); 
        deformableTransform = inverseTransform; 
      }
    }
    else
    {
      std::cerr << "Unknown deformable transform: " << deformableTransform->GetNameOfClass() << std::endl; 
      return EXIT_FAILURE; 
    }
  }
  
  std::cout << "Starting resampling" << std::endl;
  
  typedef typename itk::ResampleImageFilter<InputImageType, OutputImageType >   ResampleFilterType;
  typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
  
  if (args.isSymMidway)
    {
      // We need to resample using the average spacing of the fixed and moving images. 
      // Probably need to use average center, direction, size ... later. 
      typename FixedImageReaderType::Pointer* symImageReader = new typename FixedImageReaderType::Pointer[args.numberOfSymImages]; 
      typename InputImageType::SpacingType minImageSpacing; 
      typename InputImageType::RegionType::SizeType maxSize; 
      double* preciseMaxSize = new double[Dimension]; 
      for (unsigned int j = 0; j < InputImageType::ImageDimension; j++)
      {
        preciseMaxSize[j] = 0.0; 
      }
      minImageSpacing.Fill(std::numeric_limits<double>::max()); 
      maxSize.Fill(0); 
      for (int i=0; i<args.numberOfSymImages; i++)
      {
        symImageReader[i] = FixedImageReaderType::New(); 
        symImageReader[i]->SetFileName(args.symImages[i]); 
        symImageReader[i]->Update(); 
        typename InputImageType::SpacingType spacing = symImageReader[i]->GetOutput()->GetSpacing();
        typename InputImageType::RegionType::SizeType size = symImageReader[i]->GetOutput()->GetLargestPossibleRegion().GetSize();
        for (unsigned int j = 0; j < InputImageType::ImageDimension; j++)
        {
          minImageSpacing[j] = std::min<double>(minImageSpacing[j], spacing[j]); 
          if (args.doIsotropicVoxels)
          {
            minImageSpacing[j] = args.isoVoxelSize; 
          }
          preciseMaxSize[j] = std::max<double>(preciseMaxSize[j], spacing[j]*size[j]); 
        }
      }
      for (unsigned int j = 0; j < InputImageType::ImageDimension; j++)
      {
        maxSize[j] = static_cast<unsigned int>(preciseMaxSize[j]/minImageSpacing[j]+0.5); 
      }
      resampleFilter->SetSize(maxSize); 
      resampleFilter->SetOutputSpacing(minImageSpacing); 
      std::cerr << "maxSize=" << maxSize << ", minImageSpacing=" << minImageSpacing << std::endl; 
      resampleFilter->SetInput(fixedImageReader->GetOutput());
      resampleFilter->SetDefaultPixelValue(static_cast<typename InputImageType::PixelType>(0)); 
      resampleFilter->SetOutputDirection(symImageReader[0]->GetOutput()->GetDirection()); 
      resampleFilter->SetOutputOrigin(symImageReader[0]->GetOutput()->GetOrigin()); 
      delete [] symImageReader; 
      delete [] preciseMaxSize; 
    }
  else 
    {  
      if (!args.doIsotropicVoxels)
        {
          std::cout << "Using fixed image as reference frame" << std::endl;
          resampleFilter->SetUseReferenceImage(true); 
          resampleFilter->SetReferenceImage(fixedImageReader->GetOutput()); 
        }
      else
        {
          std::cout << "Resampling to isotropic voxels of size=" << niftk::ConvertToString(args.isoVoxelSize) << std::endl;
          resampleFilter->SetUseReferenceImage(false);
          
          typename InputImageType::SizeType fixedSize;
          typename InputImageType::SpacingType fixedSpacing;
          typename InputImageType::PointType fixedOrigin;
          typename InputImageType::DirectionType fixedDirection;
          typename InputImageType::IndexType fixedIndex;
          typename InputImageType::RegionType fixedRegion;
    
          typename InputImageType::SizeType newSize;
          typename InputImageType::SpacingType newSpacing;
          typename InputImageType::PointType newOrigin;
          typename InputImageType::DirectionType newDirection;
          typename InputImageType::IndexType newIndex;
          typename InputImageType::RegionType newRegion;
    
          fixedRegion = fixedImageReader->GetOutput()->GetLargestPossibleRegion();
          fixedSize = fixedRegion.GetSize();
          fixedIndex = fixedRegion.GetIndex();
          fixedSpacing = fixedImageReader->GetOutput()->GetSpacing();
          fixedOrigin = fixedImageReader->GetOutput()->GetOrigin();
          fixedDirection = fixedImageReader->GetOutput()->GetDirection();
          
          newIndex.Fill(0);
          newSpacing.Fill(args.isoVoxelSize);
          newDirection = fixedDirection;
          for (unsigned int i = 0; i < Dimension; i++)
            {
              newSize[i] = (int)((fixedSize[i] * fixedSpacing[i]) / newSpacing[i]);
              newOrigin[i] = fixedOrigin[i] + (fixedIndex[i] * fixedSpacing[i]) // this second term will almost always be zero.
                            + ((fixedSize[i]-1)*fixedSpacing[i]/2.0)
                            - ((newSize[i]-1)*newSpacing[i]/2.0);
            }
          newRegion.SetSize(newSize);
          newRegion.SetIndex(newIndex);
          
          std::cerr << "Fixed image size=" << fixedSize << ", index=" << fixedIndex << ", spacing=" << fixedSpacing << ", origin=" << fixedOrigin << ", direction=\n" << fixedDirection << std::endl; 
          std::cerr << "New image size=" << newSize << ", index=" << newIndex << ", spacing=" << newSpacing << ", origin=" << newOrigin << ", direction=\n" << newDirection << std::endl;
    
          resampleFilter->SetSize(newSize);
          resampleFilter->SetOutputDirection(newDirection); 
          resampleFilter->SetOutputOrigin(newOrigin); 
          resampleFilter->SetOutputSpacing(newSpacing); 
          resampleFilter->SetOutputStartIndex(newIndex);
        }
    }
  resampleFilter->SetInput(movingImageReader->GetOutput());
  resampleFilter->SetDefaultPixelValue(static_cast<typename OutputImageType::PixelType>(args.defaultPixelValue)); 
  if (deformableTransform.IsNotNull()) 
  {
    resampleFilter->SetTransform(deformableTransform);
  }
  else if (globalTransform.IsNotNull())
  {
    resampleFilter->SetTransform(globalTransform);
  }
  else
  {
    std::cerr << "No valid transform found." << std::endl;
    return EXIT_FAILURE; 
  }
  resampleFilter->SetInterpolator(interpolator);
  
  typedef typename itk::ShiftScaleImageFilter<InputImageType, InputImageType> ShiftScaleImageFilterType; 
  typename ShiftScaleImageFilterType::Pointer shiftScaleImageFilter = ShiftScaleImageFilterType::New(); 
      
  typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
  outputImageWriter->SetFileName(args.outputImage);
  outputImageWriter->SetInput(resampleFilter->GetOutput());
  
  if (args.affineJacobianIntensityCorrection == true)
  {
    shiftScaleImageFilter->SetInput(resampleFilter->GetOutput()); 
    shiftScaleImageFilter->SetScale(affineScalingFactor); 
    outputImageWriter->SetInput(shiftScaleImageFilter->GetOutput());
  }
  
  outputImageWriter->Update();
  std::cout << "Done" << std::endl;
  
  return EXIT_SUCCESS;     
}
/**
 * \brief Does Affine Transformation.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set the defaults
  args.finalInterp = 3;
  args.halfAffineTransformation = false;  
  args.invertAffineTransformation = false; 
  args.halfNonLinearTransformation = false;  
  args.invertNonLinearTransformation = false; 
  args.isIdentity = false;
  args.doIsotropicVoxels = false;
  args.isoVoxelSize = 1;
  args.defaultPixelValue = 0;
  args.affineJacobianIntensityCorrection = false; 
  args.isSymMidway = false; 
  args.symImages = NULL; 
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-ti") == 0){
      args.fixedImage=argv[++i];
      std::cout << "Set -ti=" << args.fixedImage << std::endl;
    }
    else if(strcmp(argv[i], "-si") == 0){
      args.movingImage=argv[++i];
      std::cout << "Set -si=" << args.movingImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -o=" << args.outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-g") == 0){
      args.globalTransformName=argv[++i];
      std::cout << "Set -g=" << args.globalTransformName << std::endl;
    }
    else if(strcmp(argv[i], "-df") == 0){
      args.deformableTransformName=argv[++i];
      std::cout << "Set -df=" << args.deformableTransformName << std::endl;
    }
    else if(strcmp(argv[i], "-di") == 0){
      args.deformationFieldName=argv[++i];
      std::cout << "Set -di=" << args.deformationFieldName << std::endl;
    }
    else if(strcmp(argv[i], "-j") == 0){
      args.finalInterp=atoi(argv[++i]);
      std::cout << "Set -j=" << niftk::ConvertToString(args.finalInterp) << std::endl;
    }
    else if(strcmp(argv[i], "-default") == 0){
      args.defaultPixelValue=atoi(argv[++i]);
      std::cout << "Set -default=" << niftk::ConvertToString(args.defaultPixelValue) << std::endl;
    }    
    else if(strcmp(argv[i], "-halfAffine") == 0){
      args.halfAffineTransformation=true;
      std::cout << "Set -halfAffine=" << niftk::ConvertToString(args.halfAffineTransformation) << std::endl;
    }
    else if(strcmp(argv[i], "-invertAffine") == 0){
      args.invertAffineTransformation=true;
      std::cout << "Set -invertAffine=" << niftk::ConvertToString(args.invertAffineTransformation) << std::endl;
    }
    else if(strcmp(argv[i], "-halfNonLinear") == 0){
      args.halfNonLinearTransformation=true;
      std::cout << "Set -halfNonLinear=" << niftk::ConvertToString(args.halfNonLinearTransformation) << std::endl;
    }
    else if(strcmp(argv[i], "-invertNonLinear") == 0){
      args.invertNonLinearTransformation=true;
      std::cout << "Set -invertNonLinear=" << niftk::ConvertToString(args.invertNonLinearTransformation) << std::endl;
    }    
    else if(strcmp(argv[i], "-half") == 0){
      args.halfAffineTransformation=true;
      args.halfNonLinearTransformation=true;
      std::cout << "Set -halfAffine=" << niftk::ConvertToString(args.halfAffineTransformation) << std::endl;
      std::cout << "Set -halfNonLinear=" << niftk::ConvertToString(args.halfNonLinearTransformation) << std::endl;
    }
    else if(strcmp(argv[i], "-invert") == 0){
      args.invertAffineTransformation=true;
      args.invertNonLinearTransformation=true;
      std::cout << "Set -invertAffine=" << niftk::ConvertToString(args.invertAffineTransformation) << std::endl;
      std::cout << "Set -invertNonLinear=" << niftk::ConvertToString(args.invertNonLinearTransformation) << std::endl;
    }
    else if(strcmp(argv[i], "-identity") == 0){
      args.isIdentity=true;
      std::cout << "Set -identity=" <<  niftk::ConvertToString(args.isIdentity) << std::endl;
    } 
    else if(strcmp(argv[i], "-iso") == 0){
      args.isoVoxelSize=atof(argv[++i]);
      args.doIsotropicVoxels=true;
      std::cout << "Set -iso=" << niftk::ConvertToString(args.isoVoxelSize) << std::endl;
      std::cout << "Set args.doIsotropicVoxels=" << niftk::ConvertToString(args.doIsotropicVoxels) << std::endl;
    }    
    else if(strcmp(argv[i], "-ajc") == 0){
      args.affineJacobianIntensityCorrection = true;
      std::cout << "Set -ajc=" <<  niftk::ConvertToString(args.affineJacobianIntensityCorrection) << std::endl;
    } 
    else if(strcmp(argv[i], "-sym_midway") == 0){
      args.isSymMidway = true; 
      args.numberOfSymImages = 2; 
      std::string nextArg = argv[i+1]; 
      if (nextArg.find_first_not_of("0123456789") == std::string::npos) 
      {
        args.numberOfSymImages = atoi(nextArg.c_str()); 
        i++; 
      }
      args.symImages = new std::string[args.numberOfSymImages]; 
      for (int index=0; index<args.numberOfSymImages; index++)
      {
        args.symImages[index]=argv[++i];
        std::cout << "Set -sym_midway=" << args.symImages[index] << std::endl;
      }
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.fixedImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if (args.movingImage.length() == 0)
    {
      std::cout << "You didn't specify a moving image, so just resampling the fixed one" << std::endl;
      args.movingImage = args.fixedImage;
    }
  
  if (args.isIdentity == false && args.globalTransformName.length() == 0 && args.deformableTransformName.length() == 0 && args.deformationFieldName.length() == 0)
    {
      std::cerr << argv[0] << "\tYou must specify at least one of -identity -g -df and -di." << std::endl;
      return EXIT_FAILURE;
    }

  if (args.isIdentity && (args.globalTransformName.length() != 0 || args.deformableTransformName.length() != 0 || args.deformationFieldName.length() != 0))
    {
      std::cerr << argv[0] << "\t-identity and any of -g -df and -di are mutually exclusive" << std::endl;
      return EXIT_FAILURE;      
    }
  
  if (args.deformableTransformName.length() != 0 && args.deformationFieldName.length() != 0)
    {
      std::cerr << argv[0] << "\t-df and -di are mutually exclusive" << std::endl;
      return EXIT_FAILURE;
    }

  int dims = 0;
  if (args.movingImage.length() > 0)
    {
      dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.movingImage);
    }
  else 
    {
      dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fixedImage);
    }
  
  if ((dims != 3) && (dims != 2))
    {
      std::cout << "Unsupported image dimension: " << dims << std::endl;
      return EXIT_FAILURE;
    }
  
  int result;

  // You could template for 2D and 3D, and all datatypes, but 64bit gcc compilers seem
  // to struggle here, so I've just done the bare minimum for now.

  itk::ImageIOBase::IOComponentType componentType;
  if (args.movingImage.length() > 0)
    {
      componentType = itk::PeekAtComponentType(args.movingImage);
    }
  else
    {
      componentType = itk::PeekAtComponentType(args.fixedImage);
    }
  
  switch (componentType)
    {
    case itk::ImageIOBase::UCHAR:
      if (dims == 2)
        result = DoMain<2, short>(args);
      else
        result = DoMain<3, short>(args);
      break;    
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        result = DoMain<2, short>(args);
      else
        result = DoMain<3, short>(args);
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        result = DoMain<2, float>(args);
      else
        result = DoMain<3, float>(args);
      break;
    default:
      std::cerr << "Unsupported pixel format" << std::endl;
      return EXIT_FAILURE;
    }
    
  delete [] args.symImages; 
    
  return result;

}


