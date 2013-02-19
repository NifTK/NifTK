/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/** 
 * Create a transformation from a user specified list of parameters.
 */

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkCommand.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"

#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkGradientDescentOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"

#include "itkMaskedImageRegistrationMethod.h"

#include "itkTransformFileWriter.h"
#include "itkEuler3DTransform.h"

#include "itkPCADeformationModelTransform.h"
#include "itkTranslationPCADeformationModelTransform.h"



struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_INT|OPT_REQ, "type",  "number", "The type of the ITK transform to create. Options are:\n"
   " 0: Translation\n"
   " 1: Rigid, so rotations and translations, 3DOF in 2D and 6DOF in 3D.\n"
   " 2: Rigid plus scale, 5DOF in 2D, 9DOF in 3D.\n"
   " 3: Affine. 7DOF in 2D, 12DOF in 3D.\n"
   " 4: B-spline Free-form Deformation\n"
   " 5: Fluid transformation\n"
   " 6: PCA deformation transformation\n"
   " 7: PCA deformation transformation with translation\n\n"
  },

  {OPT_DOUBLE, "grid",  "size", "The resolution/spacing in mm of the b-spline control point grid."},

  {OPT_STRING, "ti",  "filename", "Target/Fixed image over which the transformation is defined."},

  {OPT_STRING|OPT_REQ, "ot", "filename", "The output transformation filename."},

  {OPT_DOUBLE|OPT_LONELY|OPT_REQ, NULL, "parameters", "The list of transformation parameters."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Program to create a transformation from a user specified list of parameters."
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_TRANSFORM_TYPE,

  O_GRID_SIZE,

  O_TARGET_IMAGE,

  O_OUTPUT_TRANSFORMATION,

  O_PARAMETERS,
  O_MORE
};


// -------------------------------------------------------------------
// main()
// -------------------------------------------------------------------

int main(int argc, char** argv)
{

  bool debug(false);                    // Output debugging information

  double parameter = 0;   
  double gridSize = 0;		// B-spline control grid size

  unsigned int i = 0;		// Loop counter variable
  unsigned int nParameters = 0;	// The number of input parameters

  int arg;			// Index of arguments in command line 
  int transformType = 0;	// The type of transform to create

  std::string fileOutputTransformation;
  std::string fileFixedImage;

  const unsigned int ImageDimension = 3; // 3D images
  typedef double VectorComponentType; // Eigen vector displacement type

  typedef double PixelType;
  typedef itk::Image< PixelType, ImageDimension >  InputImageType; 
  InputImageType::Pointer fixedImage;

  typedef itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef itk::ImageRegistrationFactory<InputImageType, ImageDimension, double> FactoryType;

  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_TRANSFORM_TYPE, transformType);
  CommandLineOptions.GetArgument(O_GRID_SIZE, gridSize);
  CommandLineOptions.GetArgument(O_TARGET_IMAGE, fileFixedImage);
  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMATION, fileOutputTransformation);

  

  // Get the input parameters

  FactoryType::TransformType::ParametersType parameters; 
  
  CommandLineOptions.GetArgument(O_PARAMETERS, parameter);
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {		   // Many parameters
    nParameters = argc - arg + 1;

    parameters.SetSize( nParameters );
    parameters.Fill( 0.0 );

    if (debug) 
      std::cout << "Input Parameters: " << std::endl;

    for (i=0; i<nParameters; i++) {
      parameters.SetElement(i, niftk::ConvertToDouble( argv[arg - 1 + i] ));

      if (debug)
	std::cout <<  niftk::ConvertToString( (int) i+1) << " " + niftk::ConvertToString( parameters[i] ) << std::endl;
    }
  }
  else if (parameter) { // Single deformation field
    nParameters = 1;

    parameters.SetSize( nParameters );
    parameters[i] = parameter;

    std::cout << "Input Parameter: " << niftk::ConvertToString( parameters[0] ) << std::endl;
  }
  else {
    nParameters = 0;
    parameters.SetSize(0);
  }

  if (debug)
    std::cout << std::endl << "Parameters: " << parameters << std::endl << std::endl << std::endl;


  // Read the fixed/target image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileFixedImage.length() > 0 ) {
    
    FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
    fixedImageReader->SetFileName(fileFixedImage);
    
    try 
      { 
	std::cout << "Loading fixed image: " << fileFixedImage << std::endl;
	fixedImageReader->Update();
      } 
    
    catch( itk::ExceptionObject & err ) { 
      
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                
    
    fixedImage = fixedImageReader->GetOutput();
  }


  // Create a transformation of the correct type
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // The factory.

  FactoryType::Pointer factory = FactoryType::New();
  FactoryType::TransformType::Pointer transform; 
  
  // According to transform type

  switch (transformType)
    {
    case 0: {			// Translation
      std::cout << "Creating translation transformation" << std::endl;
      transform = factory->CreateTransform(itk::TRANSLATION);

      break;
    }

    case 1: {			// Rigid, so rotations and translations, 3DOF in 2D and 6DOF in 3D.
      std::cout << "Creating rigid transformation (3DOF in 2D and 6DOF in 3D)" << std::endl;
      transform = factory->CreateTransform(itk::RIGID);

      break;
    }

    case 2: {			// Rigid plus scale, 5DOF in 2D, 9DOF in 3D.
      std::cout << "Creating rigid plus scale transformation (5DOF in 2D, 9DOF in 3D)" << std::endl;
      transform = factory->CreateTransform(itk::RIGID_SCALE);

      break;
    }

    case 3: {			// Affine. 7DOF in 2D, 12DOF in 3D.
      std::cout << "Creating affine transformation (7DOF in 2D, 12DOF in 3D)" << std::endl;
      transform = factory->CreateTransform(itk::AFFINE);

      break;
    }

    case 4: {			// B-spline Free-form Deformation
      std::cout << "Creating b-spline free-form transformation" << std::endl;

      if ( fileFixedImage.length() == 0 ) {
	std::cerr <<"B-Spline transformation requires a target image.";
	return -1;
      }

      if ( gridSize == 0 ) {
	std::cerr <<"B-Spline control point grid resolution in mm must be specified.";
	return -1;
      }

      typedef itk::BSplineTransform<InputImageType, double, ImageDimension, float > BSplineTransformType;
      transform = BSplineTransformType::New();

      BSplineTransformType *bsplineTransform = static_cast<BSplineTransformType*>(transform.GetPointer());
      bsplineTransform->Initialize(fixedImage.GetPointer(), gridSize, 1);

      FactoryType::TransformType::Pointer globalTransform = factory->CreateTransform(itk::AFFINE);
      bsplineTransform->SetGlobalTransform(globalTransform); 
  
      break;
    }

    case 5: {			// Fluid transformation
      std::cout << "Creating fluid transformation" << std::endl;

      if ( fileFixedImage.length() == 0 ) {
	std::cerr <<"Fluid transformation requires a target image.";
	return -1;
      }

      typedef itk::FluidDeformableTransform< InputImageType, double, ImageDimension, float > FluidTransformType;
      transform = FluidTransformType::New();
      
      FluidTransformType *fluidTransform = static_cast< FluidTransformType* >( transform.GetPointer() );
	
      fluidTransform->Initialize(fixedImage.GetPointer());

      FactoryType::TransformType::Pointer globalTransform = factory->CreateTransform(itk::AFFINE);
      fluidTransform->SetGlobalTransform(globalTransform); 

      break;
    }

    case 6: {			// PCA deformation transformation
      std::cout << "Creating PCA deformation transformation" << std::endl;

      typedef itk::PCADeformationModelTransform< VectorComponentType, ImageDimension > PCATransformType;
      transform = PCATransformType::New();

      break;
    }

    case 7: {			// PCA deformation transformation with translation
      std::cout << "Creating PCA deformation with translation transformation" << std::endl;

      typedef itk::TranslationPCADeformationModelTransform< VectorComponentType, ImageDimension > TranslationPCATransformType;
      transform = TranslationPCATransformType::New();

      break;
    }

    default: {
      std::cerr <<"Unrecognised transformation type.";
      return -1;
    }
    }

  try 
    {
      transform->SetParameters(parameters);
    }
  catch( itk::ExceptionObject & excp )
    {
      std::cerr <<"Exception thrown on setting parameters";
      std::cerr << excp << std::endl; 
      return -1;
    }

  if (debug) {
    std::cout << std::endl << "Transform: " << std::endl;
    transform->Print(std::cout);
  }


  // Write the transformation to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  itk::TransformFileWriter::Pointer transformWriter;
  transformWriter = itk::TransformFileWriter::New();
      
  transformWriter->SetFileName( fileOutputTransformation );
  transformWriter->SetInput( transform );

  std::cout << "Writing transformation to file: " << fileOutputTransformation << std::endl;
  transformWriter->Update();

  return 0;
}
