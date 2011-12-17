
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: ad                  $
 $Date:: 2011-09-20 14:34:44 +#$
 $Rev:: 7333                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImageRegistrationFactory.h"
#include "itkImageFileReader.h"
#include "itkTransformFactory.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "invert", NULL, "Invert the input global (affine) 3D transformation."},
  {OPT_STRING, "g", "filename", "The input global (affine) 3D transformation."},

  {OPT_STRING, "df", "filename", "Deformation DOF (degrees of freedom) file name."},
  {OPT_STRING, "di", "filename", "Deformation vector image transformation file name."},

  {OPT_STRING, "o", "filename", "Write the transformed point to a file."},

  {OPT_DOUBLEx3|OPT_REQ, "pt3D", "x,y,z", "The input 3D coordinate to transform"},

  {OPT_DONE, NULL, NULL, 
   "Program to load a 2D-3D transformation and transform a 3D point into 2D.\n"
  }
};


enum { 
  O_INVERT,
  O_INPUT_GLOBAL_AFFINE_TRANSFORM_3D,

  O_INPUT_DOF_TRANSFORM_3D,
  O_INPUT_DISPLACEMENT_TRANSFORM_3D,

  O_OUTPUT_TRANSFORMED_POINT,

  O_INPUT_COORDINATE_3D
};


int main( int argc, char *argv[] )
{
  bool flgInvert;

  std::string fileOutputTransformedPoint;
  std::string fileGlobalAffine3D;
  std::string fileDeformationDOF3D;
  std::string fileDisplacementField3D;

  double *inPoint3D = 0;          // The input 3D coordinate to transform and project into 2D

  typedef short VoxelType;
  const unsigned int ImageDimension = 2;
  typedef itk::Image< VoxelType, ImageDimension >  InputImageType; 

  typedef itk::Vector<float, ImageDimension>            VectorPixelType;
  typedef itk::Image<VectorPixelType, ImageDimension>   VectorImageType;
  typedef itk::ImageFileReader < VectorImageType > VectorImageReaderType;
  
  typedef itk::ImageRegistrationFactory<InputImageType, ImageDimension, double> FactoryType;
  
  typedef FactoryType::EulerAffineTransformType     EulerAffineTransformType;
  typedef EulerAffineTransformType*                 EulerAffineTransformPointer;

  typedef FactoryType::TransformType TransformType; 

  TransformType::Pointer globalTransform; 
  TransformType::Pointer deformableTransform; 
  
  VectorImageReaderType::Pointer vectorImageReader = VectorImageReaderType::New();

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_INVERT, flgInvert );

  CommandLineOptions.GetArgument( O_INPUT_GLOBAL_AFFINE_TRANSFORM_3D, fileGlobalAffine3D );

  if (    CommandLineOptions.GetArgument( O_INPUT_DOF_TRANSFORM_3D, fileDeformationDOF3D )
       && CommandLineOptions.GetArgument( O_INPUT_DISPLACEMENT_TRANSFORM_3D, fileDisplacementField3D )) {

    std::cerr <<"Options '-df' and '-di' are mutually exclusive";
    return EXIT_FAILURE;
  }

  CommandLineOptions.GetArgument( O_OUTPUT_TRANSFORMED_POINT, fileOutputTransformedPoint );

  CommandLineOptions.GetArgument( O_INPUT_COORDINATE_3D, inPoint3D );


  // Create the factory used to load the various transformations
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // The factory.

   FactoryType::Pointer factory = FactoryType::New();

  
   // Load the global affine transformation
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileGlobalAffine3D.length() > 0 ) {

    try {

      std::cout << "Creating global transform from:" << fileGlobalAffine3D << std::endl; 
      globalTransform = factory->CreateTransform( fileGlobalAffine3D );
      std::cout << "Done" << std::endl; 
    }  
    
    catch (itk::ExceptionObject& exceptionObject) {
      std::cerr << "Failed to load global tranform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }
    
    EulerAffineTransformPointer affineTransform 
      = dynamic_cast<EulerAffineTransformPointer>(globalTransform.GetPointer()); 

    if (flgInvert) {
      affineTransform->InvertTransformationMatrix(); 
      std::cout << "inverted:" << std::endl << affineTransform->GetFullAffineMatrix() << std::endl; 
    }

    std::cout << affineTransform->GetFullAffineMatrix() << std::endl; 
  }

  // Displacement field transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileDisplacementField3D.length() > 0) {

    VectorImageType::Pointer vectorImage;

    try {
      std::cerr << "Reading vector image from: " << fileDisplacementField3D << std::endl; 
      vectorImageReader->SetFileName( fileDisplacementField3D );
      vectorImageReader->SetDebug(true);
      vectorImageReader->Update();
      std::cerr << "Done" << std::endl; 
    }  
    
    catch (itk::ExceptionObject& exceptionObject) {
      
      std::cerr << "Failed to load vector image of deformation field:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }
      
    FactoryType::FluidDeformableTransformType::Pointer fluidTransform;

    fluidTransform = FactoryType::FluidDeformableTransformType::New();
    fluidTransform->SetParametersFromField(vectorImageReader->GetOutput(), true);

    deformableTransform = fluidTransform;
  }

  
  // Load the non-rigid deformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  else if ( fileDeformationDOF3D.length() > 0 ) {

    try {

      std::cout << "Creating deformable transform from: " + fileDeformationDOF3D;
      deformableTransform = factory->CreateTransform(fileDeformationDOF3D);
      std::cout << "Done";
    }  
    catch (itk::ExceptionObject& exceptionObject) {

      std::cerr << "Failed to load deformableTransform tranform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }
  }


  TransformType::InputPointType point3D;

  point3D[0] = inPoint3D[0];
  point3D[1] = inPoint3D[1];


  // Transform the input coordinate using the global affine transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileGlobalAffine3D.length() > 0 ) {

    EulerAffineTransformType::OutputPointType globalTransformedPoint3D;

    globalTransformedPoint3D = globalTransform->TransformPoint( point3D );

    std::cout << "3D coordinate: ("
				  << niftk::ConvertToString(point3D[0]) + ", "
				  << niftk::ConvertToString(point3D[1])
				  << ") globally transforms to coordinate: ("
				  << niftk::ConvertToString(globalTransformedPoint3D[0]) << ", "
				  << niftk::ConvertToString(globalTransformedPoint3D[1]) << ")";

    point3D[0] = globalTransformedPoint3D[0];
    point3D[1] = globalTransformedPoint3D[1];
  }


  // Transform the globally transformed coordinate using the deforming transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( (fileDisplacementField3D.length() > 0) || (fileDeformationDOF3D.length() > 0) ) {

    TransformType::OutputPointType deformedPoint3D;

    deformedPoint3D = deformableTransform->TransformPoint( point3D );

    std::cout << "3D coordinate: ("
				  << niftk::ConvertToString(point3D[0]) << ", "
				  << niftk::ConvertToString(point3D[1])
				  << ") non-rigidly transforms to coordinate: ("
				  << niftk::ConvertToString(deformedPoint3D[0]) << ", "
				  << niftk::ConvertToString(deformedPoint3D[1]) << ")";

    point3D[0] = deformedPoint3D[0];
    point3D[1] = deformedPoint3D[1];
  }


  // Write the point to a file?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputTransformedPoint.length() > 0 ) {


    std::fstream fout;

    fout.open(fileOutputTransformedPoint.c_str(), std::ios::out);
    
    if ((! fout) || fout.bad()) {
      std::cerr << "Failed to open file: "
				     << fileOutputTransformedPoint.c_str();
      return EXIT_FAILURE;
    }
    
    fout << point3D[0] << " " << point3D[1] << " 0" << std::endl;;
    
    fout.close();
  }


  return EXIT_SUCCESS;     
}
