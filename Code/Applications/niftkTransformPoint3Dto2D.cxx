/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <math.h>
#include <float.h>

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkTransform2D3D.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageFileReader.h"
#include "itkPerspectiveProjectionTransform.h"
#include "itkTransformFactory.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_STRING, "g", "filename", "The input global (affine) 3D transformation."},

  {OPT_STRING, "df", "filename", "Deformation DOF (degrees of freedom) file name."},
  {OPT_STRING, "di", "filename", "Deformation vector image transformation file name."},
  
  {OPT_STRING, "p", "filename", "The input perspective projection transformation."},

  {OPT_SWITCH, "k1", NULL, "Invert the 'x' dimension of the 2D projection"},
  {OPT_SWITCH, "k2", NULL, "Invert the 'y' dimension of the 2D projection"},
  {OPT_DOUBLE, "f", "focalLength", "Focal length of the projection in mm [1000]"},
  {OPT_DOUBLEx2, "uv", "u0,v0", "The location of the projection normal on the 2D plane in mm [0,0]"},

  {OPT_DOUBLEx3|OPT_REQ, "pt3D", "x,y,z", "The input 3D coordinate to transform and project into 2D"},

  {OPT_DONE, NULL, NULL, 
   "Program to load a 2D-3D transformation and transform a 3D point into 2D.\n"
  }
};


enum {

  O_INPUT_GLOBAL_AFFINE_TRANSFORM_3D,

  O_INPUT_DOF_TRANSFORM_3D,
  O_INPUT_DISPLACEMENT_TRANSFORM_3D,

  O_INPUT_PERSPECTIVE,

  O_K1,
  O_K2,
  O_FOCAL_LENGTH,
  O_NORMAL_POSITION,

  O_INPUT_COORDINATE_3D
};


int main( int argc, char *argv[] )
{
  std::string fileGlobalAffine3D;
  std::string fileDeformationDOF3D;
  std::string fileDisplacementField3D;
  std::string filePerspectiveTransform;

  bool k1negative;              // Invert the 'x' dimension of the 2D projection
  bool k2negative;              // Invert the 'y' dimension of the 2D projection

  double focalLength = 1000.;	// The focal length of the 3D to 2D projection
  double *normalPosn = 0;       // The position of the normal in the 2D plane

  double *inPoint3D = 0;          // The input 3D coordinate to transform and project into 2D

  typedef short VoxelType;
  const unsigned int ImageDimension = 3;
  typedef itk::Image< VoxelType, ImageDimension >  InputImageType; 

  typedef itk::Vector<float, ImageDimension>            VectorPixelType;
  typedef itk::Image<VectorPixelType, ImageDimension>   VectorImageType;
  typedef itk::ImageFileReader < VectorImageType > VectorImageReaderType;
  
  VectorImageReaderType::Pointer vectorImageReader = VectorImageReaderType::New();

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_INPUT_GLOBAL_AFFINE_TRANSFORM_3D, fileGlobalAffine3D );
  

  if (    CommandLineOptions.GetArgument( O_INPUT_DOF_TRANSFORM_3D, fileDeformationDOF3D )
       && CommandLineOptions.GetArgument( O_INPUT_DISPLACEMENT_TRANSFORM_3D, fileDisplacementField3D )) {

    std::cout << "Options '-df' and '-di' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }

  CommandLineOptions.GetArgument( O_INPUT_PERSPECTIVE, filePerspectiveTransform );

  bool flgSet_k1negative = CommandLineOptions.GetArgument( O_K1, k1negative );
  bool flgSet_k2negative = CommandLineOptions.GetArgument( O_K2, k2negative );
  bool flgSet_focalLength = CommandLineOptions.GetArgument( O_FOCAL_LENGTH, focalLength );
  bool flgSet_normalPosn = CommandLineOptions.GetArgument( O_NORMAL_POSITION, normalPosn );

  if ( filePerspectiveTransform.length() 
       && ( flgSet_k1negative ||
            flgSet_k2negative || 
            flgSet_focalLength ||
            flgSet_normalPosn ) ) {

    std::cout << "Options '-p' and any of '-k1','-k2', -f' and '-uv' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }

  CommandLineOptions.GetArgument( O_INPUT_COORDINATE_3D, inPoint3D );


  // Create the 2D-3D transform object
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::Transform2D3D< double > Transform2D3DType;

  Transform2D3DType::Pointer transform2D3D = Transform2D3DType::New();

  transform2D3D->Print( std::cout );


  // Create the factory used to load the various transformations
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef  itk::ImageRegistrationFactory<InputImageType, ImageDimension, double> FactoryType;
  
  // The factory.

   FactoryType::Pointer factory = FactoryType::New();

  
   // Load the global affine transformation
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileGlobalAffine3D.length() > 0 ) {

    FactoryType::TransformType::Pointer globalTransform; 

    try {

      std::cout << "Creating global transform from:" << fileGlobalAffine3D << std::endl; 
      globalTransform = factory->CreateTransform( fileGlobalAffine3D );
      std::cout << "Done" << std::endl; 
    }  
    
    catch (itk::ExceptionObject& exceptionObject) {
      std::cerr << "Failed to load global tranform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }
    
    Transform2D3DType::GlobalAffineTransformType::Pointer affineTransform 
      = dynamic_cast< Transform2D3DType::GlobalAffineTransformType *>(globalTransform.GetPointer()); 

    std::cout << affineTransform->GetFullAffineMatrix() << std::endl; 

    transform2D3D->SetGlobalAffineTransform( affineTransform );
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

    transform2D3D->SetDeformableTransform( fluidTransform );
  }

  
  // Load the non-rigid deformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  else if ( fileDeformationDOF3D.length() > 0 ) {

    FactoryType::TransformType::Pointer deformableTransform; 
  
    try {

      std::cout << "Creating deformable transform from: " << fileDeformationDOF3D << std::endl;
      deformableTransform = factory->CreateTransform(fileDeformationDOF3D);
      std::cout << "Done" << std::endl;
    }  
    catch (itk::ExceptionObject& exceptionObject) {

      std::cerr << "Failed to load deformableTransform tranform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }

    transform2D3D->SetDeformableTransform( deformableTransform );
  }


  // Load the perspective transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  Transform2D3DType::PerspectiveProjectionTransformType::Pointer perspectiveTransform;

  if (filePerspectiveTransform.length() != 0) {

    FactoryType::TransformType::Pointer perspProjTransform; 

    try {

      std::cout << "Creating perspective transform from: " << filePerspectiveTransform << std::endl;
      perspProjTransform = factory->CreateTransform( filePerspectiveTransform );
      std::cout << "Done" << std::endl;
    }  
    catch (itk::ExceptionObject& exceptionObject) {

      std::cerr << "Failed to load perspective tranform:" << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }

    perspectiveTransform
      = dynamic_cast< Transform2D3DType::PerspectiveProjectionTransformType *>(perspProjTransform.GetPointer()); 
  }

  else {
    perspectiveTransform = Transform2D3DType::PerspectiveProjectionTransformType::New();

    perspectiveTransform->SetFocalDistance(focalLength);

    if ( normalPosn )
      perspectiveTransform->SetOriginIn2D( normalPosn[0], normalPosn[1] );
    else
      perspectiveTransform->SetOriginIn2D( 0., 0. );

    if ( k1negative )
      perspectiveTransform->SetK1IsNegative();
    if ( k2negative )
      perspectiveTransform->SetK2IsNegative();
  }


  perspectiveTransform->Print(std::cout);

  transform2D3D->SetPerspectiveTransform( perspectiveTransform );


  // Transform the input coordinate
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Transform2D3DType::InputPointType point3D;
  Transform2D3DType::OutputPointType point2D;

  point3D[0] = inPoint3D[0];
  point3D[1] = inPoint3D[1];
  point3D[2] = inPoint3D[2];

  point2D = transform2D3D->TransformPoint( point3D );

  std::cout << "3D coordinate: ("
                                << niftk::ConvertToString(point3D[0]) << ", "
                                << niftk::ConvertToString(point3D[1]) << ", "
                                << niftk::ConvertToString(point3D[2])
                                << ") projects to 2D coordinate: ("
                                << niftk::ConvertToString(point2D[0]) << ", "
                                << niftk::ConvertToString(point2D[1]) << ")" << std::endl;
 
  return EXIT_SUCCESS;     
}
