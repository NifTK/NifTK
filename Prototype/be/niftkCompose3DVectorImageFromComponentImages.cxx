/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-11-16 18:04:05 +0100 (Fr, 28 Mai 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: be $

 Original author   : bjoern.eiben.10@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkVector.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCompose3DVectorImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkMultiplyByConstantImageFilter.h"
#include "itkFlipImageFilter.h"


#include "boost/filesystem.hpp"

namespace bfs = boost::filesystem;


/*
 * Enumeration for the relation of the Command line argument description and
 * the
 */
enum
{
	O_X_COMP_IMG = 0,
	O_Y_COMP_IMG,
	O_Z_COMP_IMG,
	O_OUT_IMG,
	O_INVERT_X_IMG,
	O_INVERT_Y_IMG,
	O_INVERT_Z_IMG,
	O_FLIP_COMPONENTS_XY,   // which is equivalent to rotation by 180 degrees
};



/*
 * Command line argument description
 * (type), (flag-name), (short desc.), (long desc.)
 */
niftk::CommandLineArgumentDescription clArgList[] =
{
	{ OPT_STRING | OPT_REQ, "x", "filename", "Image file name which holds the x component of the result image." },
	{ OPT_STRING | OPT_REQ, "y", "filename", "Image file name which holds the y component of the result image." },
	{ OPT_STRING | OPT_REQ, "z", "filename", "Image file name which holds the z component of the result image." },
	{ OPT_STRING | OPT_REQ, "o", "filename", "The output image file name."                                      },
	{ OPT_SWITCH,           "invertX", 0,    "Invert the values of the x image."                                },
	{ OPT_SWITCH,           "invertY", 0,    "Invert the values of the y image."                                },
	{ OPT_SWITCH,           "invertZ", 0,    "Invert the values of the z image."                                },
	{ OPT_SWITCH,           "flipXY",  0,    "Flip the component images."                                       },
	{ OPT_DONE, NULL, NULL,                  "Program to combine three images into one vector image\n\n"
				                             "  - The input image pixel type is assumed to be 'float'\n"
	                                         "  - Internally the itk filter itkCompose3DVectorImageFilter\n"
			                                 "    is used\n\n"                                                  }
};





/*
 * main
 */
int main(int argc, char** argv)
{
	/* Variables holding the image file names */
	std::string strXCompImageFileName;
	std::string strYCompImageFileName;
	std::string strZCompImageFileName;
	std::string strOutputImageFileName;

	bool bInvertXImg = false;
	bool bInvertYImg = false;
	bool bInvertZImg = false;

	bool bFlipComponentsXY = false;


	/* Build the command line parser */
	niftk::CommandLineParser CommandLineOptions( argc, argv, clArgList, true );

	/* Parse the command line  */
	CommandLineOptions.GetArgument( O_X_COMP_IMG,         strXCompImageFileName  );
	CommandLineOptions.GetArgument( O_Y_COMP_IMG,         strYCompImageFileName  );
	CommandLineOptions.GetArgument( O_Z_COMP_IMG,         strZCompImageFileName  );
	CommandLineOptions.GetArgument( O_OUT_IMG,            strOutputImageFileName );
	CommandLineOptions.GetArgument( O_INVERT_X_IMG,       bInvertXImg            );
	CommandLineOptions.GetArgument( O_INVERT_Y_IMG,       bInvertYImg            );
	CommandLineOptions.GetArgument( O_INVERT_Z_IMG,       bInvertZImg            );
	CommandLineOptions.GetArgument( O_FLIP_COMPONENTS_XY, bFlipComponentsXY      );



	/*
	 * Some checking before getting started
	 */
	bfs::path inPathImageX( strXCompImageFileName );
	bfs::path inPathImageY( strYCompImageFileName );
	bfs::path inPathImageZ( strZCompImageFileName );

	if ( ( ! bfs::exists( inPathImageX ) ) ||
		 ( ! bfs::exists( inPathImageY ) ) ||
		 ( ! bfs::exists( inPathImageZ ) )    )
	{
		std::cout << "The input images do not exist!" << std::endl
				  << "Exiting program." << std::endl;

		return EXIT_FAILURE;
	}



	/*
	 * Prepare images
	 */
	const unsigned int DIMENSION = 3;

	typedef float                                       InputPixelType;
	typedef itk::Vector< float, DIMENSION >             OutputPixelType;

	typedef itk::Image< InputPixelType,  DIMENSION >    InputImageType;
	typedef InputImageType::Pointer			            InputImagePointerType;
	typedef InputImageType::PointType                   InputImagePointType;      // for origin
	typedef InputImageType::SpacingType                 InputImageSpacingType;    // for spacing
	typedef InputImageType::DirectionType               InputImageDirectionType;  // for direction

	typedef itk::Image< OutputPixelType, DIMENSION >    OutputImageType;
	typedef OutputImageType::Pointer		            OutputImagePointerType;



	/*
	 * Prepare the image readers
	 */
	typedef itk::ImageFileReader< InputImageType >      InputReaderType;
	typedef InputReaderType::Pointer                    InputReaderPointerType;

	InputReaderPointerType xReader = InputReaderType::New();
	InputReaderPointerType yReader = InputReaderType::New();
	InputReaderPointerType zReader = InputReaderType::New();

	/* Feed them with the image names */
	xReader->SetFileName( strXCompImageFileName );
	yReader->SetFileName( strYCompImageFileName );
	zReader->SetFileName( strZCompImageFileName );

	/*
	 * Read the images, origin is needed by flip image filter
	 */
	try
	{
		xReader->Update();
		yReader->Update();
		zReader->Update();
	}
	catch ( itk::ExceptionObject & e )
	{
		std::cout << e << std::endl;
		return EXIT_FAILURE;
	}

	/*
	 * Extract the information from the input images
	 */
	InputImagePointType xOrigin = xReader->GetOutput()->GetOrigin();	
	InputImagePointType yOrigin = yReader->GetOutput()->GetOrigin();
	InputImagePointType zOrigin = zReader->GetOutput()->GetOrigin();
	
	InputImageSpacingType xSpacing = xReader->GetOutput()->GetSpacing();
	InputImageSpacingType ySpacing = yReader->GetOutput()->GetSpacing();
	InputImageSpacingType zSpacing = zReader->GetOutput()->GetSpacing();

	InputImageDirectionType xDirection = xReader->GetOutput()->GetDirection();
	InputImageDirectionType yDirection = yReader->GetOutput()->GetDirection();
	InputImageDirectionType zDirection = zReader->GetOutput()->GetDirection();

	std::cout << "Origin of input images" << std::endl
		      << " x: " << xOrigin << std::endl
			  << " y: " << yOrigin << std::endl
			  << " z: " << zOrigin << std::endl;
	
	std::cout << "Spacing of input images" << std::endl
		      << " x: " << xSpacing << std::endl
			  << " y: " << ySpacing << std::endl
			  << " z: " << zSpacing << std::endl;
	
	std::cout << "Direction of input images" << std::endl
		      << " x: " << xDirection << std::endl
			  << " y: " << yDirection << std::endl
			  << " z: " << zDirection << std::endl;

	/*
	 * Check if the image data are the same...
	 */
	if ( (xOrigin != yOrigin) || (yOrigin != zOrigin) || (xOrigin != zOrigin) )
	{
		std::cout << "Origin of component images is NOT identical, consider manual resampling to solve this problem." << std::endl;
		return EXIT_FAILURE;
	}

	if ( (xSpacing != ySpacing) || (ySpacing != zSpacing) || (xSpacing != zSpacing) ) 
	{
		std::cout << "Spacing of component images is NOT identical, consider manual resampling to solve this problem." << std::endl;
		return EXIT_FAILURE;
	}

	if ( (xDirection != yDirection) || (yDirection != zDirection) || (xDirection != zDirection) ) 
	{
		std::cout << "Direction of component images is NOT identical, consider manual resampling to solve this problem." << std::endl;
		return EXIT_FAILURE;
	}


	/*
	 * Prepare the writer
	 */
	typedef itk::ImageFileWriter< OutputImageType > WriterType;
	typedef WriterType::Pointer                     WriterPointerType;

	WriterPointerType imageWriter = WriterType::New();
	imageWriter->SetFileName( strOutputImageFileName );



	/*
	 * Prepare the multiplication filter
	 */
	typedef itk::MultiplyByConstantImageFilter< InputImageType,
			                                    InputImageType::PixelType,
			                                    InputImageType>                 MultiplicationFilterType;
	typedef MultiplicationFilterType::Pointer                                   MultiplicationFilterPointerType;

	MultiplicationFilterPointerType xInversionFilter = MultiplicationFilterType::New();
	MultiplicationFilterPointerType yInversionFilter = MultiplicationFilterType::New();;
	MultiplicationFilterPointerType zInversionFilter = MultiplicationFilterType::New();;

	xInversionFilter->SetConstant( -1.0f );
	yInversionFilter->SetConstant( -1.0f );
	zInversionFilter->SetConstant( -1.0f );



	/*
	 * Prepare the flipping filter
	 */
	typedef itk::FlipImageFilter< InputImageType >  FlipFilterType;
	typedef FlipFilterType::Pointer                 FlipFilterPointerType;

	FlipFilterPointerType xFlipFilter = FlipFilterType::New();
	FlipFilterPointerType yFlipFilter = FlipFilterType::New();
	FlipFilterPointerType zFlipFilter = FlipFilterType::New();

	FlipFilterType::FlipAxesArrayType axes;
	axes[0] = 1;
	axes[1] = 1;
	
	xFlipFilter->SetFlipAxes( axes );
	yFlipFilter->SetFlipAxes( axes ); 
	zFlipFilter->SetFlipAxes( axes );


	/*
	 * Prepare the composition filter
	 */
	typedef itk::Compose3DVectorImageFilter< InputImageType, OutputImageType > ComposeFilterType;
	typedef ComposeFilterType::Pointer                                         ComposeFilterPointerType;

	ComposeFilterPointerType composer = ComposeFilterType::New();


	/*
	 * Construct pipeline
	 */

	// x-component image
	if ( bInvertXImg )
	{
		xInversionFilter->SetInput ( xReader->GetOutput()          );

		if ( bFlipComponentsXY )
		{
			xFlipFilter->SetInput ( xInversionFilter->GetOutput() );
			composer   ->SetInput1( xFlipFilter->GetOutput()      );
		}
		else
		{
			composer->SetInput1( xInversionFilter->GetOutput() );
		}
	}
	else
	{
		if ( bFlipComponentsXY )
		{
			xFlipFilter->SetInput ( xReader->GetOutput()     );
			composer   ->SetInput1( xFlipFilter->GetOutput() );
		}	
		else
		{
			composer->SetInput1( xReader->GetOutput() );
		}
	}


	// y-component image
	if ( bInvertYImg )
	{
		yInversionFilter->SetInput ( yReader->GetOutput()          );

		if ( bFlipComponentsXY )
		{
			yFlipFilter->SetInput ( yInversionFilter->GetOutput() );
			composer   ->SetInput2( yFlipFilter->GetOutput()      );
		}
		else
		{
			composer->SetInput2( yInversionFilter->GetOutput() );
		}
	}
	else
	{
		if ( bFlipComponentsXY )
		{
			yFlipFilter->SetInput ( yReader->GetOutput()     );
			composer   ->SetInput2( yFlipFilter->GetOutput() );
		}	
		else
		{
			composer->SetInput2( yReader->GetOutput() );
		}
	}


	// z-component image
	if ( bInvertZImg )
	{
		zInversionFilter->SetInput ( zReader->GetOutput()          );

		if ( bFlipComponentsXY )
		{
			zFlipFilter->SetInput ( zInversionFilter->GetOutput() );
			composer   ->SetInput3( zFlipFilter->GetOutput()      );
		}
		else
		{
			composer->SetInput3( zInversionFilter->GetOutput() );
		}
	}
	else
	{
		if ( bFlipComponentsXY )
		{
			zFlipFilter->SetInput ( zReader->GetOutput()     );
			composer   ->SetInput3( zFlipFilter->GetOutput() );
		}	
		else
		{
			composer->SetInput3( zReader->GetOutput() );
		}
	}


	OutputImagePointerType outImage = composer->GetOutput();

	try
	{
		outImage->Update();
	}
	catch( itk::ExceptionObject & e )
	{
		std::cout << e << std::endl;
		return EXIT_FAILURE;
	}

	/*
	 * Feed the outoput image with the original image properties
	 */
	outImage->DisconnectPipeline();
	outImage->SetSpacing  ( xSpacing   );
	outImage->SetDirection( xDirection );
	outImage->SetOrigin   ( xOrigin    );

	imageWriter->SetInput( outImage );


	/* Write the image */
	try
	{
		imageWriter->Update();
	}
	catch ( itk::ExceptionObject & e )
	{
		std::cout << e << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

