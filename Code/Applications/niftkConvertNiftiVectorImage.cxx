/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNiftiImageIO.h"
#include "itkVector.h"
#include "itkExceptionObject.h"
#include "itkImageAdaptor.h"
#include "itkMultiplyByConstantImageFilter.h"

#include "CommandLineParser.h"
#include "itkDisplacementVectorCoordinateAdaptionPixelAccessor.h"

#include "nifti1_io.h"

#include "boost/filesystem.hpp"

/*
 * Purpose: Convert a vector image saved as a nifti-file into a different format...
 *          The issue with itk is, that it does not recognise a vector valued image
 *          purely by the image dimensions. For itk the intent_code must be correctly
 *          set to NIFIT_INTENT_VECTOR = 1007. For correct itk-internal image handling
 *          an intermediate image is generated () prior to reading.
 *
 *          The file extension of the output image file name will be stripped and
 *          replaced with "nii" to define an inter mediate image.
 */

/*
 * Stucture defining the command line options
 */
struct niftk::CommandLineArgumentDescription clArgList[] =
{
	{ OPT_STRING | OPT_REQ, "i", "inputFilename.nii",  "Filename of the input image."},
	{ OPT_STRING | OPT_REQ, "o", "outputFilename.nii", "Filename of the output image."},
 	{ OPT_DONE, NULL, NULL,                            "Converts a 3D nifti vector valued image into another format." }
};


/*
 * Keep track of the option order
 */
enum
{
	O_INPUT_IMAGE = 0,
	O_OUTPUT_IMAGE,
};


/*
 * Define a shortcut for the boost filesystem namespace
 */
namespace bfs = boost::filesystem;


/*
 * Get to work...
 */
int main(int argc, char ** argv)
{
	const bfs::path cwd = bfs::initial_path();


	/*
	 * Handle the user input
	 */
	std::string strInputFileName;
	std::string strOutputFileName;

	 /* Parse the command line
	 */
	niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

	CommandLineOptions.GetArgument( O_INPUT_IMAGE,  strInputFileName  );
	CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, strOutputFileName );


	/*
	 * First find out if the given file has the intent code set correctly...
	 */
	bfs::path inPath         ( strInputFileName  );
	bfs::path outPath        ( strOutputFileName );
	bfs::path outIntermedPath( strOutputFileName );

	outIntermedPath.replace_extension( "nii" );

	/* Does input exist? */
	if ( ! bfs::exists( inPath ) )
	{
		std::cerr << "Input " << inPath.string() << " file does not exist!\n";
		return EXIT_FAILURE;
	}

	/* The output should NOT exist (nifti writing will fail otherwise) */
	if ( bfs::exists( outIntermedPath ) )
	{
		std::cerr << "The output image " << outIntermedPath.string() << " already exists!";
		return EXIT_FAILURE;
	}

	/* read nifti image and the data */
	nifti_image * nim = NULL;
	nim = nifti_image_read( strInputFileName.c_str() , true );

	if ( nim == NULL )
	{
		std::cerr << "Failed to read nifti image!";
		return EXIT_FAILURE;
	}

	/*
	 * Check if the sform and qform transformations are the same, otherwise this could
	 * induces real trouble...
	 */
	mat44 * sFormMatrix = (mat44 *) calloc( 1, sizeof( mat44 ) ); //&(nim->sto_xyz);
	mat44 * qFormMatrix = (mat44 *) calloc( 1, sizeof( mat44 ) ); //&(nim->qto_xyz);

	memcpy( sFormMatrix, &( nim->sto_xyz ), sizeof( mat44 ) );
	memcpy( qFormMatrix, &( nim->qto_xyz ), sizeof( mat44 ) );

	bool bQSFromCheckPassed = true;

	for( int iI = 0;  iI < 4;  ++iI )
	{
		for ( int iJ = 0;  iJ < 4;  ++iJ )
		{
			if ( sFormMatrix->m[iI][iJ] !=  qFormMatrix->m[iI][iJ])
			{
				bQSFromCheckPassed = false;
			}
		}
	}


	if ( ! bQSFromCheckPassed )
	{
		std::cerr << "qform and sform are different.";
		return EXIT_FAILURE;
	}

	/*
	 * modify the nifti-imgae
	 */

	/* set the intent code (for itk internal use) */
	nim->intent_code = NIFTI_INTENT_VECTOR;

	/* set the output file name */
	nifti_set_filenames( nim, outIntermedPath.string().c_str(), 1, 1 );

	/* print some debug information
	 * TODO Make it dependend on the detail level... */
	//nifti_image_infodump( nim );

	/* Check if the file is valid... */
	int iValidNim  = nifti_nim_is_valid      ( nim, 1 );
	int iValidDims = nifti_nim_has_valid_dims( nim, 1 );

	if ( iValidNim != 1 || iValidDims != 1 )
	{
		std::cerr << "The created image does not seem to be valid. Exiting";
		return EXIT_FAILURE;
	}

	/* write the image */
	nifti_image_write( nim );
	nifti_image_free ( nim );

	nim = NULL;

	/*
	 * Start the itk part of the processing
	 */
	const unsigned int Dimension = 3;
	typedef float VectorElementType;
	typedef itk::Vector<VectorElementType, Dimension> PixelType;

	typedef itk::Image<PixelType, Dimension> ImageType;
	typedef ImageType::Pointer               ImagePointerType;

	typedef itk::ImageFileReader<ImageType> ReaderType;
	typedef ReaderType::Pointer ReaderPointerType;
	typedef itk::ImageFileWriter<ImageType> WriterType;
	typedef WriterType::Pointer WriterPointerType;

	typedef itk::NiftiImageIO ImageIOType;
	typedef ImageIOType::Pointer ImageIOPointerType;
	typedef ImageIOType::IOPixelType IOPixelType;

	ReaderPointerType reader = ReaderType::New();
	WriterPointerType writer = WriterType::New();

	ImageIOPointerType niftiIO = ImageIOType::New();


	niftiIO->SetPixelType( niftiIO->VECTOR );

	reader->SetImageIO ( niftiIO          );
	reader->SetFileName( outIntermedPath.string() );

	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject e)
	{
		std::cerr << std::string( e.GetDescription() );
		return EXIT_FAILURE;
	}

	/*
	 * Modify the read image via an image adaptor:
	 * extract the modification from the image file...
	 */

	/* Construct the transformation matrix for the itk and nifti part in homogenous coordinates */
	typedef ImageType::DirectionType DirectionType;
	typedef ImageType::SpacingType   SpacingType;
	typedef ImageType::PointType     PointType;

	typedef itkDisplacementVectorCoordinateAdaptionPixelAccessor<Dimension, float> AccessorType;
	typedef AccessorType::HomogenousMatrixType                                     HomogenousMatrixType;
	typedef itk::ImageAdaptor< ImageType, AccessorType >                           AdaptorType;
	typedef AdaptorType::Pointer                                                   AdaptorPointerType;

	ImagePointerType inImage   = reader->GetOutput();
	DirectionType    direction = inImage->GetDirection();
	SpacingType      spacing   = inImage->GetSpacing();
	PointType        origin    = inImage->GetOrigin();

	/*
	 * Construct just like the itkImageBase.h does
	 */
	DirectionType spacingMat;
	for (unsigned int uiI = 0;  uiI < Dimension;  ++uiI)
	{
		spacingMat[uiI][uiI] = spacing[ uiI ];
	}

	DirectionType scaledDirMat = direction * spacingMat;

	HomogenousMatrixType itkPartMatrix;
	itkPartMatrix[ Dimension ][ Dimension ] = 1.0;

	for ( unsigned int i = 0;  i < Dimension;  ++i )
	{
		for ( unsigned int j = 0;   j < Dimension;  ++j )
		{
			itkPartMatrix[i][j] = scaledDirMat[i][j];
		}

		itkPartMatrix[ i ][ Dimension ] = origin[ i ];
	}


	/*
	 * use the sform (as here we can be sure it is the same as the qform)
	 */
	HomogenousMatrixType niftiPartMatrix;
	niftiPartMatrix[ Dimension ][ Dimension ] = 1;

	for ( unsigned int i = 0;  i < Dimension + 1;  ++i )
	{
		for (unsigned int j = 0;  j < Dimension + 1; ++j )
			niftiPartMatrix[ i ][ j ] = sFormMatrix->m[ i ][ j ];
	}

	HomogenousMatrixType conversionMatrix = itkPartMatrix * niftiPartMatrix.GetInverse();

	
	std::cout << "Evaluated homogenous conversion matrix:" << std::endl << conversionMatrix << std::endl;

	AdaptorPointerType adaptor = AdaptorType::New();

	AccessorType accessor;
	accessor.SetHomogenousMatrix( conversionMatrix );

	adaptor->SetPixelAccessor( accessor );
	adaptor->SetImage( reader->GetOutput() );

	typedef itk::MultiplyByConstantImageFilter< AdaptorType,
			                                    VectorElementType ,
			                                    ImageType >          MultiplyFilerType;
	typedef MultiplyFilerType::Pointer                               MultiplyFilerPointerType;

	MultiplyFilerPointerType doNothingFilter = MultiplyFilerType::New();
	doNothingFilter->SetConstant( 1.0f );

	doNothingFilter->SetInput( adaptor );

	writer->SetFileName( outPath.string() );
	
	writer->SetInput   ( doNothingFilter->GetOutput() );
	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject e)
	{
		std::cerr << std::string( e.GetDescription() );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
