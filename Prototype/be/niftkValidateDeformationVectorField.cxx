/*
 * Author:  Bjoern Eiben
 * Purpose: Validate the deformation vector field which was generated with niftkConvertNiftiVectorImage.cxx
 * Date:    14th Jan 2010
 *
 */


/*
 * includes
 */
#include <stdlib.h>

/* itk */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVector.h"
#include "itkWarpImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageAdaptor.h"


/* niftk */
#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include "DisplacementVectorCoordinateAdaptionPixelAccessor.h"



/*
 * Command line parser options
 */
struct niftk::CommandLineArgumentDescription clArgList[] =
{
	{ OPT_STRING | OPT_REQ, "i",           "filename",    "Input image (short pixel type assumed)."                },
	{ OPT_STRING | OPT_REQ, "o",           "filename",    "Output image (short)."                                  },
	{ OPT_STRING | OPT_REQ, "def",         "filename",    "Deformation field (float 3D vector)."                   },
	{ OPT_STRING | OPT_REQ, "interpolate", "nn|lin|bspl", "For \"nn\" nearest neighbour, \"lin\" for linear or \"bspl\" for bspline interpolation" },
	{ OPT_SWITCH,           "vox",         0,             "Deformation given in voxel spacing. Scale the transformation by the voxel spacing"},
    { OPT_DONE, NULL, NULL,                                      "Warps the input image by the deformation field."        }
};



enum
{
    O_INPUT = 0,
    O_OUTPUT,
    O_FIELD,
	O_INTERPOLATION,
	O_VOXELSPACING
};




enum
{
	NEAREST_NEIGHBOUR = 0,
	LINEAR,
	BSPLINE
} eInterpolation;


int main(int argc, char ** argv)
{
	const unsigned int Dimension = 3;

	typedef float                                        VectorComponentType;
	typedef short                                        InputPixelType;
	typedef short                                        OutputPixelType;
	typedef itk::Vector<VectorComponentType, Dimension>  VectorType;
	typedef itk::Image<VectorType, Dimension>            DeformationImageType;
	typedef DeformationImageType::Pointer                DeformationImagePointerType;
	typedef itk::Image<InputPixelType, Dimension>        InputImageType;
	typedef InputImageType::Pointer                      InputImagePointerType;

	typedef itk::Image<OutputPixelType, Dimension>       OutputImageType;
	typedef OutputImageType::Pointer                     OutputImagePointerType;

	/*
	 * Get the user-input
	 */

	std::string strInputImageName;
	std::string strOutputImageName;
	std::string strVectorFieldImageName;
	std::string strGivenInterpolation;
	bool        bVoxelSpacing = false;

    niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true  );
    CommandLineOptions.GetArgument( O_INPUT,         strInputImageName       );
    CommandLineOptions.GetArgument( O_OUTPUT,        strOutputImageName      );
    CommandLineOptions.GetArgument( O_FIELD,         strVectorFieldImageName );
	CommandLineOptions.GetArgument( O_INTERPOLATION, strGivenInterpolation   );
	CommandLineOptions.GetArgument( O_VOXELSPACING,  bVoxelSpacing           );

	/*
	 * Which Interpolation to use...
	 */
	if ( strGivenInterpolation.compare("nn") == 0 ) 
	{
		eInterpolation = NEAREST_NEIGHBOUR;
		std::cout << "Using nearest neighbor interpolation." << std::endl;

	}
	else if (strGivenInterpolation.compare("lin") == 0)
	{
		eInterpolation = LINEAR;
		std::cout << "Using nearest neighbor interpolation." << std::endl;
	}
	else if (strGivenInterpolation.compare("bspl") == 0)
	{
		eInterpolation = BSPLINE;
		std::cout << "Using b-spline interpolation." << std::endl;
	}else
	{
		eInterpolation = BSPLINE;
		std::cout << "Using default b-spline interpolation." << std::endl;
	}

	/*
	 * Read the images
	 */

	/* input image */
    typedef itk::ImageFileReader< InputImageType >   InputImageReaderType;
    typedef InputImageReaderType::Pointer            InputImageReaderPointerType;

    InputImageReaderPointerType inputReader = InputImageReaderType::New();
    inputReader->SetFileName( strInputImageName );


	/* deformation field */
    typedef itk::ImageFileReader< DeformationImageType >  DeformationImageReaderType;
    typedef DeformationImageReaderType::Pointer           DeformationImageReaderPointerType;

    DeformationImageReaderPointerType deformReader = DeformationImageReaderType::New();
    deformReader->SetFileName( strVectorFieldImageName );

    try
    {
		deformReader->Update();
	}
    catch (itk::ExceptionObject e)
    {
    	std::cerr << "Exception caugth!" << std::endl
    	          << e << std::endl;
	}


	/*
	 * Adapt the vector field from voxel to real world spacing if required
	 */
	typedef DisplacementVectorCoordinateAdaptionPixelAccessor<Dimension, float> AccessorType;
	typedef AccessorType::HomogenousMatrixType                                  HomogenousMatrixType;
	typedef itk::ImageAdaptor< DeformationImageType, AccessorType >             AdaptorType;
	typedef AdaptorType::Pointer                                                AdaptorPointerType;

	DeformationImageType::SpacingType spacing = deformReader->GetOutput()->GetSpacing();

	// Construct the scaling matrix
	HomogenousMatrixType scalingMatrix;

	if ( bVoxelSpacing ) 
	{
		for (unsigned int uiI = 0;  uiI <= Dimension;  ++uiI )
			scalingMatrix[uiI][uiI] = spacing[uiI];
	}
	else
	{
		for (unsigned int uiI = 0;  uiI <= Dimension;  ++uiI )
			scalingMatrix[uiI][uiI] = 1;

	}
	scalingMatrix[Dimension][Dimension] = 1;

	std::cout << "Using scaling matrix for vector field:" << std::endl << scalingMatrix << std::endl;

	AdaptorPointerType adaptor = AdaptorType::New();

	AccessorType accessor;
	accessor.SetHomogenousMatrix( scalingMatrix );

	adaptor->SetPixelAccessor( accessor );
	adaptor->SetImage( deformReader->GetOutput() );



	/*
	 * Warp the image
	 */
	typedef itk::WarpImageFilter< InputImageType,
    		                      OutputImageType,
    		                      AdaptorType >      WarpFilterType;
    typedef WarpFilterType::Pointer                  WarpFilterPointerType;

    /* create the interpolators */
    typedef itk::BSplineInterpolateImageFunction< InputImageType, double >  BSplineInterpolatorType;
	typedef BSplineInterpolatorType::Pointer                                BSplineInterpolatorPointerType;
    BSplineInterpolatorPointerType bSplineInterpolator = BSplineInterpolatorType::New();

	typedef itk::LinearInterpolateImageFunction< InputImageType, double >  LinearInterpolatorType;
	typedef LinearInterpolatorType::Pointer                                LinearInterpolatorPointerType;
	LinearInterpolatorPointerType linearInterpolator = LinearInterpolatorType::New();

	typedef itk::NearestNeighborInterpolateImageFunction< InputImageType, double >  NearestNeighborInterpolatorType;
	typedef NearestNeighborInterpolatorType::Pointer                                NearestNeighborInterpolatorPointerType;
	NearestNeighborInterpolatorPointerType nnInterpolator = NearestNeighborInterpolatorType::New();


    bSplineInterpolator->SetSplineOrder( 3u );

    WarpFilterPointerType warpFilter = WarpFilterType::New();
    
	// Set the correct interpolation type
	if ( eInterpolation == NEAREST_NEIGHBOUR )
	{
		warpFilter->SetInterpolator( linearInterpolator );
	}
	else if (eInterpolation == LINEAR )
	{
		warpFilter->SetInterpolator( linearInterpolator );
	}else
	{
		warpFilter->SetInterpolator( bSplineInterpolator );
	}

	// feed the warpFilter with the adapted deformation field 
	warpFilter->SetInput           ( inputReader->GetOutput()                  );
	warpFilter->SetDeformationField( adaptor                                   );
    warpFilter->SetOutputOrigin    ( deformReader->GetOutput()->GetOrigin()    );
    warpFilter->SetOutputSpacing   ( deformReader->GetOutput()->GetSpacing()   );
    warpFilter->SetOutputDirection ( deformReader->GetOutput()->GetDirection() );


	/*
	 * Write the result
	 */
    typedef itk::ImageFileWriter< OutputImageType >  ImageWriterType;
    typedef ImageWriterType::Pointer                 ImageWriterPointerType;

    ImageWriterPointerType imageWriter = ImageWriterType::New();
    imageWriter->SetInput   ( warpFilter->GetOutput() );
    imageWriter->SetFileName( strOutputImageName      );


    /*
     * Update the pipeline...
     */
    try
    {
    	imageWriter->Update();
	}
    catch (itk::ExceptionObject e)
	{
    	std::cout << "Something went terribly wrong..." << e << std::endl;
    	return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
