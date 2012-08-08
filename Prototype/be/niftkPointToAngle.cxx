


/*
 * Includes
 */


/* nifTK */
#include "CommandLineParser.h"

/* ITK */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageFileWriter.h"
#include "itkExceptionObject.h"

#include "itkLocationToAngleImageFilter.h"



/*
 * Command line enum
 */
enum
{
	O_IMG_IN = 0,
	O_VTK_POLYDATA_OUT,
	O_PBODY1,
    O_PBODY2,
	O_PUP,
};


niftk::CommandLineArgumentDescription clArgList[] =
{
	{ OPT_STRING | OPT_REQ, "i",   "filename",    "Input image file."                },
	{ OPT_STRING | OPT_REQ, "o",   "filename",    "Output image file."               },
	{ OPT_DOUBLEx3,         "pB1", "coordinates", "First point on the body axis."    },
	{ OPT_DOUBLEx3,         "pB2", "coordinates", "Second point on the body axis."   },
	{ OPT_DOUBLEx3,         "pUp", "coordinates", "Point defining the up direction." },
	{ OPT_DONE,             NULL,  NULL,          "ToDo.....\n\n"                    }
};



int main(int argc, char ** argv)
{	
	
	const unsigned int Dimension = 3;
	
	typedef float                              PixelType;
	typedef itk::Image< PixelType, Dimension > ImageType;
	typedef ImageType::Pointer                 ImagePointerType;
	typedef ImageType::IndexType               ImageIndexType;
	typedef ImageType::SizeType                ImageSizeType;
	typedef ImageType::RegionType              ImageRegionType;
	typedef ImageType::SpacingType             ImageSpacingType;
	typedef ImageType::DirectionType           ImageDirectionType;
	typedef ImageType::PointType               ImagePointType;
	typedef itk::ImageFileWriter< ImageType >  ImageFileWriterType;
	typedef ImageFileWriterType::Pointer       ImageFileWriterPointerType;

	// parameters...
	std::string strInImageName;
	std::string strOutImageName;
	

	// Set up the command line
	niftk::CommandLineParser CommandLineOptions( argc, argv, clArgList, true );


	double* pdPointBodyAxis1;
	double* pdPointBodyAxis2;
	double* pdUp;
	
	CommandLineOptions.GetArgument( O_IMG_IN,           strInImageName  );
	CommandLineOptions.GetArgument( O_VTK_POLYDATA_OUT, strOutImageName );
	CommandLineOptions.GetArgument( O_PBODY1, pdPointBodyAxis1          );
	CommandLineOptions.GetArgument( O_PBODY2, pdPointBodyAxis2          );
	CommandLineOptions.GetArgument( O_PUP,    pdUp                      );

	// Read the imgae
	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	ImageReaderType::Pointer reader = ImageReaderType::New();
	reader->SetFileName( strInImageName );
	
	try
	{
		reader->Update();
	}
	catch( itk::ExceptionObject &e )
	{
		std::cout << "Exception caught while reading the image: " << std::endl << e ;
	}
	
	typedef itk::LocationToAngleImageFilter< PixelType, Dimension > AngleFilterType;
	typedef AngleFilterType::PointType                              PointType;

	AngleFilterType::Pointer angleFilter = AngleFilterType::New();

	// Copy input points from double array into itkPoints 
	PointType ptBodyAxis1, ptBodyAxis2, ptUp;
	
	for ( unsigned int uiI=  0;  uiI < Dimension; uiI++ )
	{
		ptBodyAxis1[ uiI ] = pdPointBodyAxis1[ uiI ];
		ptBodyAxis2[ uiI ] = pdPointBodyAxis2[ uiI ];
		ptUp       [ uiI ] = pdUp            [ uiI ];
	}

	// Set the points in the filter
	angleFilter->SetBodyAxisPoint1( ptBodyAxis1 );
	angleFilter->SetBodyAxisPoint2( ptBodyAxis2 );
	angleFilter->SetUpPoint       ( ptUp        );
	
	angleFilter->SetInput( reader->GetOutput() );
	
	try
	{
		angleFilter->Update();
	}
	catch( itk::ExceptionObject &e )
	{
		std::cout << "Exception caught while reading the image: " << std::endl << e ;
	}
	

	typedef itk::ImageFileWriter< ImageType > WriterType;
	WriterType::Pointer writer = WriterType::New();
	
	writer->SetFileName( strOutImageName );
	writer->SetInput( angleFilter->GetOutput() );

	writer->Update();

	return EXIT_SUCCESS;
}
