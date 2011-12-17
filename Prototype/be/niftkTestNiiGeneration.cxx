


/*
 * Includes
 */


/* Toolkit */
#include "CommandLineParser.h"

/* ITK */
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkExceptionObject.h"
#include "itkSpatialOrientationAdapter.h"





int main(int argc, char ** argv)
{
	const unsigned int Dimension = 3;
	typedef float PixelType;
	typedef itk::Image< PixelType, Dimension > ImageType;
	typedef ImageType::Pointer                 ImagePointerType;
	typedef ImageType::IndexType               ImageIndexType;
	typedef ImageType::SizeType                ImageSizeType;
	typedef ImageType::RegionType              ImageRegionType;
	typedef ImageType::SpacingType             ImageSpacingType;
	typedef ImageType::DirectionType           ImageDirectionType;

	typedef itk::ImageFileWriter< ImageType >  ImageFileWriterType;
	typedef ImageFileWriterType::Pointer       ImageFileWriterPointerType;

	/* Create image */
	ImagePointerType image = ImageType::New();


	/* Set size */
	ImageSizeType size;
	size.Fill( 128 );

	/* Set index */
	ImageIndexType index;
	index.Fill( 0 );

	ImageRegionType region;
	region.SetSize( size );
	region.SetIndex( index );

	image->SetRegions( region );

	/* Spacing */
	ImageSpacingType spacing;
	spacing[0] = 1.0;
	spacing[1] = 1.5;
	spacing[2] = 2.0;

	image->SetSpacing( spacing );

	/* Direction */
	ImageDirectionType direction;
	direction.SetIdentity();

	image->SetDirection( direction );

	image->Allocate();

	ImageFileWriterPointerType writer = ImageFileWriterType::New();
	writer->SetFileName("test.nii");
	writer->SetInput( image );


	ImageIndexType idx;

	idx.Fill(0);

	image->SetPixel( idx, 1.0f );

	writer->SetDebug( true );

	typedef itk::SpatialOrientationAdapter       SpatialAdapterType;
	typedef SpatialAdapterType::DirectionType    DirectionType;
	typedef SpatialAdapterType::OrientationType  OrientationType;

	SpatialAdapterType spatApt;
	//SpatialAdapterType().ToDirectionCosines();


	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject e)
	{
		std::cout << e.GetDescription() <<std::endl;
	}

	return EXIT_SUCCESS;
}
