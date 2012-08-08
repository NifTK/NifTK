


/*
 * Includes
 */


/* nifTK */
#include "CommandLineParser.h"

/* ITK */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExceptionObject.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkSobelEdgeDetectionImageFilter.h"
#include "itkBSplineDownsampleImageFilter.h"

/* VTK */
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkDelaunay3D.h"
#include "vtkUnstructuredGrid.h"
#include "vtkSmartPointer.h"
#include "vtkXMLPolyDataWriter.h"
#include "vtkPolyDataWriter.h"
#include "vtkXMLUnstructuredGridWriter.h"
#include "vtkCellArray.h"
#include "vtkDataSetSurfaceFilter.h"


/*
 * Command line enum
 */
enum
{
	O_IMG_IN = 0,
	O_VTK_POLYDATA_OUT,
};


niftk::CommandLineArgumentDescription clArgList[] =
{
	{ OPT_STRING | OPT_REQ, "i", "filename", "Image file which is thresholded and then used to generate the convex hull." },
	{ OPT_STRING | OPT_REQ, "o", "filename", "Output vtk-PolyData file with the convex hull." },
	{ OPT_DONE, NULL, NULL,                  "Calculate the convex hull around a thresholded image.\n\n"        }
};



int main(int argc, char ** argv)
{	
	
	const unsigned int Dimension = 3;
	typedef short PixelType;
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
	std::string strOutPolyDataName;
	PixelType   ptThreshold = 128; // TODO: allow to be set as command line parameter

	// Set up the command line
	niftk::CommandLineParser CommandLineOptions( argc, argv, clArgList, true );

	CommandLineOptions.GetArgument( O_IMG_IN,           strInImageName      );
	CommandLineOptions.GetArgument( O_VTK_POLYDATA_OUT, strOutPolyDataName  );



	// Read the imgae
	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	ImageReaderType::Pointer reader = ImageReaderType::New();
	reader->SetFileName( strInImageName.c_str() );
	
	try
	{
		reader->Update();
	}
	catch( itk::ExceptionObject e )
	{
		std::cout << "Exception caught while reading the image: " << std::endl << e ;
	}
	
	
	// Downsample the image as the number of points is extensive
	std::cout << "Downsampling the image..." << std::endl;
	typedef itk::BSplineDownsampleImageFilter<ImageType, ImageType> DownSampleFilterType;
	DownSampleFilterType::Pointer downsampler = DownSampleFilterType::New();
	
	downsampler->SetInput( reader->GetOutput() );
	downsampler->Update();

	// Only consider the image edges:
	std::cout << "Edge filterung the image..." << std::endl;
	typedef itk::SobelEdgeDetectionImageFilter<ImageType, ImageType> SobelFilterType;
	SobelFilterType::Pointer sobelFilter = SobelFilterType::New();

	sobelFilter->SetInput( downsampler->GetOutput() );
	sobelFilter ->Update();
	
	ImagePointerType img = sobelFilter->GetOutput();
	
	typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
	IteratorType it( sobelFilter->GetOutput(), sobelFilter->GetOutput()->GetLargestPossibleRegion() );
	

	// Prepare vtk points
	vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();
	
	// Iterate through the image and add the points above a certain threshold level to the list of vtk points. 
	std::cout << "Adding vtk points..." << std::endl;

	for ( it.GoToBegin();  !it.IsAtEnd()  ; it++ )
	{
		if ( it.Get() > ptThreshold ) 
		{
			ImagePointType p;
			img->TransformIndexToPhysicalPoint( it.GetIndex(), p );
			points->InsertNextPoint( p[0], p[1], p[2]);
		}
	}
	
	vtkSmartPointer< vtkPolyData > polydata = vtkSmartPointer< vtkPolyData >::New();
	polydata->SetPoints( points );

	vtkSmartPointer< vtkDelaunay3D > delaunay = vtkSmartPointer< vtkDelaunay3D >::New();
	delaunay->SetInput( polydata );
	delaunay->Update();
	
	vtkSmartPointer<vtkDataSetSurfaceFilter> surfaceFilter = vtkSmartPointer<vtkDataSetSurfaceFilter>::New();
	surfaceFilter->SetInputConnection(delaunay->GetOutputPort());
	surfaceFilter->Update();  

	vtkSmartPointer<vtkPolyDataWriter> outputWriter = vtkSmartPointer<vtkPolyDataWriter>::New();
	outputWriter->SetFileName( strOutPolyDataName.c_str() );
	outputWriter->SetInput(surfaceFilter->GetOutput());
	outputWriter->Write();

	return EXIT_SUCCESS;
}
