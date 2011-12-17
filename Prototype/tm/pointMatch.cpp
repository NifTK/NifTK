/*=========================================================================


=========================================================================*/


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkNewAffineTransform.h"
#include "itkRayCastInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkIndex.h"
#include "itkContinuousIndex.h"

#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

bool detailedOutput;

double pointLineDistance(double annot2D[3], double xRaySource[3], double annot3D[3]);
//void linePlaneIntersection(double p1[3], double p2[3], double zValue, double* intersection);
void linePlaneIntersection(double p1[3], double p2[3], double normal[3], double
planePoint[3], double* intersection);
template <typename ImageTypePointType, 
          typename TransformTypePointer, 
	  typename InterpolatorTypeInputPointType,
	  typename InterpolatorTypeOutputPointType,
	  typename OutputVnlVectorType> 
void forwardProjection(	ImageTypePointType physPoint3DGT, 
			TransformTypePointer transformIni, 
			InterpolatorTypeInputPointType focalPoint,
			double* origin,
			double* intersection);
template <typename ImageTypePointType, 
          typename TransformTypePointer,
	  typename InterpolatorTypeInputPointType,
	  typename InterpolatorTypeOutputPointType> 
double reprojectionError(	ImageTypePointType physPoint, 
                        ImageTypePointType physPoint3D, 
			TransformTypePointer transform, 
			TransformTypePointer transformIni,
			InterpolatorTypeInputPointType focalPoint);
 
int main( int argc, char * argv[] )
{
  if( argc < 10 ) 
  { 
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " 3DundefVolume 3DGroundTruthVolume 2DdrrRegImage" 
                         << " paramFileFromRegistration pointsGT pointsUndef" 
			 << " areaForRegistration NumberOfLandmarks detailedOutput?";
    std::cerr <<  std::endl;
    return EXIT_FAILURE;
  }

  float sid = 660.;  

  // input and output decl  
  const int dimension = 3;
  typedef float InputPixelType;
  typedef float OutputPixelType;
  
  typedef itk::Image< InputPixelType,  dimension >   InputImageType;
  typedef itk::Image< OutputPixelType,  dimension >   OutputImageType;
  typedef itk::ImageFileReader< InputImageType >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  typedef itk::ResampleImageFilter< InputImageType, InputImageType > FilterType;

  FilterType::Pointer filter = FilterType::New();
  ReaderType::Pointer reader = ReaderType::New();
  ReaderType::Pointer readerGT = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  InputImageType::Pointer inputImage;
  InputImageType::Pointer inputImageGT;
  OutputImageType::Pointer outputImage;
  
  reader->SetFileName( argv[1] );
  readerGT->SetFileName( argv[2] );
  writer->SetFileName( argv[3] );

  detailedOutput = atoi( argv[9] );

  typedef itk::NewAffineTransform< double, 3 >  TransformType;
  typedef TransformType::ParametersType ParametersType;

  TransformType::Pointer transform = TransformType::New();
  ParametersType parameters (12); 
 
  // ********************************************
  // Reading the parameters and points from files
  // ********************************************
  int i = 0; // counter to know the position in the file
  float x; // variable for reading the file params
 
  // read the parameters from the registration
  ifstream inFile;
  inFile.open( argv[4] );
  
  if (!inFile)   
  {
    std::cout << "Unable to open file "<<  argv[4] << std::endl;
    exit(1); // terminate with error
  }
  while (inFile >> x) 
  {
    if ( i < 12 )
    {
      parameters[i] = x;
      i++;
    }
    else
      break;
  }

  inFile.close();

  if ( detailedOutput )
    std::cout << "The parameters for projection are: " << parameters << std::endl;

  // read the points/landmarks
  ifstream inFileGT;
  inFileGT.open( argv[5] );
  int j = 0; // counter to know the position in the files
  float y; // variable for reading the file params

  int N = atoi( argv[8] ); //number of points 
  float ** pointsGT;
  pointsGT = new float * [N];
  for(int i=0; i<N; i++)
    pointsGT[i] = new float[3];

  if (!inFileGT)  
  {
    std::cout << "Unable to open file "<<  argv[7] << std::endl;
    exit(1); // terminate with error
  }
  while (inFileGT >> y)
  {
    if ( j < 3*N )
    {
      pointsGT[j/3][j%3] = y;
      j++;
    }
    else
	break;
  }

  inFileGT.close();
  
  for (j=0; j<N; j++)
  {
    pointsGT[j][0] = pointsGT[j][0];
    pointsGT[j][1] = pointsGT[j][1];
    pointsGT[j][2] = pointsGT[j][2];    
  }
  
  if ( detailedOutput )
  {
    for (j=0; j<N; j++)
    {
      std::cout << "\nGT Point " << j << " -> \t" ;
      for (i=0; i<3; i++)
      {
        std::cout << pointsGT[j][i] << "\t"; ;
      }
    }
    std::cout << "\n";
  }

  // read the points/landmarks
  ifstream inFileUndef;
  inFileUndef.open( argv[6] );
  j = 0; // counter to know the position in the files
  float z; // variable for reading the file params

  float ** pointsUndef;
  pointsUndef = new float * [N];
  for(i=0; i<N; i++)
    pointsUndef[i] = new float[3];
    
  if (!inFileUndef)   
  {
    std::cout << "Unable to open file " <<  argv[6] << std::endl;
    exit(1); // terminate with error
  }
  while (inFileUndef >> z) 
  {
    if ( j < 3*N )
    {
      pointsUndef[j/3][j%3] = z;
      j++;
    }
    else
	break;
  }

  inFileUndef.close();
  
  for (j=0; j<N; j++)
  {
    pointsUndef[j][0] = pointsUndef[j][0];
    pointsUndef[j][1] = pointsUndef[j][1];
    pointsUndef[j][2] = pointsUndef[j][2];    
  }
  
  if ( detailedOutput )
  {
    for (j=0; j<N; j++)
    {
      std::cout << "\nUndef Point " << j << " -> \t" ;
      for (i=0; i<3; i++)
      {
        std::cout << pointsUndef[j][i] << "\t"; ;
      }
    }
    std::cout << "\n";
  }

  // read the area used for Registration
  ifstream inFileArea;
  inFileArea.open( argv[7] );
  j = 0; // counter to know the position in the files
  int w; // variable for reading the file params

  int area[4] = {0, 0, 0, 0};
    
  if (!inFileArea)   
  {
    std::cout << "Unable to open file " <<  argv[7] << std::endl;
    exit(1); // terminate with error
  }
  while (inFileArea >> w) 
  {
    if ( j < 4 )
    {
      area[j] = w;
      j++;
    }
    else
	break;
  }

  inFileArea.close();

  if ( detailedOutput )
  {
    std::cout << "\nArea used for registration " << " -> \n" 
              << "Xmin\t Xmax\t Ymin\t Ymax" << std::endl;
    for (j=0; j<4; j++)
      std::cout << area[j] << "\t"; ;
    std::cout << "\n\n";
  }

  // ********************************************

  reader->Update();
  readerGT->Update();

  inputImage = reader->GetOutput();
  inputImage->DisconnectPipeline();

  inputImageGT = readerGT->GetOutput();
  inputImageGT->DisconnectPipeline();

  double imOrigin[ dimension ];

  const itk::Vector<double, 3> imRes = inputImage->GetSpacing();

  typedef InputImageType::RegionType     InputImageRegionType;
  typedef InputImageRegionType::SizeType InputImageSizeType;

  const itk::Size<3>   imSize   = inputImage->GetBufferedRegion().GetSize();

  // ATTENTION!!! CHANGE THESE IF THE SIZE OF THE VOLUME IS ODD (put -1)
  imOrigin[0] = imRes[0]*((double) imSize[0] )/2.; 
  imOrigin[1] = imRes[1]*((double) imSize[1] )/2.; 
  imOrigin[2] = imRes[2]*((double) imSize[2] )/2.;

  transform->SetCenter( imOrigin );
  transform->SetParameters( parameters );

  if ( detailedOutput )
  {
    std::cout << "The transform is: " << transform->GetMatrix() << std::endl;
    std::cout << "The inverse transform is: " << transform->GetInverseMatrix() << std::endl;
    std::cout << "\nThe centre of the volume is: " << imOrigin[0] << " " << imOrigin[1] << " " << imOrigin[2] << std::endl;
  }

  typedef itk::RayCastInterpolateImageFunction<InputImageType,double> InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetTransform(transform);

  InterpolatorType::InputPointType focalpoint;

  focalpoint[0]= imOrigin[0]; 
  focalpoint[1]= imOrigin[1]; 
  focalpoint[2]= imOrigin[2] - (sid-80.); 

  interpolator->SetFocalPoint(focalpoint);

  if ( detailedOutput )
    std::cout << "The focal point is: " << focalpoint[0] << " " << focalpoint[1] << " " << focalpoint[2] << std::endl;

  filter->SetInterpolator( interpolator );
  filter->SetTransform( transform );

  InputImageType::SizeType size;

  size[0] = 501;  // number of pixels along X of the 2D DRR image 
  size[1] = 501;  // number of pixels along Y of the 2D DRR image 
  size[2] = 1;   // only one slice

  filter->SetSize( size );

  double spacing[ dimension ];

  spacing[0] = 1.0;  // pixel spacing along X of the 2D DRR image [mm]
  spacing[1] = 1.0;  // pixel spacing along Y of the 2D DRR image [mm]
  spacing[2] = 1.0; // slice thickness of the 2D DRR image [mm]

  filter->SetOutputSpacing( spacing );
  filter->SetInput( inputImage );

  double origin[ dimension ];

  origin[0] = imOrigin[0] - ((double) 501 - 1.)/2.; 
  origin[1] = imOrigin[1] - ((double) 501 - 1.)/2.; 
  origin[2] = imOrigin[2] + 80.;

  filter->SetOutputOrigin( origin );
  
  if ( detailedOutput )
    std::cout << "The origin of the DRR is: " << origin[0] << " " << origin[1] << " " << origin[2] << std::endl;

  filter->Update(); 
  outputImage = filter->GetOutput();
 
  // 
  std::cout << "The centre of the volume is: " << imOrigin[0] << "  " << imOrigin[1] << " " << imOrigin[2] << std::endl;
  
  // inital transform, to position the volume etc in the scene 
  TransformType::Pointer transformIni = TransformType::New();
  ParametersType parametersIni (12); 

  parametersIni[0] = 0;
  parametersIni[1] = 1.5708;//0;
  parametersIni[2] = 0;
  parametersIni[3] = 1;
  parametersIni[4] = 1;
  parametersIni[5] = 1;
  parametersIni[6] = 0;
  parametersIni[7] = 0;
  parametersIni[8] = 0;
  parametersIni[9] = -imOrigin[0];//-30;
  parametersIni[10] = -imOrigin[1];//-30;
  parametersIni[11] = -imOrigin[2];//-30;

  transformIni->SetCenter( imOrigin );
  transformIni->SetParameters( parametersIni );

  if ( detailedOutput )
  {
    std::cout << "The initial transformIni params are: " << transformIni->GetParameters() << std::endl;
    std::cout << "The inverse initial transformIni is: " <<
    transformIni->GetInverseMatrix() << std::endl;
    std::cout << "\n*************************************\n" << std::endl;
  }
  // *********************************************************
  // Run through all the points and calculate the reproj error
  // *********************************************************
  double totalReprojError = 0.0;
  double totalReprojErrorInside = 0.0;  
  OutputImageType::PointType physPoint3DGT;
  OutputImageType::PointType physPoint2DGT;
  OutputImageType::PointType physPoint3D;
  double * reprojError;
  reprojError = new double[N];
  bool * insideArea;
  insideArea = new bool[N];
  int numberOfPointsInside = 0;
  double intersection[]={0.0, 0.0, 0.0};

  for ( i=0; i<N; i++)
  {
    if ( detailedOutput )
      std::cout << "\n-------  Processing point " << i << "  -------\n" << std::endl;
    
    physPoint3DGT[0] = pointsGT[i][0];
    physPoint3DGT[1] = pointsGT[i][1];
    physPoint3DGT[2] = pointsGT[i][2];  
  
    forwardProjection<OutputImageType::PointType, 
  		      TransformType::Pointer,
		      InterpolatorType::InputPointType,
		      InterpolatorType::OutputPointType,
		      TransformType::OutputVnlVectorType>(physPoint3DGT,
		      transformIni, focalpoint, origin, intersection);

    OutputImageType::PointType intersectionPoint;
    intersectionPoint[0] = intersection[0];
    intersectionPoint[1] = intersection[1];
    intersectionPoint[2] = intersection[2]; 

    OutputImageType::IndexType intersectionIndex;
    
    outputImage->TransformPhysicalPointToIndex(intersectionPoint, intersectionIndex);
    if ( detailedOutput )
      std::cout << "The intersection Index (image coords) is: " << intersectionIndex << std::endl;

    //check if it is inside the registration area
    if ( ( intersectionIndex[0]>area[0] )&&( intersectionIndex[0]<area[1] )&&
         ( intersectionIndex[1]>area[2] )&&( intersectionIndex[1]<area[3] ) )
    {
      insideArea[i] = true;
      numberOfPointsInside += 1;
    }
    else
      insideArea[i] = false;

    itk::ContinuousIndex< double, 3> intersectionContIndex;

    outputImage->TransformPhysicalPointToContinuousIndex(intersectionPoint, intersectionContIndex);

    if ( detailedOutput )
      std::cout << "The intersection Continuous Index (image coords) is: " << intersectionContIndex << std::endl;

    physPoint2DGT[0] = intersectionContIndex[0]+origin[0];
    physPoint2DGT[1] = intersectionContIndex[1]+origin[1];
    physPoint2DGT[2] = origin[2];  
    
    if ( detailedOutput )
      std::cout << "The physical point 2DGT is: " << physPoint2DGT << std::endl;

    physPoint3D[0] = pointsUndef[i][0];
    physPoint3D[1] = pointsUndef[i][1];
    physPoint3D[2] = pointsUndef[i][2];  
    
    if ( detailedOutput )
      std::cout << "The physical point 3D is: " << physPoint3D << std::endl;

  
    reprojError[i] = 0.0;
    reprojError[i] = reprojectionError<OutputImageType::PointType,
  				  TransformType::Pointer,
				  InterpolatorType::InputPointType,
	                          InterpolatorType::OutputPointType> 
               (physPoint2DGT, physPoint3D, transform, transformIni, focalpoint);
    totalReprojError += reprojError[i]; 
  }

  totalReprojError = totalReprojError/N;
  std::cout << "\n--------------\n" << std::endl;
  std::cout << "The mean reprojection error is: " << totalReprojError << std::endl; 
  std::cout << "\n--------------\n" << std::endl;


  if( detailedOutput )
      std::cout << "The number of points inside the reg.area is: " << 
                    numberOfPointsInside << "/" << N << std::endl;

  std::cout << "Points inside the area -> " ;
  // calculate the mean reproj.error without the points that fall outside the registration area
  for (i=0; i<N; i++)
  {
    if (insideArea[i])
    {
      totalReprojErrorInside += reprojError[i];
      std::cout << i << "  " ;
    }
  }
  totalReprojErrorInside = totalReprojErrorInside/numberOfPointsInside;
 
  std::cout << "\n--------------\n" << std::endl;
  std::cout << "The mean reprojection error (Inside the registration area) is: " 
            << totalReprojErrorInside << std::endl; 
  std::cout << "\n--------------\n" << std::endl;

  float stdInside = 0;
  float maxInside = -1;

  // Compute the std and the max
  for (i=0; i<N; i++)
  {
    if (insideArea[i])
    {
      if (reprojError[i]>maxInside)
        maxInside = reprojError[i];
      stdInside += pow(reprojError[i] - totalReprojErrorInside,2);
    }
  }  
  stdInside /= numberOfPointsInside;
  stdInside = sqrt(stdInside);

  std::cout << "\n--------------\n" << std::endl;
  std::cout << "The std of reprojection error (Inside the registration area) is: " 
            << stdInside << std::endl;
  std::cout << "The max reprojection error (Inside the registration area) is: " 
            << maxInside << std::endl; 
  std::cout << "\n--------------\n" << std::endl;
  

  double myOrigin[] = {0, 0, 0}; // used to reset the origin of the DRR
  outputImage->SetOrigin(myOrigin);

  writer->SetInput( outputImage );

  try 
  { 
    std::cout << "Writing output image..." << std::endl;
    writer->Update();
  } 
  catch( itk::ExceptionObject & err ) 
  {      
    std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
  } 
 
  // release the memory
  for(int i=0; i<N; i++)
  {
    delete [] pointsGT[i];
    delete [] pointsUndef[i];
  }

  delete [] pointsGT;
  delete [] pointsUndef;

  delete [] reprojError;
  delete [] insideArea;
  
  return 0;

}

double pointLineDistance(double annot2D[3], double xRaySource[3], double annot3D[3])
{
  double distance; // the value to be returned

  double vector1[3]; // vector between annot3D and annot2D
  double vector2[3]; // vector between annot3D and xRaySource

  vector1[0] = annot3D[0] - annot2D[0];
  vector1[1] = annot3D[1] - annot2D[1];
  vector1[2] = annot3D[2] - annot2D[2];

  vector2[0] = annot3D[0] - xRaySource[0];
  vector2[1] = annot3D[1] - xRaySource[1];
  vector2[2] = annot3D[2] - xRaySource[2];

  // calculate the cross product of the 2 vectors
  double xProduct[3];
  xProduct[0] = vector1[1]*vector2[2] - vector2[1]*vector1[2];
  xProduct[1] = vector1[2]*vector2[0] - vector2[2]*vector1[0];
  xProduct[2] = vector1[0]*vector2[1] - vector2[0]*vector1[1];

  // calculate the denominator
  double line[3];
  line[0] = xRaySource[0] - annot2D[0];
  line[1] = xRaySource[1] - annot2D[1];
  line[2] = xRaySource[2] - annot2D[2];
 
  distance = sqrt( pow(xProduct[0],2)+pow(xProduct[1],2)+pow(xProduct[2],2) ) / sqrt( pow(line[0],2)+pow(line[1],2)+pow(line[2],2) );

  return distance;
}

/*void linePlaneIntersection(double p1[3], double p2[3], double zValue, double* intersection)
{
  // p1 and p2 define the line
  // I need the intersection with the plane z=zValue
  // plane equation: Ax + By + Cz + D = 0
  double A = 0.;
  double B = 0.;
  double C = 1.;
  double D = -zValue;

  if ( detailedOutput )
  {
    std::cout << "D: "<<D<< std::endl;
    std::cout<<"P1: "<<p1[0]<<", "<<p1[1]<<", "<<p1[2]<<std::endl;
    std::cout<<"P2: "<<p2[0]<<", "<<p2[1]<<", "<<p2[2]<<std::endl;
  } 

  double t;

  t = ( A*p1[0] + B*p1[1] + C*p1[2] + D ) / ( A*(p1[0]-p2[0]) + B*(p1[1]-p2[1]) + C*(p1[2]-p2[2]) ); 
 
  intersection[0] = p1[0] + t * ( p2[0]-p1[0] );
  intersection[1] = p1[1] + t * ( p2[1]-p1[1] );
  intersection[2] = p1[2] + t * ( p2[2]-p1[2] );
}*/

void linePlaneIntersection(double p1[3], double p2[3], double normal[3], double
planePoint[3], double* intersection)
{
  // p1 and p2 define the line
  // I need the intersection with the plane
  // plane equation: N dot (P - P3) = 0

  if ( detailedOutput )
  {
    std::cout << "Normal: "<<normal[0]<<", "<<normal[1]<<", "<<normal[2]<< std::endl;
    std::cout<<"P1: "<<p1[0]<<", "<<p1[1]<<", "<<p1[2]<<std::endl;
    std::cout<<"P2: "<<p2[0]<<", "<<p2[1]<<", "<<p2[2]<<std::endl;
    std::cout<<"Plane point: "<<planePoint[0]<<", "<<planePoint[1]<<", "<<planePoint[2]<<std::endl;
  } 

  double u;

  u = ( normal[0]*(planePoint[0]-p1[0]) + 
        normal[1]*(planePoint[1]-p1[1]) +
	normal[2]*(planePoint[2]-p1[2]) ) 
    / ( normal[0]*(p2[0]-p1[0]) + 
        normal[1]*(p2[1]-p1[1]) +
        normal[2]*(p2[2]-p1[2]) ) ; 
 
  intersection[0] = p1[0] + u * ( p2[0]-p1[0] );
  intersection[1] = p1[1] + u * ( p2[1]-p1[1] );
  intersection[2] = p1[2] + u * ( p2[2]-p1[2] );
}

template <typename ImageTypePointType, 
          typename TransformTypePointer, 
	  typename InterpolatorTypeInputPointType,
	  typename InterpolatorTypeOutputPointType,
	  typename OutputVnlVectorType> 
void forwardProjection(	ImageTypePointType physPoint3DGT, 
			TransformTypePointer transformIni, 
			InterpolatorTypeInputPointType focalPoint,
			double* origin,
			double* intersection)
{
  if ( detailedOutput )
    std::cout << "In the forward projection ... " << std::endl;

  if ( detailedOutput )
    std::cout << "The physical 3D GT point is: " << physPoint3DGT << std::endl;

  ImageTypePointType transformedPoint3DGT;
  transformedPoint3DGT = transformIni->BackTransform(physPoint3DGT);

  if ( detailedOutput )
    std::cout << "The back-transformed 3D GT point is: " << transformedPoint3DGT << std::endl;

  /*ImageTypePointType testPoint3DGT;
  testPoint3DGT = transformIni->TransformPoint(transformedPoint3DGT);
  if ( detailedOutput )
    std::cout << 
    "The TEST 3D GT point that should be equal to the physical point 3D GT is: "
     << testPoint3DGT << std::endl;
  */

  // calculate intersection of the line (connecting the transformed3DGT point and the X-ray source)
  // with the plane (projection slice in 3D)
  //double intersection[3] = {0.0, 0.0, 0.0};
  double xRaySourceIni[3] = {focalPoint[0], focalPoint[1], focalPoint[2]}; 
  double pointTrans3DGT[3] = {transformedPoint3DGT[0], transformedPoint3DGT[1], transformedPoint3DGT[2]}; 
  double planePoint[3] = {origin[0], origin[1], origin[2]}; 

  // the normal of the plane was before transformation [0,0,1] since the plane was z=value
  // now I need to tranform it with transformIni, to see where to position the plane
  OutputVnlVectorType normalOriginal;
  normalOriginal[0] = 0;
  normalOriginal[1] = 0;
  normalOriginal[2] = 1;

  if ( detailedOutput )
  {
    std::cout << "The transform used in fwd projection is: "<<transformIni->GetParameters() << std::endl;
    std::cout << "with matrix:\n"<<transformIni->GetMatrix() << std::endl;
  }  
  
  double normal[3]={normalOriginal[0],normalOriginal[1],normalOriginal[2]};

  if ( detailedOutput )
    std::cout << "The plane is: z = " << origin[2] << std::endl;
  //linePlaneIntersection(xRaySourceIni,pointTrans3DGT,transformed2Dorigin[2], intersection);
  linePlaneIntersection(xRaySourceIni, pointTrans3DGT, normal, planePoint, intersection);
  if ( detailedOutput )
    std::cout << "The intersection point is: " <<intersection[0]<<" "<<intersection[1]<<" "<<intersection[2]<<std::endl;

}


template <typename ImageTypePointType, 
          typename TransformTypePointer,
	  typename InterpolatorTypeInputPointType,
	  typename InterpolatorTypeOutputPointType> 
double reprojectionError(	ImageTypePointType physPoint, 
                        ImageTypePointType physPoint3D, 
			TransformTypePointer transform, 
			TransformTypePointer transformIni,
			InterpolatorTypeInputPointType focalPoint)
{
  if ( detailedOutput )
    std::cout << "In the reprojection error ... " << std::endl;

  //find the line equation, given two points
  //transform point in image, into world coordinates
  if ( detailedOutput )
    std::cout << "The physical 3D point is: " << physPoint3D << std::endl;

  // now the 3D point needs to be transformed to the initial position 
  ImageTypePointType transformedPoint3D;
  transformedPoint3D = transform->BackTransform(physPoint3D);
  if ( detailedOutput )
    std::cout << "The back-transformed 3D point is: " << transformedPoint3D << std::endl;

  // calculate the 3D distance between the 3D point that was annotated and the line
  // that connects the 2D annotated point and the xRaySource
  double distance;

  double xRaySource[] = {focalPoint[0], focalPoint[1], focalPoint[2]}; 
  double annot2D[] = {physPoint[0], physPoint[1], physPoint[2]}; 
  double annot3D[] = {transformedPoint3D[0], transformedPoint3D[1], transformedPoint3D[2]}; 

  distance = pointLineDistance(annot2D, xRaySource, annot3D);
  if ( detailedOutput )
    std::cout << "The distance/reprojection error is: " << distance << std::endl;  

  return distance;
}


