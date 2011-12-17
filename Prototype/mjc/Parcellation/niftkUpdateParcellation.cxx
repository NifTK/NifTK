/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <set>
#include <map>
#include <vector>
#include <math.h>
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkDataArray.h"
#include "vtkCellData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkType.h"
#include "vtkCellTypes.h"
#include "vtkCellLinks.h"
#include "vtkCell.h"
#include "vtkFloatArray.h"
#include "vtkMath.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itkSSDImageToImageMetric.h"
#include "itkIdentityTransform.h"
#include "itkArray.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Update parcellation. This registers the surface image to the mean label image." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -s surfaceFile.vtk -m meanLabelImage.nii -o outputSurfaceFile.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -s    <filename>        The VTK PolyData file containing a spherical surface, with scalar label values at each point, and lines connecting points." << std::endl;
    std::cout << "    -m    <filename>        The mean label histogram represented as an image." << std::endl;
    std::cout << "    -o    <filename>        The output VTK PolyData file containing a spherical surface, with the same connectivity as input, but different labels." << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl; 
    std::cout << "    -nl   [int]  72         The number of labels" << std::endl;
    std::cout << "    -ni   [int] 100         The max number of iterations" << std::endl;
    std::cout << "    -ga   [float] 10        The Gamma (from spherical demons paper)" << std::endl;
    std::cout << "    -si   [int] 10          The smoothing iterations (from spherical demons paper)" << std::endl;
    std::cout << "    -ss   [float] 10        Step size when testing where to move. Default 2mm." << std::endl;
    std::cout << "    -averageVec             Use vector averaging when choosing next label" << std::endl;
    std::cout << "    -forceNoConnections     Force an update to happen, just depending on a different label, ie. even if a point has no connections" << std::endl;
    std::cout << "    -forceBoundary          Force an update to happen, even if changing a points label did not improve similarity. This may be useful to move a whole boundary cocohesively" << std::endl;
    std::cout << "    -testLabel [int]        Specify a label to test, and we only optimise that one" << std::endl;
    std::cout << "    -smooth [float]         Smooth connections in space with gaussian sigma=float" << std::endl;
  }

struct arguments
{
  std::string surfaceDataFile;
  std::string meanMatrixFile; 
  std::string outputSurfaceFile;
  int numberOfLabels;
  int numberOfIterations;
  int numberOfSmoothingIterations;
  double gamma;
  double stepSize;
  bool useVectorAveraging;
  bool forceNoConnections;
  bool forceBoundary;
  int testLabel;
  // These for connection smoothing
  bool smooth;
  double smoothcutoff;
  double smoothvariance;
};

struct LabelInfoType
{
  vtkIdType pointNumber;
  vtkIdType nextPointNumber;
  vtkIdType pointNumberAtOtherEndOfLine;
  int currentLabel;
  int nextLabel;
  int otherLabel;
};

struct LabelUpdateInfoType
{
  vtkIdType pointNumber;
  int currentLabel;
  int changedLabel;
};


#define CUTOFFRANGE 3.0

class pointAndWeight
  {
  public:
    vtkIdType m_point;
    double m_weight;
    pointAndWeight (vtkIdType point, double weight) {
      m_point = point;
      m_weight = weight;
    }
  };

  typedef std::vector<pointAndWeight> pointAndWeightVectorT;
  typedef std::map<vtkIdType, pointAndWeightVectorT> neighbourMapT;


double vectorMagnitude(double *a)
  {
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
  }

void normalise(double *a, double *b)
  {
    double magnitude = vectorMagnitude(a);
    b[0] /= magnitude;
    b[1] /= magnitude;
    b[2] /= magnitude;
  }

double distanceBetweenPoints(double* a, double *b)
  {
    return sqrt(
          ((a[0]-b[0]) * (a[0]-b[0])) 
        + ((a[1]-b[1]) * (a[1]-b[1]))
        + ((a[2]-b[2]) * (a[2]-b[2]))
        );
  }


double GaussianAmplitude (double mean, double variance, double position)
  {
    // vtk 5.7 vtkMath provides this function...
    return std::exp(- (position-mean)*(position-mean)/(2.0*variance) ) / std::sqrt ( 2 * NIFTK_PI * variance ) ;
  }


vtkIdType getClosestPoint(
    vtkPoints *points,
    double *point
    )
  {
    
    vtkIdType closestIndex = -1;
    
    double distance = 0;
    double closestDistance = std::numeric_limits<double>::max();
    double pointOnSurface[3];
    
    vtkIdType numberOfPoints = points->GetNumberOfPoints();
    
    for (vtkIdType i = 0; i < numberOfPoints; i++)
      {
        points->GetPoint(i, pointOnSurface);
        
        distance =  distanceBetweenPoints(point, pointOnSurface);
        if (distance < closestDistance)
          {
            closestIndex = i;
            closestDistance = distance;
          }
      }    
    return closestIndex;
  }

double totalArray(unsigned int size, double *a)
  {
    double result = 0;
    
    for (unsigned int i = 0; i < size; i++)
      {
        result += a[i];
      }        
    
    return result;
  }

void copyArray(unsigned int size, double *a, double *b)
  {
    for (unsigned int i = 0; i < size; i++)
      {
        b[i] = a[i];
      }        
  }

void zeroArray(unsigned int size, double *a)
  {
    for (unsigned int i = 0; i < size; i++)
      {
        a[i] = 0;
      }    
  }

void divideArray(unsigned int size, double *a, double divisor)
  {
    for (unsigned int i = 0; i < size; i++)
      {
        a[i] /= divisor;
      }    
  }

double computeSSDOfArray(unsigned int size, double* a, double *b)
  {
    double result = 0;
    
    for (unsigned int i = 0; i < size; i++)
      {
        result += ((a[i]-b[i])*(a[i]-b[i]));
      }
    
    return result;
  }

double computeSSDOfLines(unsigned int arraySize, unsigned int xSize, double* testLabelImageArray, double* meanLabelImageArray, vtkDataArray *scalarData, vtkCellArray *lines, neighbourMapT neighbours, double cutoffthreshold)
  {
    int label1, label2;
    vtkIdType numberOfPointsInCell;
    vtkIdType *pointIdsInLine = new vtkIdType[2];  
    double totalWeight = 0;
    
    zeroArray(arraySize, testLabelImageArray);
    
    // Get each line, plot label value in labelImage, building up a histogram.
    lines->InitTraversal();
    while(lines->GetNextCell(numberOfPointsInCell, pointIdsInLine))
      {
	vtkIdType p1, p2;
	double connectweight;
	p1 = pointIdsInLine[0];
	p2 = pointIdsInLine[1];
      
	for ( pointAndWeightVectorT::iterator p1neighbour = neighbours[p1].begin();
	      p1neighbour != neighbours[p1].end() ; p1neighbour++ ) {
	  for ( pointAndWeightVectorT::iterator p2neighbour = neighbours[p2].begin();
		p2neighbour != neighbours[p2].end() ; p2neighbour++ ) {

	    label1 = (int)scalarData->GetTuple1(p1neighbour->m_point);
	    label2 = (int)scalarData->GetTuple1(p2neighbour->m_point);

	    connectweight = p1neighbour->m_weight * p2neighbour->m_weight;
	    if (connectweight > cutoffthreshold ) {
        
	      testLabelImageArray[label2*xSize + label1] += connectweight;
	      testLabelImageArray[label1*xSize + label2] += connectweight;
	      totalWeight += connectweight * 2.0;
	    }
	  }
	}
      }
    
    // Normalise
    //divideArray(arraySize, testLabelImageArray, lines->GetNumberOfCells() * 2);
    divideArray(arraySize, testLabelImageArray, totalWeight);
    
    // Evaluate the similarity measure
    double currentScore = computeSSDOfArray(arraySize, testLabelImageArray, meanLabelImageArray);
    
    return currentScore;
  }

vtkPolyData* extractBoundaryModel(vtkPolyData *polyData)
  {
    vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
    vtkPoints *points = polyData->GetPoints();
    vtkIdType numberOfPoints = polyData->GetNumberOfPoints();
    vtkIdType pointNumber = 0;
    vtkIdType cellNumber = 0;
    short unsigned int numberOfCellsUsingCurrentPoint;
    vtkIdType *listOfCellIds;
    vtkIdType *listOfPointIds;
    int currentLabel;
    int nextLabel;
    std::set<int> nextLabels;
    vtkIdType numberOfPointsInCurrentCell;
    double point[3];
    vtkIdType *outputLine = new vtkIdType[2];
    
    vtkPoints *outputPoints = vtkPoints::New();
    outputPoints->SetDataTypeToFloat();
    outputPoints->Allocate(numberOfPoints);
    
    vtkIntArray *outputLabels = vtkIntArray::New();
    outputLabels->SetNumberOfComponents(1);
    outputLabels->SetNumberOfValues(numberOfPoints);
    
    vtkCellArray *outputLines = vtkCellArray::New();
    
    vtkFloatArray *outputVectors = vtkFloatArray::New();
    outputVectors->SetNumberOfComponents(3);
    outputVectors->SetNumberOfValues(numberOfPoints);
    
    unsigned long int numberOfBoundaryPoints = 0;
    unsigned long int numberOfJunctionPoints = 0;
    
    // First we identify junction points, those with at least 3 labels in neighbours.
    for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
      {
        polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);

        currentLabel = (int)scalarData->GetTuple1(pointNumber);
        nextLabels.clear();

        // Get all cells containing this point, if triangle, store any surrounding labels that differ.
        for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
          {
            polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                          
            if (numberOfPointsInCurrentCell == 3)
              {
                for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                  {
                    nextLabel = (int)scalarData->GetTuple1(listOfPointIds[i]);
                    if (nextLabel != currentLabel)
                      {
                        nextLabels.insert(nextLabel);
                      }
                  }                  
              }
          }

        // Using nextLabels we know how many labels are in neighbourhood, so we can label points as 
        // 0 = within a region.
        // 1 = on boundary
        // 2 = at junction
        
        points->GetPoint(pointNumber, point);
        outputPoints->InsertPoint(pointNumber, point);
        outputVectors->InsertTuple3(pointNumber, 0, 0, 0);
        
        if (nextLabels.size() == 0)
          {
            // We are in a region
            outputLabels->SetValue(pointNumber, 0);
          }
        else if (nextLabels.size() == 1)
          {
            // We are on boundary
            outputLabels->SetValue(pointNumber, 1);
            numberOfBoundaryPoints++;
          }
        else
          {
            // We are on junction
            outputLabels->SetValue(pointNumber, 2);
            numberOfJunctionPoints++;
          }
      }
    
    // Also, for each boundary or junction point, connect it to all neighbours that are also boundary or junction points.
    for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
      {
        if (outputLabels->GetValue(pointNumber) > 0)
          {
            polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);
            
            for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
              {
                polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                
                if (numberOfPointsInCurrentCell == 3)
                  {
                    for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                      {
                        
                        if (listOfPointIds[i] != pointNumber && outputLabels->GetValue(listOfPointIds[i]) > 0)
                          {
                            outputLine[0] = pointNumber;
                            outputLine[1] = listOfPointIds[i];
                            outputLines->InsertNextCell(2, outputLine);
                            
                          }
                      }                    
                  }
              }                    
          }
      }
    
    std::cout << "Number of boundary points=" << numberOfBoundaryPoints << ", number of junction points=" << numberOfJunctionPoints << std::endl;
    
    vtkPolyData *outputPolyData = vtkPolyData::New();
    outputPolyData->SetPoints(outputPoints);
    outputPolyData->GetPointData()->SetScalars(outputLabels);
    outputPolyData->GetPointData()->SetVectors(outputVectors);
    outputPolyData->SetLines(outputLines);
    outputPolyData->BuildCells();
    outputPolyData->BuildLinks();
    
    outputPoints->Delete();
    outputLabels->Delete();
    outputLines->Delete();
    outputVectors->Delete();
    
    return outputPolyData;
  }


bool hasConnectedLines(vtkPolyData* polyData, vtkIdType pointNumber)
{
	bool result = false;
	short unsigned int numberOfCellsUsingCurrentPoint = 0;
	vtkIdType *listOfCellIds;
	vtkIdType *listOfPointIds;
	vtkIdType cellNumber = 0;
	vtkIdType numberOfPointsInCurrentCell = 0;

    polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);

    for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
    {
        polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
        if (numberOfPointsInCurrentCell == 2)
        {
        	result = true;
        	break;
        }
    }
    return result;
}

/**
 * \brief Performs update to parcellation.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.numberOfLabels = 72;
  args.numberOfIterations = 100;
  args.gamma = 10;
  args.numberOfSmoothingIterations = 10;
  args.stepSize = 10;
  args.useVectorAveraging = false;
  args.forceNoConnections = false;
  args.forceBoundary = false;
  args.testLabel = -1;
  args.smooth = false;
  args.smoothcutoff = 0;
  args.smoothvariance = 0;


  bool allLabels = true;


  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-s") == 0){
      args.surfaceDataFile=argv[++i];
      std::cout << "Set -i=" << args.surfaceDataFile;
    }
    else if(strcmp(argv[i], "-m") == 0){
      args.meanMatrixFile=argv[++i];
      std::cout << "Set -o=" << args.meanMatrixFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputSurfaceFile=argv[++i];
      std::cout << "Set -o=" << args.outputSurfaceFile;
    }   
    else if(strcmp(argv[i], "-nl") == 0){
      args.numberOfLabels=atoi(argv[++i]);
      std::cout << "Set -nl=" << niftk::ConvertToString(args.numberOfLabels);
    }
    else if(strcmp(argv[i], "-ni") == 0){
      args.numberOfIterations=atoi(argv[++i]);
      std::cout << "Set -ni=" << niftk::ConvertToString(args.numberOfIterations);
    }        
    else if(strcmp(argv[i], "-si") == 0){
      args.numberOfSmoothingIterations=atoi(argv[++i]);
      std::cout << "Set -si=" << niftk::ConvertToString(args.numberOfIterations);
    }        
    else if(strcmp(argv[i], "-ga") == 0){
      args.gamma=atof(argv[++i]);
      std::cout << "Set -ga=" << niftk::ConvertToString(args.gamma);
    }
    else if(strcmp(argv[i], "-ss") == 0){
      args.stepSize=atof(argv[++i]);
      std::cout << "Set -ss=" << niftk::ConvertToString(args.stepSize);
    }
    else if(strcmp(argv[i], "-averageVec") == 0){
      args.useVectorAveraging=true;
      std::cout << "Set -averageVec=" << niftk::ConvertToString(args.useVectorAveraging);
    }
    else if(strcmp(argv[i], "-forceNoConnections") == 0){
      args.forceNoConnections=true;
      std::cout << "Set -forceNoConnections=" << niftk::ConvertToString(args.forceNoConnections);
    }
    else if(strcmp(argv[i], "-forceBoundary") == 0){
      args.forceBoundary=true;
      std::cout << "Set -forceBoundary=" << niftk::ConvertToString(args.forceBoundary);
    }
    else if(strcmp(argv[i], "-testLabel") == 0){
      args.testLabel=atoi(argv[++i]);
      allLabels = false;
      std::cout << "Set -testLabel=" << niftk::ConvertToString(args.testLabel);
    }
    else if(strcmp(argv[i], "-smooth") == 0){
      args.smooth=true;
      char *argopt = argv[++i];
      double argoptf;
      std::cout << "Set -smooth=" << niftk::ConvertToString(argopt);
      if ( sscanf(argopt,"%lf",&argoptf) != 1 ) {
	std::cerr << argv[0] << ":\tParameter " << argopt << " not allowed." << std::endl;
	return -1;
      }
      args.smoothvariance = argoptf * argoptf;
      args.smoothcutoff = CUTOFFRANGE * argoptf;
    }    
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.surfaceDataFile.length() == 0 || args.outputSurfaceFile.length() == 0 || args.meanMatrixFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  // Load poly data containing surface and connections.
  vtkPolyDataReader *surfaceReader = vtkPolyDataReader::New();
  surfaceReader->SetFileName(args.surfaceDataFile.c_str());
  surfaceReader->Update();

  // Get hold of the point data, number points, scalars etc.
  vtkPolyData *polyData = surfaceReader->GetOutput();
  vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
  vtkPoints *points = polyData->GetPoints();
  vtkIdType numberOfPoints = polyData->GetNumberOfPoints();
  vtkIdType numberOfCells = polyData->GetNumberOfCells();
  
  if (scalarData == NULL)
    {
      std::cerr << "Couldn't find scalar data (labels)." << std::endl;
      return EXIT_FAILURE;
    }

  // Debug info
  std::cout << "Loaded file " << args.surfaceDataFile \
    << " containing " << surfaceReader->GetOutput()->GetNumberOfPolys()  \
    << " triangles, and " << surfaceReader->GetOutput()->GetNumberOfLines() \
    << " lines, and " << numberOfPoints \
    << " points, and " << numberOfCells \
    << " cells." \
    << std::endl;

  // Load mean image. This is the mean connectivity.
  typedef itk::Image<float, 2> LabelImageType;
  typedef itk::ImageFileReader<LabelImageType> LabelImageReaderType;
  typedef itk::ImageFileWriter<LabelImageType> LabelImageWriterType;
  
  LabelImageReaderType::Pointer meanLabelImageReader = LabelImageReaderType::New();
  meanLabelImageReader->SetFileName(args.meanMatrixFile);
  meanLabelImageReader->Update();
  
  LabelImageType::Pointer meanLabelImage = meanLabelImageReader->GetOutput();
  LabelImageType::SizeType meanLabelImageSize = meanLabelImage->GetLargestPossibleRegion().GetSize();
  unsigned int arraySize = meanLabelImageSize[0] * meanLabelImageSize[1];
  unsigned int arrayIndex = 0;
  
  std::cout << "Loaded image " << args.meanMatrixFile << " with size " << meanLabelImageSize << std::endl;

  // Variables to work with
  typedef std::pair<double, LabelInfoType> PairType;
  typedef std::multimap<double, LabelInfoType> MapType;
  typedef std::pair<vtkIdType, int> LabelPairType;
  std::vector<LabelPairType>::iterator it;
  std::vector<LabelPairType> nextLabels;
  LabelImageType::IndexType index;
  std::vector<LabelUpdateInfoType> updateLabels;
  MapType map;
  vtkIdType iterationNumber = 0;
  vtkIdType pointNumber = 0;
  vtkIdType nextPointNumber = 0;
  vtkIdType pointNumberAtOtherEndOfLine = 0;
  vtkIdType cellNumber = 0;
  vtkIdType numberOfPointsInCurrentCell = 0;
  vtkIdType *listOfCellIds;
  vtkIdType *listOfPointIds;
  vtkIdType neighbourPoints[100];
  vtkPolyData *boundaryModel = NULL;
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  unsigned long int numberOfCellsChanged = 0;
  short unsigned int numberOfCellsUsingCurrentPoint = 0;
  int currentLabel = 0;
  int nextLabel = 0;
  int labelAtOtherEndOfLine = 0;
  double nextScore = 0;
  double vector[3];
  double weightedVector[3];
  double currentPoint[3];
  double nextPoint[3];
  double weight = 0;
  double *tmp;
  double normalizedVector[3];
  int labelThatWeAreMoving = 0;
  
  // Build cells and links for quick lookup.
  vtkCellArray *lines = polyData->GetLines();
  polyData->BuildCells();
  polyData->BuildLinks();

  // Create a mean label image and a test label image
  double* meanLabelImageArray = new double[arraySize];
  double* testLabelImageArray = new double[arraySize];
  double* nextLabelImageArray = new double[arraySize];
  
  double bestScore = 0;
  int bestLabel = 0;
  vtkIdType bestPointNumber = 0;
  
  // Copy mean image into its array, and initialize testLabelImageArray.
  double totalInMeanArray=0;
  for (unsigned int y = 0; y < meanLabelImageSize[1]; y++)
    {
      for (unsigned int x = 0; x < meanLabelImageSize[0]; x++)
        {
          arrayIndex = y*meanLabelImageSize[0] + x;
          index[0] = x;
          index[1] = y;
          meanLabelImageArray[arrayIndex] = meanLabelImage->GetPixel(index);
          testLabelImageArray[arrayIndex] = 0;
          totalInMeanArray += totalInMeanArray;
        }
    }

  // Work out total number of label values.
  std::set<int> labelNumbers;
  for (vtkIdType pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
    {
      labelNumbers.insert((int)scalarData->GetTuple1(pointNumber));
    }
  std::set<int>::iterator labelNumbersIterator;
  std::cout << "Found labels:";
  
  for (labelNumbersIterator = labelNumbers.begin(); labelNumbersIterator != labelNumbers.end(); labelNumbersIterator++)
    {
      std::cout << (*labelNumbersIterator) << " ";
    }
  std::cout << std::endl;
  
  // Mean array should already be normalised
  std::cout << "Total of averaged histogram = " << totalArray(arraySize, meanLabelImageArray) << std::endl;
   
  double histogramAdjustment = 1.0/(polyData->GetLines()->GetNumberOfCells() * 2);
  bool atLeastOneLabelImproved = false;
  unsigned long int totalNumberChangedPerIteration = 0;
  
  neighbourMapT neighbours;
  double cutoffthreshold = args.smooth ? GaussianAmplitude(0,args.smoothvariance,args.smoothcutoff) : 0;
  if ( ! args.smooth ) {
    for ( vtkIdType thispoint=0; thispoint < polyData->GetNumberOfPoints(); thispoint++) {
      pointAndWeight nextneighbour(thispoint, 1);
      neighbours[thispoint].push_back(nextneighbour);
    }
  }
  else {
    for ( vtkIdType thispoint = 0 ; thispoint < polyData->GetNumberOfPoints(); thispoint++) {
      for ( vtkIdType neighbourpoint = 0 ; neighbourpoint < polyData->GetNumberOfPoints() ; neighbourpoint++ ) {
	// Could do one run as it's symmetric, but lets get it right first.
	double pointa[3];
	double pointb[3];
	polyData->GetPoint(thispoint,pointa);
	polyData->GetPoint(neighbourpoint,pointb);
	double distance = vtkMath::Distance2BetweenPoints(pointa, pointb);
	if ( distance < args.smoothcutoff ) {
	  const double mean = 0;
	  double weight = GaussianAmplitude(mean,args.smoothvariance,distance);
	  pointAndWeight nextneighbour(neighbourpoint, weight);
	  neighbours[thispoint].push_back(nextneighbour);
	}
      }
    }
  }


  do
    {
      atLeastOneLabelImproved = false;
      totalNumberChangedPerIteration = 0;
      
      // Extract boundary model.
      boundaryModel =  extractBoundaryModel(polyData);
  
      scalarData = (polyData->GetPointData()->GetScalars());
      double currentScore = computeSSDOfLines(arraySize, meanLabelImageSize[0], testLabelImageArray, meanLabelImageArray, scalarData, lines, neighbours, cutoffthreshold);

      std::cout << "[" << iterationNumber << "]:Before update score =" << currentScore << std::endl;
      
      // Iterate over each label in turn
      for(labelNumbersIterator = labelNumbers.begin(); labelNumbersIterator != labelNumbers.end(); labelNumbersIterator++)
        {
          if (allLabels || ((*labelNumbersIterator) == args.testLabel))
            {
              scalarData = (polyData->GetPointData()->GetScalars());
              
              // Copy scalar data
              vtkIntArray *newScalarData = vtkIntArray::New();
              newScalarData->SetNumberOfComponents(1);
              newScalarData->SetNumberOfValues(numberOfPoints);
              for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
                {
                  newScalarData->InsertTuple1(pointNumber, (int)(scalarData->GetTuple1(pointNumber)));
                }

              // We only move one label at a time
              labelThatWeAreMoving = (*labelNumbersIterator);
          
              // Possibly some redundancy here.
              // We have a map of possible candidates (that can be smoothed etc.)
              // also, a running list of ones we are updating, so we can easily roll back.
              map.clear();
              updateLabels.clear();
              
              // Evaluate the similarity measure
              double currentScore = computeSSDOfLines(arraySize, meanLabelImageSize[0], testLabelImageArray, meanLabelImageArray, scalarData, lines, neighbours, cutoffthreshold);
              //std::cout << "[" << iterationNumber << "][" << labelThatWeAreMoving<< "]:\tBefore update score =" << currentScore << std::endl;
              
              // For each point:
              // If its a boundary (label > 0), then we compute a 'force', and store the vector on the boundary model.

              for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
                {
                  currentLabel = (int)scalarData->GetTuple1(pointNumber);
                  
                  if (currentLabel == labelThatWeAreMoving && boundaryModel->GetPointData()->GetScalars()->GetTuple1(pointNumber) > 0)
                    {
                      polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);
                      nextLabels.clear();

                      // Get neighbouring labels
                      for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
                        {
                          polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                          
                          if (numberOfPointsInCurrentCell == 3)
                            {
                              for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                                {
                                  nextPointNumber = listOfPointIds[i];
                                  nextLabel = (int)scalarData->GetTuple1(nextPointNumber);

                                  if (nextLabel != currentLabel && (args.forceNoConnections || hasConnectedLines(polyData, nextPointNumber)))
                                    {
                                      nextLabels.push_back(LabelPairType(nextPointNumber, nextLabel));
                                    }
                                }                  
                              
                            }
                        } // end for each cell, working out neighboring labels.

                      // If size of neighboring labels > 0, then we must be on a boundary.
                      // The nextLabels vector contains the label and the point number it came from.

                      if (nextLabels.size() > 0)
                        {
                          bestLabel = currentLabel;
                          bestScore = currentScore;
                          bestPointNumber = pointNumber;

                          for (it = nextLabels.begin(); it != nextLabels.end(); it++)
                            {
                              copyArray(arraySize, testLabelImageArray, nextLabelImageArray);
                              
                              nextLabel = (*it).second;
                              
                              // So for each line connected to this point, get the label at other end of the line.
                              for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
                                {
                                  polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                                  
                                  // Only do lines, not triangles.
                                  if (numberOfPointsInCurrentCell == 2)
                                    {
                                      pointNumberAtOtherEndOfLine = 0;

                                      for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                                        {
                                          pointNumberAtOtherEndOfLine = listOfPointIds[i];
                                          
                                          if (pointNumberAtOtherEndOfLine != pointNumber)
                                            {
                                              labelAtOtherEndOfLine = (int)scalarData->GetTuple1(pointNumberAtOtherEndOfLine);
                                              
                                              nextLabelImageArray[labelAtOtherEndOfLine*meanLabelImageSize[0] + currentLabel] -= histogramAdjustment;
                                              nextLabelImageArray[labelAtOtherEndOfLine*meanLabelImageSize[0] + nextLabel] += histogramAdjustment;
                                            }
                                        }
                                    } // end if line, not triangle
                                } // end for each line
                              
                              nextScore = computeSSDOfArray(arraySize, nextLabelImageArray, meanLabelImageArray);

                              if (args.forceBoundary || nextScore < bestScore)
                                {
                                  bestLabel = nextLabel;
                                  bestScore = nextScore;
                                  bestPointNumber = (*it).first;
                                }                          

                            }
                          
                          // Check if we found any improvement at all amongst the neighboring labels.
                          
                          if (args.forceBoundary || bestScore < currentScore)
                            {
                              LabelInfoType labelInfo;
                              labelInfo.currentLabel = currentLabel;
                              labelInfo.nextLabel = bestLabel;
                              labelInfo.otherLabel = labelAtOtherEndOfLine;
                              labelInfo.pointNumber = pointNumber;
                              labelInfo.nextPointNumber = bestPointNumber;
                              labelInfo.pointNumberAtOtherEndOfLine = pointNumberAtOtherEndOfLine;
                              map.insert(PairType(bestScore, labelInfo));
                            }
                        } // end: if we are on boundary          
                    } // end: if we are on boundary
                } // end for each point

              // Zero vectors
              for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
                {
                  boundaryModel->GetPointData()->GetVectors()->SetTuple3(pointNumber, 0, 0, 0);
                }

              // At this point we have a list of potential candidates for improvement.
              // So create vectors for each one. Decide if we are averaging, or picking directly.
              
              MapType::iterator mapIterator;
              for (mapIterator = map.begin(); mapIterator != map.end(); mapIterator++)
                {
                  LabelInfoType labelInfo = (*mapIterator).second;
                  
                  pointNumber = labelInfo.pointNumber;
                  nextLabel = labelInfo.nextLabel;
                  currentLabel = labelInfo.currentLabel;
                  bestPointNumber = labelInfo.nextPointNumber;

                  if (args.useVectorAveraging)
                  {
                      // Iterate round the neighbourhood of a given point, and compute an update vector.

                      polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);

                      vector[0] = 0;
                      vector[1] = 0;
                      vector[2] = 0;

                      int neighboringPointsWithNextLabel = 0;

                      for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
                        {
                          polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);

                          if (numberOfPointsInCurrentCell == 3)
                            {
                              for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                                {
                                  if (listOfPointIds[i] != pointNumber && (int)scalarData->GetTuple1(listOfPointIds[i]) == nextLabel)
                                    {

                                      points->GetPoint(pointNumber, currentPoint);
                                      points->GetPoint(listOfPointIds[i], nextPoint);

                                      vector[0] += (nextPoint[0] - currentPoint[0]);
                                      vector[1] += (nextPoint[1] - currentPoint[1]);
                                      vector[2] += (nextPoint[2] - currentPoint[2]);

                                      neighboringPointsWithNextLabel++;
                                    }
                                }
                            }
                        }

                      for (int i = 0; i < 3; i++)
                        {
                          vector[i] /= (double)neighboringPointsWithNextLabel;
                        }

                  }
                  else
                  {
                      points->GetPoint(pointNumber, currentPoint);
                      points->GetPoint(bestPointNumber, nextPoint);

                      for (int i = 0; i < 3; i++)
                        {
                          vector[i] = nextPoint[i] - currentPoint[i];
                        }
                  }


                  boundaryModel->GetPointData()->GetVectors()->SetTuple3(pointNumber, vector[0], vector[1], vector[2]);

                }      
              
              // Now smoothing iterations for vectors.
              for (int i = 0; i < args.numberOfSmoothingIterations; i++)
                {
                  //std::cout << "Smoothing iteration:" << i << std::endl;
                  
                  vtkFloatArray *smoothedVectors = vtkFloatArray::New();
                  smoothedVectors->SetNumberOfComponents(3);
                  smoothedVectors->SetNumberOfValues(numberOfPoints);

                  for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
                    {
                      unsigned int numberOfNeighbourPoints = 0;
                      
                      polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);
                      
                      for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
                        {
                          polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                          
                          if (numberOfPointsInCurrentCell == 3)
                            {
                              for (int j = 0; j < numberOfPointsInCurrentCell; j++)
                                {
                                  if (listOfPointIds[j] != pointNumber)
                                    {
                                      neighbourPoints[numberOfNeighbourPoints] = listOfPointIds[j];
                                      numberOfNeighbourPoints++;
                                    }
                                }
                            }
                        }
                      
                      // Get all vectors round neighbourhood, and compute weighted average
                      weight = 1.0/(1.0 + numberOfNeighbourPoints * exp(-1/(2.0 * args.gamma)));
                      
                      tmp = boundaryModel->GetPointData()->GetVectors()->GetTuple3(pointNumber);
                      
                      weightedVector[0] = weight*tmp[0];
                      weightedVector[1] = weight*tmp[1];
                      weightedVector[2] = weight*tmp[2];
                      
                      weight = exp(-1/(2.0 * args.gamma)) / (1.0+numberOfNeighbourPoints* exp(-1/(2.0 * args.gamma)));
                      
                      for (unsigned int j = 0; j < numberOfNeighbourPoints; j++)
                        {
                          tmp = boundaryModel->GetPointData()->GetVectors()->GetTuple3(neighbourPoints[j]);
                          weightedVector[0] += weight*tmp[0];
                          weightedVector[1] += weight*tmp[1];
                          weightedVector[2] += weight*tmp[2];                  
                        }
                      
                      smoothedVectors->InsertTuple3(pointNumber, weightedVector[0], weightedVector[1], weightedVector[2]);
                    }
                  
                  boundaryModel->GetPointData()->SetVectors(smoothedVectors);
                  smoothedVectors->Delete();
                }
               
              // Now for each point, iterate through whole mesh, look at the vector. If vector non-zero move in that direction, 
              // find closest point, and if different label, swap it for current label.
              numberOfCellsChanged = 0;
              
              for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
                {
                  if (boundaryModel->GetPointData()->GetScalars()->GetTuple1(pointNumber) > 0)
                    {
                      tmp = boundaryModel->GetPointData()->GetVectors()->GetTuple3(pointNumber);
                      normalise(tmp, normalizedVector);
                      
                      if (vectorMagnitude(tmp) > 0.0001)
                        {
                          points->GetPoint(pointNumber, currentPoint);
                          
                          // Add args.stepSize * vector direction.
                          for (int i = 0; i < 3; i++)
                            {
                              currentPoint[i] += args.stepSize * tmp[i];
                            }

                          unsigned int numberOfNeighbourPoints = 0;
                          
                          polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);
                          
                          for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
                            {
                              polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                              
                              if (numberOfPointsInCurrentCell == 3)
                                {
                                  for (int j = 0; j < numberOfPointsInCurrentCell; j++)
                                    {
                                      if (listOfPointIds[j] != pointNumber)
                                        {
                                          neighbourPoints[numberOfNeighbourPoints] = listOfPointIds[j];
                                          numberOfNeighbourPoints++;
                                        }
                                    }                          
                                }
                              
                            }
                          
                          vtkIdType bestIndex =  pointNumber;
                          double bestDistance = std::numeric_limits<double>::max();
                          double distance = 0;
                          
                          for (unsigned int j = 0; j < numberOfNeighbourPoints; j++)
                            {
                              points->GetPoint(neighbourPoints[j], nextPoint);
                              distance = distanceBetweenPoints(currentPoint, nextPoint);
                              
                              if (distance < bestDistance)
                                {
                                  bestDistance = distance;
                                  bestIndex = neighbourPoints[j];
                                }
                            }
                          
                          // Check label
                          if (scalarData->GetTuple1(pointNumber) != scalarData->GetTuple1(bestIndex))
                            {
                              LabelUpdateInfoType updateType;
                              updateType.pointNumber = pointNumber;
                              updateType.currentLabel = (int)scalarData->GetTuple1(pointNumber);
                              updateType.changedLabel = (int)scalarData->GetTuple1(bestIndex);
                              
                              updateLabels.push_back(updateType);

                              newScalarData->SetTuple1(pointNumber, scalarData->GetTuple1(bestIndex));
                              numberOfCellsChanged++;
                            }
                        }              
                    }
                }      
              // Evaluate the similarity measure
              
              totalNumberChangedPerIteration += numberOfCellsChanged;
              
              double nextScore = computeSSDOfLines(arraySize, meanLabelImageSize[0], testLabelImageArray, meanLabelImageArray, newScalarData, lines, neighbours, cutoffthreshold);
              //std::cout << "[" << iterationNumber << "][" << labelThatWeAreMoving<< "]:\tAfter update score =" << nextScore << ", vectors in map " << map.size() << ", numberOfCellsChanged=" << numberOfCellsChanged << std::endl;
              
              if (nextScore >= currentScore)
                {
                  //std::cout << "Rolling back last change, as there was no improvement" << std::endl;
                  for (unsigned long int i = 0; i < updateLabels.size(); i++)
                    {
                      newScalarData->SetTuple1(updateLabels[i].pointNumber, updateLabels[i].currentLabel);
                    }
                  totalNumberChangedPerIteration -= updateLabels.size();
                }
              else
                {
                  std::cout << "[" << iterationNumber << "][" << labelThatWeAreMoving<< "]:After update score =" << nextScore << std::endl;
                  atLeastOneLabelImproved = true;
                }
              
              // Store back on polyData
              polyData->GetPointData()->SetScalars(newScalarData);
              newScalarData->Delete();
                            
            } // end if allLabels
        } // end for each label

      
      scalarData = (polyData->GetPointData()->GetScalars());
      currentScore = computeSSDOfLines(arraySize, meanLabelImageSize[0], testLabelImageArray, meanLabelImageArray, scalarData, lines, neighbours, cutoffthreshold);

      std::cout << "[" << iterationNumber << "]:After update score =" << currentScore << ", numberChanged=" << totalNumberChangedPerIteration << std::endl;

      iterationNumber++;
      
    } while (atLeastOneLabelImproved && iterationNumber < args.numberOfIterations);
    
  // Write out the resulting file.
  writer->SetFileName(args.outputSurfaceFile.c_str());
  writer->SetInput(polyData);
  writer->Update();

  return EXIT_SUCCESS;
  
}
