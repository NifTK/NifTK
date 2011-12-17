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
#include <math.h> // Only used for GaussianAmplitude
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkDataArray.h"
#include "vtkCellData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkType.h"
#include "vtkMath.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"

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


double GaussianAmplitude (double mean, double variance, double position)
  {
    // vtk 5.7 vtkMath provides this function...
    return std::exp(- (position-mean)*(position-mean)/(2.0*variance) ) / std::sqrt ( 2 * NIFTK_PI * variance ) ;
  }


typedef double LabelType;
//typedef float LabelType; // Normalisation goes wrong. Not a big overhead anyway.
  
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a vtkPolyData file which is a model containing " << std::endl;
    std::cout << "     1.) lines representing connectivity between points" << std::endl;
    std::cout << "     2.) label values at each point " << std::endl;
    std::cout << "  to a matrix image of which labels are connected to each other" << std::endl << std::endl;
    std::cout << "  This was developed to improve the parcellation of the brain cortex from DTI data. " << std::endl;
    std::cout << "  So the vtkPolyData model should be a representation of the brain surface." << std::endl;
    std::cout << "  (eg. a spherical model from FreeSurfer, with a label at each point representing the parcellation region." << std::endl; 
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -s surfaceFile.vtk -l labelImage.nii [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -s    <filename>        The VTK PolyData file." << std::endl;
    std::cout << "    -l    <filename>        The output label image, effectively a co-occurance histogram." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
    std::cout << "    -labels [int] 72        The number of labels to use." << std::endl << std::endl;
    std::cout << "    -normalise              Normalise matrix (divide by total number of entries)" << std::endl;
    std::cout << "    -smooth [float]         Smooth connections in space with gaussian sigma=float" << std::endl;
  }

struct arguments
{
  std::string surfaceDataFile;
  std::string labelImageFile;
  int numberOfLabels;
  bool normalise;
  bool smooth;
  double smoothcutoff;
  double smoothvariance;
};

/**
 * \brief Converts connections and labels to a matrix.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.numberOfLabels = 72;
  args.normalise = false;
  args.smooth = false;
  args.smoothcutoff = 0;
  args.smoothvariance = 0;
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-s") == 0){
      args.surfaceDataFile=argv[++i];
      std::cout << "Set -s=" << args.surfaceDataFile;
    }
    else if(strcmp(argv[i], "-l") == 0){
      args.labelImageFile=argv[++i];
      std::cout << "Set -l=" << args.labelImageFile;
    }
    else if(strcmp(argv[i], "-n") == 0){
      args.numberOfLabels=atoi(argv[++i]);
      std::cout << "Set -n=" << niftk::ConvertToString(args.numberOfLabels);
    } 
    else if(strcmp(argv[i], "-normalise") == 0){
      args.normalise=true;
      std::cout << "Set -normalise=" << niftk::ConvertToString(args.normalise);
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
  if (args.surfaceDataFile.length() == 0 || args.labelImageFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  // Load surface
  vtkPolyDataReader *surfaceReader = vtkPolyDataReader::New();
  surfaceReader->SetFileName(args.surfaceDataFile.c_str());
  surfaceReader->Update();

  // Get hold of the point data.
  vtkPolyData *polyData = surfaceReader->GetOutput();
  unsigned long int numberOfPoints = polyData->GetNumberOfPoints();
  unsigned long int numberOfCells = polyData->GetNumberOfCells();
  
  std::cout << "Loaded file " << args.surfaceDataFile \
    << " containing " << surfaceReader->GetOutput()->GetNumberOfPolys()  \
    << " triangles, and " << surfaceReader->GetOutput()->GetNumberOfLines() \
    << " lines, and " << numberOfPoints \
    << " points, and " << numberOfCells \
    << " cells." \
    << std::endl;

  vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
  
  if (scalarData == NULL)
    {
      std::cerr << "Couldn't find scalar data (labels)." << std::endl;
      return EXIT_FAILURE;
    }

  // Create an image for the histograms
  const unsigned int Dimension = 2;
  typedef itk::Image<LabelType, Dimension> LabelImageType;
  typedef itk::ImageFileWriter<LabelImageType> LabelImageWriterType;
  
  // Create an image of that size.
  LabelImageType::SizeType size;
  LabelImageType::IndexType index;
  size[0] = args.numberOfLabels;
  size[1] = args.numberOfLabels;
  index[0] = 0;
  index[1] = 0;
  LabelImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  
  std::cout << "Allocating image size=" << size << std::endl;
  
  LabelImageType::Pointer labelImage = LabelImageType::New();
  labelImage->SetRegions(region);
  labelImage->Allocate();
  labelImage->FillBuffer(0);

  
  // Create the neighbour weights list
  typedef std::vector<pointAndWeight> pointAndWeightVectorT;
  std::map<vtkIdType, pointAndWeightVectorT> neighbours;
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

  std::cout << "Neighbours mapped" << std::endl;
  
  // Now loop through each connectivity line.
  vtkCellArray *lines = polyData->GetLines();
  vtkIdType *pointIdsInLine = new vtkIdType[2];
  vtkIdType numberOfPointsInCell;
  
  // write matrix out
  LabelType label1, label2;
  lines->InitTraversal();
  
  double cutoffthreshold = args.smooth ? GaussianAmplitude(0,args.smoothvariance,args.smoothcutoff) : 0;
  double totalWeight = 0;
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

	  label1 = (LabelType)scalarData->GetTuple1(p1neighbour->m_point);
	  label2 = (LabelType)scalarData->GetTuple1(p2neighbour->m_point);
	  connectweight = p1neighbour->m_weight * p2neighbour->m_weight;
	  if (connectweight > cutoffthreshold ) {
	    index[0] = (int)label1;
	    index[1] = (int)label2;

	    labelImage->SetPixel(index, labelImage->GetPixel(index) + connectweight);

	    index[0] = (int)label2;
	    index[1] = (int)label1;

	    labelImage->SetPixel(index, labelImage->GetPixel(index) + connectweight);
	    totalWeight += connectweight * 2.0;
	  }
	}
      }
    }
  
  if (args.normalise)
    {
      double total = 0;
      
      itk::ImageRegionIterator<LabelImageType> iterator(labelImage, labelImage->GetLargestPossibleRegion());
      //double totalPoints = lines->GetNumberOfCells() * 2.0;
      
      for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
        {
          //iterator.Set(iterator.Get()/totalPoints);
          iterator.Set(iterator.Get()/totalWeight);
          total += iterator.Get();
        }
      
      std::cout << "Total after normalisation=" << total << std::endl;
    }
  
  LabelImageWriterType::Pointer writer = LabelImageWriterType::New();
  writer->SetFileName(args.labelImageFile);
  writer->SetInput(labelImage);
  writer->Update();
  
  return EXIT_SUCCESS;
}
