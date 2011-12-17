/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: $

 Original author   : malone@drc.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <iostream>
#include <sstream>
#include <vector>

#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#include <vector>

#include "itkQuadEdgeMesh.h"
#include "itkVTKPolyDataReader.h"
#include "itkCellInterface.h"

#include <itkMersenneTwisterRandomVariateGenerator.h>

#include "vtkPolyDataWriter.h"
#include "vtkPolyData.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"

// Seems a little odd to use itk to read vtk polydata when we are going to process and write it back mainly with
// vtk, but the intent is that we may want to use itk mesh filters here at some point in the future.

void usage (void) {
  printf ("itk-sphere-connect inname outname [options]\n\n");
  printf ("\t-l connectfile\n\tconnectivity profile for labels (ASCII)\n\n");
  printf ("\t-N n\n\tTotal counts mode (interprets rows as connection out\n");
  printf ("\tprobabilities per area and normalises to get total\n\n");
  printf ("\t-c\n\tCount mode, rows are connection out counts for each\n");
  printf ("\tlabel. Can be used with -N to turn off scaling by area\n\n");
  printf ("\t-s i\n\tSet seed for random number generator\n\n");
  printf ("\t-r\n\tUse non-repeating seed (time based) for random number generator\n\n");
  printf ("\t-m f\n\tMinimum length, remove connections shorter than f after assignment\n\n");
  printf ("\t-v\n\tVerbose\n\n");
  return;
}

int main (int argc, char **argv) {
  const int dims = 3;

  typedef itk::QuadEdgeMesh<double, dims>   MeshT;
  typedef MeshT::PointIdentifier PointIDT;
  typedef MeshT::VectorType  VectorT;
  typedef MeshT::PointType   PointT;
  typedef std::vector<PointIDT> pointlistT;
  typedef std::vector<pointlistT> pointsbyregionT;

  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator MersenneTwister;

  vtkCellArray *connectionList = vtkCellArray::New();
  vtkIdType nextline[2];

  char *inname, *outname;
  char *connectfile = 0;
  FILE *fconnect;
  bool modeTotal = 0;
  bool modeCount = 0;
  unsigned modeTotalNumber = 0;
  char *optstring  = "crvm:s:l:N:";
  int  modeMersenneSeedNum = 0;
  bool modeMersenneSeed = false; // Only whether we've been given the arg
  bool modeRandSeed = false;
  double modeMinLengthNum = 0;
  bool   modeMinLength = false;
  bool   modeVerbose = false;
  int opt;
  bool argprob = 0;

  while ( (opt = getopt(argc, argv, optstring)) >= 0 ) {
    switch (opt) {
    case 'N' :
      modeTotal = true;
      if (sscanf(optarg,"%u",&modeTotalNumber) != 1){
	printf("-%c needs a number (positive integer) argument\n", opt);
	argprob = 1;
      }
      break;
    case 's' :
      modeMersenneSeed = true;
      if (sscanf(optarg,"%i",&modeMersenneSeedNum) != 1){
	printf("-%c needs a number (integer) argument\n", opt);
	argprob = 1;
      }
      break;
    case 'm' :
      modeMinLength = true;
      if (sscanf(optarg,"%lf",&modeMinLengthNum) != 1 || modeMinLengthNum < 0){
	printf("-%c needs number greater than zero\n", opt);
	argprob = 1;
      }
      break;
    case 'r' :
      modeRandSeed = true;
      break;
    case 'v' :
      modeVerbose = true;
      break;
    case 'l' :
      connectfile = optarg;
      break;
    case 'c' :
      modeCount = true;
      break;
    case ':' :
      argprob = 1;
      break;
    case '?' :
      argprob = 1;
      break;
    default :
      printf ("Error processing arguments\n");
      argprob = 1;
  }
  }

  if ( optind != (argc-2) ) {
    argprob = 1;
    printf ("Need input and output filename\n");
  } else {
    inname = argv[optind];
    outname = argv[optind+1];
  }

  if ( modeRandSeed && modeMersenneSeed ) {
    argprob = 1;
    printf ("Can't specify random (non-repeating) seed and explicit seed\n");
  }

  if ( argprob ) {
    usage();
    return 0;
  }
  
  if (connectfile) {
    fconnect = fopen(connectfile, "r");
    if ( ! fconnect ) {
      printf ("Couldn't read connections file %s\n", connectfile);
      return 0;
    }
  } else {
    fconnect = 0;
  }

  std::vector<std::vector<double> > connectivity;
  char *line = 0;
  size_t linesize = 0;
  while ( fconnect && getline(&line,&linesize,fconnect) >= 0 ) {
    std::stringstream lineparse(line);
    std::vector<double> row;
    double element;
    while ( lineparse >> element ) {
      row.push_back(element);
    }
    connectivity.push_back(row);
  }
  bool connectivityproblem = 0;
  if ( connectfile ) {
    if ( connectivity.size() == 0 ) {
      connectivityproblem = 1;
      printf ("Connectivity file empty\n");
    }
    for ( unsigned row=0 ;row<connectivity.size(); row++ ) {
      if ( connectivity[row].size() != connectivity.size() ) {
	printf ("Connectivity file not square matrix at row %u\n", row);
	connectivityproblem = 1;
      }
    }
  }
  if (connectivityproblem) {
    return 0;
  }
  if (connectfile) {
    printf ("Connection matrix %u x %u\n", connectivity.size(), connectivity[0].size());
  }


  typedef itk::VTKPolyDataReader<MeshT> ReaderT;

  ReaderT::Pointer inReader = ReaderT::New();
  try
    {
  inReader->SetFileName(inname);
    }
  catch( itk::ExceptionObject &err) {
        std::cerr << err << std::endl;
  }
  inReader->Update();

  MeshT::Pointer myMesh = inReader->GetOutput();
  PointT myPoint;
  int min = 0;
  int max = INT_MIN;


  pointsbyregionT pointsbyregion;

  for( unsigned ii=0; ii < myMesh->GetNumberOfPoints(); ii++ ) {
    int region;
    double regiondbl;
    myMesh->GetPointData(ii, &regiondbl); // Check value and do something
    if ( regiondbl > INT_MAX || regiondbl < INT_MIN ) {
      printf ("Encountered a region label out of range, stopping.\n");
      return 0;
    }
    region = (int) regiondbl;
    if ( region < 0 ) {
      printf("Minimum region label less than zero, stopping.\n");
      return 0;
    }
    else if ( region < min ) {
      min = region;
    }
    else if ( region > max ) {
      max = region;
      pointsbyregion.resize(max+1);
    }
    pointsbyregion[region].push_back(ii);
  }

  // Array index and region index start at zero.
  if ( ! connectfile ) {
    // Default testing mode is 20 connections between 1 & 2
    if (max < 2) {
      printf ("Need regions 1 & 2\n");
      return 0;
    }
    connectivity.resize(pointsbyregion.size());
    for ( unsigned row=0; row < pointsbyregion.size(); row++) {
      connectivity[row].resize(pointsbyregion.size());
    }
    connectivity[1][2] = connectivity[2][1] = 10;
  }

  // We intend to set things up so that # connections to create from each
  // region if the number stored in the row.
  if ( connectivity.size() != pointsbyregion.size() ) {
    connectivityproblem = 1;
    printf ("Number of rows/columns in %s doesn't\n",connectfile);
    printf ("correspond to number of labels in input\n");
  }
  double connectionscount = 0;
  double connectionsratio = 1;
  if ( modeTotal ) {
    // Start off by calculating as double, though we want to adjust to
    // get closest int.
    for ( unsigned row=0; row<connectivity.size(); row++) {
      double rowcount = 0;
      for ( unsigned col=0 ; col<connectivity[row].size() ; col++) {
	rowcount += connectivity[row][col];
      }
      if ( modeCount ) {
	connectionscount += rowcount;
      }
      else {
	connectionscount += rowcount * pointsbyregion[row].size();
      }
    }
    if ( connectionscount == 0 ) {
      printf ("No connections in matrix\n");
      return 0;
    }
    else {
      connectionsratio = modeTotalNumber / connectionscount;
    }
  }
  // If not modeTotal: For modeCount we need to change connections to integer
  // For probability mode, we consider row entries as connections per node,
  // need to re-cast as number of connections to make by multiplying by area.
  // If modeTotal then we multiply by connectionsratio already calculated,
  // with area adjustment if necessary.
  unsigned connectionscountint = 0;
  for ( unsigned row=0; row<connectivity.size(); row++) {
    for ( unsigned col=0 ; col<connectivity[row].size() ; col++) {
      double ratio = connectionsratio;
      if ( ! modeCount ) {
	ratio *= pointsbyregion[row].size();
      }
      connectivity[row][col] = (int) (connectivity[row][col] * ratio);
      connectionscountint += connectivity[row][col];
    }
  }
  if ( connectionscountint == 0 ) {
    printf ("Warning, calculated no connections\n");
  }

  MersenneTwister::Pointer myTwister = MersenneTwister::New();
  if ( modeRandSeed ) {
    myTwister->SetSeed();
  } else {
    myTwister->SetSeed((ITK_UINT32)modeMersenneSeedNum);
  }

  /*
  for ( int connect=0 ; connect < 20 ; connect++ ) {
    int orig_choice = myTwister->GetIntegerVariate(pointsbyregion[1].size()-1);
    int dest_choice = myTwister->GetIntegerVariate(pointsbyregion[2].size()-1);
    //int orig_choice = (int) ( pointsbyregion[1].size() * (rand() / (RAND_MAX + 1.0)));
    //int dest_choice = (int) ( pointsbyregion[2].size() * (rand() / (RAND_MAX + 1.0)));
    nextline[0] = pointsbyregion[1][orig_choice];
    nextline[1] = pointsbyregion[2][dest_choice];
    connectionList->InsertNextCell(2,nextline);
  }
  */
  for ( unsigned row=0 ; row < connectivity.size() ; row++ ) {
    for ( unsigned col=0 ; col < connectivity[row].size() ; col++ ) {
      for ( int count=(int)connectivity[row][col] ; count > 0 ; count-- ) {
	int orig_choice = 
	  myTwister->GetIntegerVariate(pointsbyregion[row].size()-1);
	int dest_choice =
	  myTwister->GetIntegerVariate(pointsbyregion[col].size()-1);
    //int orig_choice = (int) ( pointsbyregion[1].size() * (rand() / (RAND_MAX + 1.0)));
    //int dest_choice = (int) ( pointsbyregion[2].size() * (rand() / (RAND_MAX + 1.0)));
	nextline[0] = pointsbyregion[row][orig_choice];
	nextline[1] = pointsbyregion[col][dest_choice];
	PointT origPoint, destPoint;
	myMesh->GetPoint (nextline[0], &origPoint);
	myMesh->GetPoint (nextline[1], &destPoint);
	if ( ! modeMinLength ||
	     (origPoint.EuclideanDistanceTo(destPoint) >= modeMinLengthNum)) {
	  connectionList->InsertNextCell(2,nextline);
	}
	if (modeVerbose) printf ("o(,%f,%f,%f,),d(,%f,%f,%f,),l,%f\n",
	       origPoint[0],
	       origPoint[1],
	       origPoint[2],
	       destPoint[0],
	       destPoint[1],
	       destPoint[2],
	       origPoint.EuclideanDistanceTo(destPoint)
	       );
      }
    }
  }

  printf ("Assigned %d connections\n", connectionList->GetNumberOfCells());

  // Copy the mesh data over.
  vtkPoints *outputPoints = vtkPoints::New();
  outputPoints->SetDataTypeToFloat();
  outputPoints->Allocate(myMesh->GetNumberOfPoints());
  
  vtkFloatArray *outputLabels = vtkFloatArray::New();
  outputLabels->SetNumberOfComponents(1);
  outputLabels->SetNumberOfValues(myMesh->GetNumberOfPoints());

  vtkCellArray *outputTriangles = vtkCellArray::New();

  vtkIdType triangle[3];
  vtkIdType pointcount = 0;
  //itk::CellsContainer Pointer myMeshCells = myMesh->GetCells();

  for ( unsigned ii=0; ii<myMesh->GetNumberOfPoints(); ii++) {
    PointT myPoint;
    double regiondbl;
    double point[3];
    myMesh->GetPoint(ii, &myPoint);
    myMesh->GetPointData(ii, &regiondbl);
    for (unsigned jj=0; jj<3; jj++) {
      point[jj] = myPoint[jj];
    }
    outputPoints->InsertPoint((vtkIdType)ii, point);
    outputLabels->SetValue(ii,(int)regiondbl);
  }
  
  for ( unsigned ii=0; ii<myMesh->GetNumberOfCells(); ii++) {
    MeshT::CellAutoPointer thisCell;
    myMesh->GetCell(ii, thisCell);
    MeshT::PointIdentifier const *points = thisCell->GetPointIds();
    if ( thisCell->GetNumberOfPoints() != 3 ) {
      printf("Cell %d not a triangle\n", ii);
      continue;
    }
    for ( vtkIdType jj=0 ; jj<3 ; jj++){
      triangle[jj] = points[jj];
    }
    outputTriangles->InsertNextCell(thisCell->GetNumberOfPoints(), triangle);
  }

  vtkPolyData *outputPolyData = vtkPolyData::New();
  outputPolyData->SetPoints(outputPoints);
  outputPolyData->GetPointData()->SetScalars(outputLabels);
  outputPolyData->SetPolys(outputTriangles);
  outputPolyData->SetLines(connectionList);

  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(outname);
  writer->SetInput(outputPolyData);

  try
    {
      writer->Update();
      // myWriter->Update();
    }
  catch( itk::ExceptionObject &err) {
        std::cerr << err << std::endl;
  }

  return 0;
}
