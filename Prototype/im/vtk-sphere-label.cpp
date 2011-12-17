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
#include <vector>

#include <unistd.h>

#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"


template <class VectorT>
class RegionSpec {
public:
  VectorT direction;
  float cut;
  int priority;
  int label;

  RegionSpec (VectorT init, unsigned len) {
    for (unsigned ii=0; ii<len; ii++) {
      direction[ii] = init[ii];
    }
  }
  RegionSpec (){};

};


double dot_product (double a[3], double b[3]) {
  double sum=0;
  for (unsigned ii=0; ii<3 ; ii++ ) {
    sum += a[ii] * b[ii];
  }
  return sum;
}

void usage (void) {
  printf ("itk-sphere-label inname outname [-l file]\n\n");
  printf ("\t-l file\t csv file of label specs (N,theta,phi,dist,priority)\n\n");
  printf ("\t-t theta shift (radians)\n\n");
  printf ("\t-p phi shift (radians)\n\n");
  return;
}

int main (int argc, char **argv) {
  const int dims = 3;

  typedef double VectorT[3];
  typedef double PointT[3];
  typedef RegionSpec<VectorT> RegionSpecT;
  typedef std::vector<RegionSpecT> regionlistT;

  char *inname, *outname, *labelfile = 0;
  FILE *flabel;
  char *optstring  = "l:t:p:";
  int opt;
  int argprob = 0;
  double shift_phi=0, shift_theta=0;

  while ( (opt = getopt(argc, argv, optstring)) >= 0 ) {
    switch (opt) {
    case 'l' :
      labelfile = optarg;
      break;
    case 'p' :
      if ( sscanf(optarg,"%lf",&shift_phi) != 1 ) {
	printf ("-%c requires a floating point arugment\n", opt);
	argprob = 1;
      }
      break;
    case 't' :
      if ( sscanf(optarg,"%lf",&shift_theta) != 1 ) {
	printf ("-%c requires a floating point arugment\n", opt);
	argprob = 1;
      }
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

  if ( argprob ) {
    usage();
    return 0;
  }
  
  if (labelfile) {
    flabel = fopen(labelfile, "r");
    if ( ! flabel ) {
      printf ("Couldn't read label file %s\n", labelfile);
      return 0;
    }
  } else {
    flabel = 0;
  }

  regionlistT regionlist;
  char *line = 0;
  size_t linesize = 0;
  while ( flabel && getline(&line,&linesize,flabel) >= 0 ) {
    int label, priority;
    float theta, phi, cut;
    char firstchar;
    if ( sscanf(line, "%d,%f,%f,%f,%d", &label, &theta, &phi, &cut, &priority)
	 == 5 ) {
      RegionSpecT region;
      // should implement some value checking.
      theta += shift_theta;
      phi   += shift_phi;

      region.direction[0] = sin(theta)*cos(phi);
      region.direction[1] = -sin(theta)*sin(phi);
      region.direction[2] = cos(theta);
      region.cut = cut;
      region.priority = priority;
      region.label = label;
      regionlist.push_back(region);
      printf ("Read spec %zd\n", regionlist.size());
      printf ("%d,%f,%f,%f,%d\n",region.label,theta,phi,
	      region.cut,region.priority);
    }
    else if ( sscanf(line, " %c", &firstchar) == 0 || firstchar == '#' ) {
      //Is a comment
      continue;
    } else {
      printf ("Error encountered while processing label file %s\n", labelfile);
      argprob = 1;
      break;
    }
  }


  vtkPolyDataReader *inReader = vtkPolyDataReader::New();

  //vtk doesn't do exceptions
  //try
  //  {
  inReader->SetFileName(inname);
  //  }
  //catch( itk::ExceptionObject &err) {
  //      std::cerr << err << std::endl;
  //}
  inReader->Update();

  vtkPolyData *myMesh = inReader->GetOutput();

  vtkIntArray *outputLabels = vtkIntArray::New();
  vtkPoints *myMeshPoints = myMesh->GetPoints();
  outputLabels->SetNumberOfComponents(1);
  outputLabels->SetNumberOfValues(myMeshPoints->GetNumberOfPoints());

  float phi, theta;
  PointT myPoint;
  int newval;

  for( int ii=0; ii < myMeshPoints->GetNumberOfPoints(); ii++ ) {
    myMeshPoints->GetPoint(ii, myPoint);
    // for ( unsigned int jj=0; jj<dims; jj++ ) {
    //    myPoint[jj] -= myCentre[jj];
    //  }
    theta = atan2(myPoint[1], myPoint[0]); 
    phi= acos(myPoint[2]); 



    RegionSpecT thisRegion(myPoint, 3);
    thisRegion.label = 0;
    thisRegion.cut = 0;
    thisRegion.priority = 0;
    if ( ! labelfile ) {
      for (int jj=0; jj<dims ; jj++) {
	thisRegion.label |= (myPoint[jj] < 0) << jj ;
      }
    }

    for ( unsigned jj=0 ; jj < regionlist.size() ; jj++ ) {
      float labeldist = dot_product(myPoint,
				    regionlist[jj].direction);
      if ( labeldist > regionlist[jj].cut &&
	   (regionlist[jj].priority > thisRegion.priority ||
	    	    (regionlist[jj].priority == thisRegion.priority
	    	     && labeldist > thisRegion.cut )
	    )
	   ) {
	thisRegion.label = regionlist[jj].label;
	thisRegion.cut = labeldist;
	thisRegion.priority = regionlist[jj].priority;
      }
    }

    newval=thisRegion.label;
    outputLabels->SetValue(ii, newval);
  }
  myMesh->GetPointData()->SetScalars(outputLabels);


  vtkPolyDataWriter *myWriter = vtkPolyDataWriter::New();
  myWriter->SetInput( myMesh );
  myWriter->SetFileName( outname );
  // try
  //  {
      myWriter->Update();
  //  }
  //catch( itk::ExceptionObject &err) {
  //      std::cerr << err << std::endl;
  //}

  return 0;
}
