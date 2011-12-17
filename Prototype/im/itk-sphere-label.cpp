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

#include "itkQuadEdgeMesh.h"
#include "itkQuadEdgeMeshScalarDataVTKPolyDataWriter.h"
#include "itkVTKPolyDataReader.h"

template <class VectorT>
class RegionSpec {
public:
  VectorT direction;
  float cut;
  int priority;
  int label;
};

void usage (void) {
  printf ("itk-sphere-label inname outname [-l file]\n\n");
  printf ("\t-l file\t csv file of label specs (N,theta,phi,dist,priority)\n\n");
  return;
}

int main (int argc, char **argv) {
  const int dims = 3;

  typedef itk::QuadEdgeMesh<float, dims>   MeshT;
  typedef MeshT::VectorType  VectorT;
  typedef MeshT::PointType   PointT;
  typedef RegionSpec<VectorT> RegionSpecT;
  typedef std::vector<RegionSpecT> regionlistT;

  char *inname, *outname, *labelfile = 0;
  FILE *flabel;
  char *optstring  = "l:";
  int opt;
  int argprob = 0;

  while ( (opt = getopt(argc, argv, optstring)) >= 0 ) {
    switch (opt) {
    case 'l' :
      labelfile = optarg;
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
  float phi, theta;
  PointT myPoint;
  float newval;

  for( unsigned int ii=0; ii < myMesh->GetNumberOfPoints(); ii++ ) {
    myMesh->GetPoint(ii, &myPoint);
    // for ( unsigned int jj=0; jj<dims; jj++ ) {
    //    myPoint[jj] -= myCentre[jj];
    //  }
    theta = atan2(myPoint[1], myPoint[0]); 
    phi= acos(myPoint[2]); 



    RegionSpecT thisRegion;
    thisRegion.label = 0;
    thisRegion.direction = myPoint.GetVectorFromOrigin();
    thisRegion.cut = 0;
    thisRegion.priority = 0;
    if ( ! labelfile ) {
      for (int jj=0; jj<dims ; jj++) {
	thisRegion.label |= (myPoint[jj] < 0) << jj ;
      }
    }

    for ( unsigned jj=0 ; jj < regionlist.size() ; jj++ ) {
      float labeldist = dot_product(myPoint.GetVnlVector(),
				    regionlist[jj].direction.GetVnlVector());
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
    myMesh->SetPointData(ii, newval);

  }


  typedef itk::QuadEdgeMeshScalarDataVTKPolyDataWriter< MeshT >   WriterT;
  WriterT::Pointer myWriter = WriterT::New();
  myWriter->SetInput( myMesh );
  myWriter->SetFileName( outname );
  try
    {
      myWriter->Update();
    }
  catch( itk::ExceptionObject &err) {
        std::cerr << err << std::endl;
  }

  return 0;
}
