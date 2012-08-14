/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-11-29 10:30:45 +0000 (Mon, 29 Nov 2010) $
 Revision          : $Revision: 4230 $
 Last modified by  : $Author: im $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

/*=============================================================================
NB needs a fixed itkNonUniformBSpline to work correctly
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
#include "vtkSmartPointer.h"

#include "vnl/vnl_matrix.h"
#include "vnl/vnl_cross.h"

#include "itkNonUniformBSpline.h"

#define LEFT1ST 1
#define RIGHT1ST 2

//#define FACECONTROLRATIO 30
#define FACECONTROLRATIO 15
//#define INITSTEP 5
#define INITSTEP 2

double lowerstepsize = 0.05; // empirically, 0.09 seems to be the cutoff below
                             // which updates run down to zero.

bool freesurfercheat = 0;
bool peredgecost=0;
bool repaintateachupdate = true;
const int dropcyclicsize = 14; // Drop hexagonal cycles.
bool edgescoremode = false;
bool pprofilemode = false; //alternative is ssd point cost

typedef itk::NonUniformBSpline<3> NonUniformBSpline;

void Usage(char *exec)
  {
    ucltk::itkLogHelper::PrintCommandLineHeader(std::cout);
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

struct labelPairBoundaryT {
  std::set <vtkIdType> edgeFaces;
  std::set <vtkIdType> junctionFaces;
} ;

class structuredPairBoundaryT {
  // Left and right, what's going on? When we create the boundary we don't know
  // which direction we're going so don't decide left-right. The labels
  // variable holds the labels in lowest first order.
  // Later when we track the boundary and work out the order we can decide
  // which is left and which is right, this gets stored in direction.
  // Finally, when the edge list is built it's done using directed traversal,
  // so edges are always left-right (i.e. left edge[X][0], right edges[X][1]).
public:
  enum boundaryT { cyclic, noncyclic };
  boundaryT boundaryType;
  std::vector <vtkIdType> faceList; // For non-cyclic, start and end are
                                    // junctions.
  std::vector<vtkIdType> junctions; // Store indices to junction map
                                    // for open splines
  std::vector<vtkIdType> labels;    // Labels in boundary
  int direction; // 0 unset, LEFT1ST 1st label on left,
                 // RIGHT1ST 2nd label on left
  NonUniformBSpline::Pointer spline;

  std::vector<std::vector<vtkIdType> > edges; // Currently only populated after
                                              // updating faces from splines
  std::vector<double> tList; // Currently only populated after
                             // updating faces from splines

  structuredPairBoundaryT();
};

structuredPairBoundaryT::structuredPairBoundaryT(){
  junctions.resize(2);
  labels.resize(2);
}

typedef std::pair<vtkIdType,vtkIdType> labelpairT;

typedef std::vector <structuredPairBoundaryT> boundList;
typedef std::map < labelpairT, labelPairBoundaryT > unsortedBoundariesT;


class sortedBoundariesT {
public:
  boundList structuredBoundaries;
  // Index junctions to boundaries, id by INITIAL face point (they can
  // move and we may add more, but doing this lets us build the list
  // directly at the start.
  typedef std::map <vtkIdType, std::vector<int> > junctionListT;
  junctionListT junctionList;

  void addEdge(std::vector<int>, vtkPolyData *,
	       structuredPairBoundaryT::boundaryT,
	       vtkIdType=0, vtkIdType=0);
  typedef boundList::iterator iterator;
  boundList::iterator begin();
  boundList::iterator end();
};

boundList::iterator sortedBoundariesT::begin()
{return structuredBoundaries.begin();}

boundList::iterator sortedBoundariesT::end()
{return structuredBoundaries.end();}


/*
const unsigned VectorDim=3;
typedef itk::Vector < double , VectorDim > VectorType ;
typedef itk::Image < VectorType , 1 > SplineCurveImageType ;
typedef itk::PointSet < VectorType , 1 > SplineCurvePointSetType ;
typedef itk::BSplineScatteredDataPointSetToImageFilter
    < SplineCurvePointSetType , SplineCurveImageType > SplineFilterType ;
*/
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
    return std::exp(- (position-mean)*(position-mean)/(2.0*variance) ) / std::sqrt ( 2 * M_PI * variance ) ;
  }



bool pointInsideTri ( double pointX[3], vtkPolyData *polyData,
		      vtkIdType tripoint[3] ) {
  bool result = 1;
  const unsigned npoints=3;
  const unsigned dim=3;
  vnl_matrix<double> triangle(npoints,dim); // Points per row
  for ( unsigned point=0; point <npoints; point++ ) {
    double thispoint[3];
      polyData->GetPoint(tripoint[point],thispoint);
      triangle.set_row(point,thispoint);
  }

  // Need > 0 for : r.a*b, r.b*c, r.c*a, so r stays in first row.
  vnl_matrix<double> tripleprod(dim,dim);
  tripleprod.set_row(0, pointX);

  // We'll could do something sensible if given differing numbers of
  // points and dimensions (not the three dimension triangles case),
  // though these are specified const within this function
  for (unsigned permute=0 ; result && permute<npoints; permute++ ) {
    tripleprod.set_row(1, triangle[permute]);
    tripleprod.set_row(2, triangle[(permute+1)%npoints]);
    result &= vnl_determinant(tripleprod) > 0;
  }
  return result;
}


// Which point is on the left crossing from faceA to faceB if 'up'
// is away from the origin?
// If A and B are the non-edge pts, and C and D are the edge points then
// (B-A).DxC > 0 when D is left, i.e. (B-A).LxR>0
std::vector<vtkIdType> getEdgePointLeftRight ( vtkIdType faceA,
					       vtkIdType faceB,
					       vtkPolyData *polyData ) {
  std::vector<vtkIdType> result(2, -1);
  vtkIdType numberOfPointsInA;
  vtkIdType numberOfPointsInB;
  vtkIdType *ptsA, *ptsB, edge[2], direct[2];
  polyData->GetCellPoints(faceA, numberOfPointsInA, ptsA);
  polyData->GetCellPoints(faceB, numberOfPointsInB, ptsB);

  unsigned bApex[] = { 0, 0, 1, 0, 2};
  unsigned bApexMas = 1 | 1<<1 | 1<<2;

  if ( numberOfPointsInA != 3 ) {
    std::cerr << "getEdgePointLeftRight on non-triangular cell #" << faceA << " " << numberOfPointsInA << "pts" << std::endl;
    return result;
  }
  if ( numberOfPointsInB != 3 ) {
    std::cerr << "getEdgePointLeftRight on non-triangular cell #" << faceB  << faceA << " " << numberOfPointsInB << "pts" << std::endl;
    return result;
  }

  if (faceA==faceB) {
    std::cerr << "getEdgePointLeftRight on faceA=faceB " << faceA << std::endl;
  }

  //for (vtkIdType ii=0;ii<3;ii++) std::cout<< ptsA[ii] << " " ;
  //std::cout << ", ";
  //for (vtkIdType ii=0;ii<3;ii++) std::cout<< ptsB[ii] << " " ;
  //std::cout << std::endl;


  unsigned edgecount=0;
  for (vtkIdType iiA=0; iiA<numberOfPointsInA; iiA++) {
    bool edgematch=false;
    for (vtkIdType iiB=0; iiB<numberOfPointsInB; iiB++) {
      if (ptsA[iiA]==ptsB[iiB]){
	edgematch=true;
	edge[edgecount]=ptsA[iiA];
	edgecount++;
	bApexMas ^= 1 << iiB;
      }
      if (edgecount>2) {
	std::cerr << "getEdgePointLeftRight more than two edge points for cell #s" << faceA << " " << faceB << std::endl;
    return result;
      }
    }
    if (!edgematch) {
      direct[0]=ptsA[iiA];
    }
  }

  if(!edgecount) {
    std::cerr << "getEdgePointLeftRight no common edge for cell #s" << faceA << " " << faceB << std::endl;
    return result;
  }

  //std::cout << "bApexMas " << bApexMas;
  //std::cout << " bApex " << bApex[bApexMas] << std::endl;

  direct[1]=ptsB[bApex[bApexMas]];

  const unsigned dim=3;
  vnl_vector<double> B(dim);
  vnl_vector<double> A(dim);
  vnl_vector<double> AB(dim);
  vnl_vector<double> edge0(dim);
  vnl_vector<double> edge1(dim);
  vnl_matrix<double> tripleprod(dim,dim);

  double point[3];
  polyData->GetPoint(direct[1],point);
  B.set(point);
  polyData->GetPoint(direct[0],point);
  A.set(point);
  AB=B-A;
  tripleprod.set_row(0,AB);

  vnl_vector<double> centre(dim);
  centre.fill(0);
  if (freesurfercheat) {
    // Need to know which way the interior of the sphere is to define a surface
    // we can have left and right on.
    // For 0 centred sphere we can just use the edge vector.
    // Better fix, if we can assume convex at all junctions, is to get vector
    // from (A+B)/2 to (edge0+edge1)/2 which will not be normal to surface,
    // but is out of the mesh locally rather than into it. Could re-factor
    // the calculation into the tetrahedron defined by those edges.
    // Seems this should be: LxR.(B-A)+AxB.(R-L) > 0
    if (A.get(0) > 0 ) {
      //Centre at 100,100,100
      centre.fill(100);
    } else {
      //Centre at -100,-100,-100
      centre.fill(-100);
    }
  }
  polyData->GetPoint(edge[0],point);
  edge0.set(point);
  edge0 = edge0 - centre;
  tripleprod.set_row(1,edge0);

  polyData->GetPoint(edge[1],point);
  edge1.set(point);
  edge1 = edge1 - centre;
  tripleprod.set_row(2,edge1);

  if (vnl_determinant(tripleprod) > 0 ) {
    result[0] = edge[0]; //Left 1st
    result[1] = edge[1]; //Right
  } else {
    result[0] = edge[1]; // Right first
    result[1] = edge[0]; 
  }
  return result;
}


vtkIdType getEdgePointLeft ( vtkIdType faceA, vtkIdType faceB,
				     vtkPolyData *polyData ) {
  std::vector<vtkIdType> leftright =
    getEdgePointLeftRight (faceA, faceB, polyData );
  return leftright[0];
}



bool pointInsideMeshTri ( double pointX[3], vtkPolyData *polyData,
		      vtkIdType tripoint[3] ) {
  // A more complex approach that doesn't implicitly use the vector from the
  // origin as normal. Not the full normals-at edges approach though.
  // return  pointInsideTri ( pointX, polyData, tripoint );
  bool result = 1;
  const unsigned npoints=3;
  const unsigned dim=3;
  vnl_matrix<double> triangle(npoints,dim); // Points per row
  for ( unsigned point=0; point <npoints; point++ ) {
    double thispoint[3];
      polyData->GetPoint(tripoint[point],thispoint);
      triangle.set_row(point,thispoint);
  }

  vnl_vector<double> norm(dim);
  norm.fill(0);
  for (unsigned permute=0; permute<npoints; permute++) {
    norm += vnl_cross_3d(triangle.get_row(permute),
			 triangle.get_row((permute+1)%npoints));
  }

  // Need > 0 for : n*e1.(p-t1), n*e2.(p-t2), n*e3.(p-t3),
  // so n stays in first row.
  vnl_matrix<double> tripleprod(dim,dim);
  tripleprod.set_row(0, norm);

  // We could do something sensible if given differing numbers of
  // points and dimensions (not the three dimension triangles case),
  // though these are specified const within this function.
  // Actually in this function, unlike the simpler version, this is sensible
  // as it's the prism interior calculation (although calculation of the
  // normal is probably wrong or incomplete for that case.
  for (unsigned permute=0 ;  permute<npoints; permute++ ) {
    tripleprod.set_row(1, triangle[(permute+1)%npoints] - triangle[permute]);
    tripleprod.set_row(2, pointX - triangle[permute]);
    result &= vnl_determinant(tripleprod) > 0;
  }
  return result;
}


bool pointInsideMeshFaceEdges ( double pointX[3], vtkPolyData *polyData,
			       vtkIdType face ) {
  // A more complex approach that doesn't implicitly use the vector from the
  // origin as normal. Calculate using average normals at edges.

  bool result = 1;
  const unsigned npoints=3;
  const unsigned dim=3;
  vnl_matrix<double> triangle(npoints,dim); // Points per row
  vtkIdList *edgeneighbours;

  vtkIdType *tripoint;
  vtkIdType pointcount;

  polyData->GetCellPoints( face, pointcount, tripoint);
  if ( polyData->GetCellType(face) != VTK_TRIANGLE ) {
    std::cerr << "Face " << face << " not a triangle" << std::endl;
    return false;
  }
  //return pointInsideTri(pointX,polyData, tripoint);
  for ( unsigned point=0; point <npoints; point++ ) {
    double thispoint[3];
    polyData->GetPoint(tripoint[point],thispoint);
    triangle.set_row(point,thispoint);
  }

  switch (face) {
  case 198028:
  case 198030:
  case 198031:
  case 198046 :
    //std::cout << triangle << std::endl;
    break;
  default:
    break;
  }
  vnl_vector<double> PX(3);
  PX.set(pointX);

  vnl_vector<double> norm(dim);
  norm.fill(0);
  for (unsigned permute=0; permute<npoints; permute++) {
    norm += vnl_cross_3d(triangle.get_row(permute),
			 triangle.get_row((permute+1)%npoints));
  }

  // Need > 0 for : n1*e1.(p-t1), n2*e2.(p-t2), n3*e3.(p-t3),
  // so n stays in first row.
  for (unsigned permute=0 ;  permute<npoints; permute++ ) {
    edgeneighbours = vtkIdList::New();
    polyData->GetCellEdgeNeighbors (face, tripoint[permute],
				    tripoint[(permute+1)%npoints],
				    edgeneighbours);
    std::vector <vtkIdType> trineighbours;
    for (unsigned ii=0; ii < edgeneighbours->GetNumberOfIds(); ii++) {
      vtkIdType neighbour = edgeneighbours->GetId(ii);
      if ( polyData->GetCellType(neighbour) == VTK_TRIANGLE ) {
	trineighbours.push_back(neighbour);
      }
    }
    if ( trineighbours.size() != 1 ) {
      std::cerr << "Face " << face << " had " << trineighbours.size()
		<< " neighbours on edge " << permute << std::endl;
      result=0;
    }
    vtkIdType otherpoint=-1;
    vtkIdType *neighbourpoints;
    vtkIdType neighbourpointcount;
    if (result) {
      polyData->GetCellPoints( trineighbours[0],
			       neighbourpointcount, neighbourpoints);
      for (unsigned ii=0; ii<neighbourpointcount; ii++) {
	if ( neighbourpoints[ii] != tripoint[permute] &&
	     neighbourpoints[ii] != tripoint[(permute+1)%npoints]) {
	  otherpoint=neighbourpoints[ii];
	  break;
	}
      }
      if (otherpoint<0){
	std::cerr << "Didn't find a point for face " << trineighbours[0]
		  << " that wasn't on edge with " << face << std::endl;
      }
    }
    vnl_vector<double> thisnorm(dim);
    thisnorm.fill(0);
    if (result) {
      vnl_matrix<double> neighbourtri(dim,neighbourpointcount);
      double thispoint[3];
      // Opposite direction along edge to originating tri; keep norm in
      // same sense.
      polyData->GetPoint(tripoint[(permute+1)%npoints],
			 thispoint);
      neighbourtri.set_row(0,thispoint);
      polyData->GetPoint(tripoint[permute],
			 thispoint);
      neighbourtri.set_row(1,thispoint);
      polyData->GetPoint(otherpoint,
			 thispoint);
      neighbourtri.set_row(2,thispoint);

      for (unsigned neighpermute=0; neighpermute<3; neighpermute++) {
	thisnorm += vnl_cross_3d(triangle.get_row(neighpermute),
			     triangle.get_row((neighpermute+1)%3));
      }
      thisnorm = (thisnorm + norm)/2;

      // Cheat. freesurfercheat flag just hacks the centre.
      vnl_vector<double> centre(dim);
      centre.fill(0);
      if (freesurfercheat) {
	// Failed on a boundary traversal last face ...
	// occurs when we lose the next face during traversal and the mesh
	// normal approach is less stable, so this basically drops it and
	// uses the sphere normal instead. A fuller fix would be to have the
	// spline traversal algorithm return to closest-face search whenever
	// it looses the next point, though the problem of traversing near
	// points (esp. narrow ones) remains.
	if (triangle.get(0,0) > 0 ) {
	  //Centre at 100,100,100
	  centre.fill(100);
	  thisnorm = PX - centre;
	} else {
	  //Centre at -100,-100,-100
	  centre.fill(-100);
	  thisnorm = PX - centre;
	}
      } else {
	thisnorm = PX;
      }
    }
    if (result) {
      vnl_matrix<double> tripleprod(dim,dim);
      tripleprod.set_row(0, thisnorm);
      tripleprod.set_row(1, (triangle.get_row((permute+1)%npoints) -
			     triangle.get_row(permute)));
      tripleprod.set_row(2, PX - triangle.get_row(permute));
      result &= vnl_determinant(tripleprod) > 0;
    }
    edgeneighbours->Delete();
  }
  return result;
}



vtkIdType labelOnEdgeLeft (vtkIdType faceA, vtkIdType faceB,
			   vtkPolyData *polyData) {
  vtkIdType result = -1;
  vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
  vtkIdType leftpt = getEdgePointLeft(faceA, faceB, polyData);
  if (leftpt >= 0) {
    result = (int)scalarData->GetTuple1(leftpt);
  }
  return result;
}



vtkIdList* getTriCellNeighboursFromData (
    vtkPolyData *polyData,
    vtkIdType thisFace
    )
  {
    vtkIdList *collectNeighbours = vtkIdList::New();
    vtkIdList *edgeneighbours    = vtkIdList::New();
    vtkIdType *listOfPointIds;
    vtkIdType numberOfPointsInCurrentCell;
    polyData->GetCellPoints(thisFace,
			    numberOfPointsInCurrentCell, listOfPointIds);
    for ( vtkIdType iipoint=0; iipoint<numberOfPointsInCurrentCell;
	  iipoint++) {
      edgeneighbours->Reset();
      polyData->GetCellEdgeNeighbors (thisFace, listOfPointIds[iipoint],
        listOfPointIds[(iipoint+1)%numberOfPointsInCurrentCell],
	edgeneighbours);
      for ( vtkIdType iineighbour=0;
        iineighbour<edgeneighbours->GetNumberOfIds();
	    iineighbour++) {
	vtkIdType neighbourId = edgeneighbours->GetId(iineighbour);
	if (VTK_TRIANGLE != polyData->GetCellType(neighbourId)) continue;
	collectNeighbours->InsertUniqueId(neighbourId);
      }
    }
    edgeneighbours->Delete();
    return collectNeighbours;
}

// Cheap cheat, have a global list of neighbours we work on. With more time
// this should be incorporated into the normal data flow.
std::vector< std::vector<vtkIdType> > faceNeighboursList;
void buildFaceNeighboursList (
			      vtkPolyData *polyData
			      ){
  //  polyData->BuildLinks(); // ary
  vtkIdType totalpoints = polyData->GetNumberOfCells();
  faceNeighboursList.resize(totalpoints);
  for (unsigned face=0; face<totalpoints; face++ ) {
    if (VTK_TRIANGLE != polyData->GetCellType(face)) {
      continue;
    }
    vtkIdList* neighbours = getTriCellNeighboursFromData (polyData, face);
    faceNeighboursList[face].resize(3,-1);
    if ( neighbours->GetNumberOfIds() !=3 ) {
      std::cerr << "Error building neighbours list at face "
		<< face << " " << neighbours->GetNumberOfIds() << std::endl;
    } else {
      for (unsigned neigh=0; neigh<3; neigh++) {
	faceNeighboursList[face][neigh] = neighbours->GetId(neigh);
      }
    }
    if (neighbours) {
      neighbours->Delete();
    }
  }
  return;
}

// Lazy implementation on faceNeighboursList
vtkIdList* getTriCellNeighbours (
    vtkPolyData *polyData,
    vtkIdType thisFace
    )
  {
    vtkIdList *collectNeighbours = vtkIdList::New();
    for (unsigned neigh=0; faceNeighboursList.size() > thisFace &&
	   neigh<faceNeighboursList[thisFace].size();
	 neigh++) {
      collectNeighbours->InsertNextId(faceNeighboursList[thisFace][neigh]);
    }    
    return collectNeighbours;
  }


// Lazy implementation on faceNeighboursList
vtkIdList *getTriCellCrossBoundaryNeighbours (
    vtkPolyData *polyData,
    vtkIdType thisFace
    )
  {
    vtkIdList *collectNeighbours = vtkIdList::New();
    vtkIdType *listOfPointIds;
    vtkIdType numberOfPointsInCurrentCell;
    vtkDataArray *labels = polyData->GetPointData()->GetScalars();
    polyData->GetCellPoints(thisFace,
			    numberOfPointsInCurrentCell, listOfPointIds);
    for (unsigned neigh=0; faceNeighboursList.size() > thisFace
	   && neigh<faceNeighboursList[thisFace].size();
	 neigh++) {
      vtkIdType neighbourId = faceNeighboursList[thisFace][neigh];
      std::vector<vtkIdType> edgePts;
      edgePts = getEdgePointLeftRight ( thisFace, neighbourId,
					polyData );
      if ( labels->GetTuple1(edgePts[0]) !=
	   labels->GetTuple1(edgePts[1]) ) {
	collectNeighbours->InsertNextId(faceNeighboursList[thisFace][neigh]);
      }
    }  
    return collectNeighbours;
  }


/*
vtkIdList* getTriCellNeighbours (
    vtkPolyData *polyData,
    vtkIdType thisFace
    )
  {
    vtkIdList *collectNeighbours = vtkIdList::New();
    vtkIdList *edgeneighbours    = vtkIdList::New();
    vtkIdType *listOfPointIds;
    vtkIdType numberOfPointsInCurrentCell;
    polyData->GetCellPoints(thisFace,
			    numberOfPointsInCurrentCell, listOfPointIds);
    for ( vtkIdType iipoint=0; iipoint<numberOfPointsInCurrentCell;
	  iipoint++) {
      edgeneighbours->Reset();
      polyData->GetCellEdgeNeighbors (thisFace, listOfPointIds[iipoint],
        listOfPointIds[(iipoint+1)%numberOfPointsInCurrentCell],
	edgeneighbours);
      for ( vtkIdType iineighbour=0;
        iineighbour<edgeneighbours->GetNumberOfIds();
	    iineighbour++) {
	vtkIdType neighbourId = edgeneighbours->GetId(iineighbour);
	if (VTK_TRIANGLE != polyData->GetCellType(neighbourId)) continue;
	collectNeighbours->InsertUniqueId(neighbourId);
      }
    }
    edgeneighbours->Delete();
    return collectNeighbours;
}
*/

/*
vtkIdList *getTriCellCrossBoundaryNeighbours (
    vtkPolyData *polyData,
    vtkIdType thisFace
    )
  {
    vtkIdList *collectNeighbours = vtkIdList::New();
    vtkIdList *edgeneighbours    = vtkIdList::New();
    vtkIdType *listOfPointIds;
    vtkIdType numberOfPointsInCurrentCell;
    vtkDataArray *labels = polyData->GetPointData()->GetScalars();
    polyData->GetCellPoints(thisFace,
			    numberOfPointsInCurrentCell, listOfPointIds);
    for ( vtkIdType iipoint=0; iipoint<numberOfPointsInCurrentCell;
	  iipoint++) {
      edgeneighbours->Reset();
      if ( labels->GetTuple1(listOfPointIds[iipoint]) !=
	   labels->GetTuple1(listOfPointIds[
             (iipoint+1)%numberOfPointsInCurrentCell]) ) {
	vtkIdType edgep1 = listOfPointIds[iipoint];
	vtkIdType edgep2 = listOfPointIds[(iipoint+1)%numberOfPointsInCurrentCell];
	polyData->GetCellEdgeNeighbors (thisFace, edgep1, edgep2,
					edgeneighbours);
        for ( vtkIdType iineighbour=0;
          iineighbour<edgeneighbours->GetNumberOfIds();
          iineighbour++) {
            vtkIdType neighbourId = edgeneighbours->GetId(iineighbour);
	    if (VTK_TRIANGLE != polyData->GetCellType(neighbourId)) continue;
	    collectNeighbours->InsertUniqueId(neighbourId);
        }
      }
    }
    edgeneighbours->Delete();
    return collectNeighbours;
  }
*/

// GetConnectedVertices from vtk Mailing list, chopped down slightly.
void getConnectedVertices(vtkPolyData *mesh, vtkIdType seed,
			  vtkSmartPointer<vtkIdList> connectedVertices)
{

  //get all cells that vertex 'seed' is a part of
  vtkIdType *cellIdList;
  short unsigned cellcount;
  if ( mesh->GetNumberOfPoints()<=seed ) {
    std::cerr
      << "getConnectedVertices called on seed above number of points\n"
      << std::endl;
    connectedVertices=0;
  }
  mesh->GetPointCells(seed, cellcount, cellIdList);

  for(vtkIdType ii = 0; ii < cellcount; ii++)
    {
      vtkCell *cell;
      cell = mesh->GetCell(cellIdList[ii]);

      // modified
      if(cell->GetNumberOfEdges() <= 0 ||
	 cell->GetCellType() != VTK_TRIANGLE )
	{
	  continue;
	}

    for(vtkIdType ee = 0; ee < cell->GetNumberOfEdges(); ee++)
      {
      vtkCell* edge = cell->GetEdge(ee);
      vtkIdList* pointIdList = edge->GetPointIds();
      if(pointIdList->GetId(0) == seed || pointIdList->GetId(1) == seed)
        {
        if(pointIdList->GetId(0) == seed)
          {
          connectedVertices->InsertNextId(pointIdList->GetId(1));
          }
        else
          {
          connectedVertices->InsertNextId(pointIdList->GetId(0));
          }
        }
      }
    }
  return;
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

double computeSSDOfArray(unsigned int size, std::vector<double> a,
			 std::vector<double> b)
  {
    double result = 0;
    
    for (unsigned int i = 0; i < size; i++)
      {
        result += ((a[i]-b[i])*(a[i]-b[i]));
      }
    
    return result;
  }


double computeProfilePScore(unsigned int size, std::vector<double> profile,
		     std::vector<double> testprofile,
		     std::vector<double> altprofile)
  {
    double result = 0;
    
    for (unsigned int ii = 0; ii < size; ii++)
      {
	double delta = altprofile[ii] - testprofile[ii];
	double mean = (testprofile[ii] + altprofile[ii])/2;
	double score=0;
	if ( (profile[ii] - mean) > 0 ) {
	  // profile above mean, then low score (better) if test > alt
	  // high if alt>test
	  score = delta;
	}
	else if ((profile[ii] - mean) < 0 ) {
	  // profile below mean, then low score (better) if test > alt
	  // high if alt>test
	  score = -delta;
	}
        result += score;
      }
    
    return result;
  }

double computeProfileSSDScore(unsigned int size, std::vector<double> profile,
			      std::vector<double> testprofile,
			      std::vector<double> altprofile)
  {
    // So we can swap back to SSD method easily. Ignores the altprofile.
    return computeSSDOfArray(size, profile, testprofile);
  }


void
sortedBoundariesT::addEdge(std::vector<int> buildEdge,
			   vtkPolyData *polyData,
			   structuredPairBoundaryT::boundaryT cyclic,
			   vtkIdType label1, vtkIdType label2) {
  structuredPairBoundaryT thisBound;
  thisBound.boundaryType = cyclic;
  thisBound.faceList =  buildEdge;
  vtkIdType leftLabel =  labelOnEdgeLeft (buildEdge[0],buildEdge[1],polyData);
  if ( leftLabel==label1 ) {
    thisBound.direction=LEFT1ST;
  }
  else if ( leftLabel==label2 ) {
    thisBound.direction=RIGHT1ST;
  } else {
    std::cerr << "Error determining left label for " << label1
	      << ":" << label2 << " got " << leftLabel << std::endl;
  }
  thisBound.labels[0]=label1;
  thisBound.labels[1]=label2;
  if (cyclic == structuredPairBoundaryT::noncyclic) {
    thisBound.junctions[0]= *buildEdge.begin();
    thisBound.junctions[1]= *buildEdge.rbegin();
    unsigned nextindex=structuredBoundaries.size();
    junctionList[thisBound.junctions[0]].push_back(nextindex);
    junctionList[thisBound.junctions[1]].push_back(nextindex);
  }
  structuredBoundaries.push_back(thisBound);
  return;
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
	      // These if clauses are the only way to deal with the fact
	      // we're not properly checking the image array size against
	      // the number of labels in the connection data, start of
	      // main needs some tidy-up to handle it better (and allow
	      // different numbers for ease of testing)
	      unsigned index=label2*xSize + label1;
	      if (index<arraySize) testLabelImageArray[index] += connectweight;
	      index=label1*xSize + label2;
	      if (index<arraySize) testLabelImageArray[index] += connectweight;
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



void cleanLabels(vtkPolyData *polyData)
  {
    vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
    vtkPoints *points = polyData->GetPoints();
    vtkIdType numberOfPoints = polyData->GetNumberOfPoints();

    std::cout << "Clean labels" << std::endl;
    // For each point, test neighbours to see if it is an isolated label,
    // if it is then take a vote on what it should be. In ties go for the
    // lowest number (arbitrary, but otherwise we get into curvature and things
    vtkSmartPointer<vtkIdList> neighbours = 
      vtkSmartPointer<vtkIdList>::New();
    for (vtkIdType point=0 ; point<numberOfPoints; point++) {
      vtkIdType thisLabel = scalarData->GetTuple1(point);
      neighbours->Reset();
      getConnectedVertices(polyData, point, neighbours);
      std::map<vtkIdType,unsigned> labels;
      for (unsigned n_neigh=0; n_neigh < neighbours->GetNumberOfIds();
	   n_neigh++){
	vtkIdType neighbour = neighbours->GetId(n_neigh);
	vtkIdType n_label = scalarData->GetTuple1(neighbour);
	if ( labels.find(n_label) != labels.end() ) {
	  labels[n_label] += 1;
	} else {
	  labels[n_label] = 1;
	}
	if (n_label == thisLabel) {
	  // Wont need to keep counting for vote,
	  break;
	}
      }
      if ( labels.find(thisLabel) == labels.end() ) {
	unsigned highestcount=0;
	vtkIdType highestlabel=-1;
	std::map<vtkIdType,unsigned>::iterator n_label;
	for ( n_label = labels.begin(); n_label != labels.end(); n_label++) {
	  if ( highestcount < n_label->second ) {
	    highestcount = n_label->second;
	    highestlabel = n_label->first;
	  }
	}
	std::cout << "Replacing label, point " << point;
	if ( highestlabel > -1 ) {
	  scalarData->SetTuple1(point, highestlabel);
	  std::cout << " new label " << highestlabel;
	}
	std::cout <<std::endl;
      }
    }
    return;
  }



NonUniformBSpline::Pointer derivativeOfSpline( NonUniformBSpline::Pointer splinein ) {
  NonUniformBSpline::Pointer splinederiv = NonUniformBSpline::New();
  /* =====================
     cylic case is not correct yet
     =====================*/
  int orderin = splinein->GetSplineOrder();
  int cyclic = splinein->GetCyclic();
  int degreein = orderin - 1;
  if (orderin < 2 ) {
    return 0;
  }

  // Order drops by one, control points drop by one, we eliminate start and
  // end knots and evaluate new control points as:
  // (degree / (u_(i+degree+1) - u_(i+1)))*(P_(i+1) - P_(i))
  // if we do the knots first then indices for u in this eq reduce by 1 and
  // match those of P.
  splinederiv->SetSplineOrder(orderin - 1);
  splinederiv->SetCyclic(cyclic);
  NonUniformBSpline::KnotListType knotlist = splinein->GetKnots();
  NonUniformBSpline::ControlPointListType cplist = splinein->GetControlPoints();
  for (unsigned ii=0; ii< knotlist.size(); ii++) {
    std::cout << cyclic << " " << ii << " " << knotlist[ii] << std::endl;
  }

  knotlist.pop_back();
  if(!cyclic) {
    knotlist.erase(knotlist.begin());
  } else {
    knotlist.pop_back();
  }
  NonUniformBSpline::ControlPointType firstpoint = *cplist.begin();
  NonUniformBSpline::ControlPointType lastpoint = *cplist.rbegin();
  for ( unsigned ii=0; ii+1 < cplist.size() ; ii++ ) {
    // Since the derivative control points only depend on points after them we
    // can update the list in sequence.
    unsigned topknot=ii+degreein;
    if (topknot >= knotlist.size() ) {
      std::cerr << "Error in calculating spline derivative, not enough knots"
		<< std::endl;
      return 0;
    }
    double knotdelta = knotlist[topknot] - knotlist[ii];
    NonUniformBSpline::ControlPointType controldelta;
    bool controldeltazero = 1;
    for (unsigned el=0; el<3; el++) {
      controldelta[el] = cplist[ii+1][el] - cplist[ii][el];
      controldeltazero &= controldelta[el] == 0;
    }
    if ( knotdelta == 0 ) {
      if ( ! controldeltazero ) {
	std::cerr << "Error in calculating spline derivative, duplicate knots"
		  << " without duplicate control points (possible, but not"
		  << " differentiable)." << std::endl;
	return 0;
      } else {
	for (unsigned el=0; el<3; el++) {
	  cplist[ii][el]=0;
	}
      }
    } else {
      for (unsigned el=0; el<3; el++) {
	cplist[ii][el] = degreein * controldelta[el] / knotdelta;
      }
    }
  }
  cplist.pop_back(); // and the last control point drops off

  if(cyclic) {
    cplist.insert(cplist.begin(),*cplist.rbegin());
    cplist.pop_back();
  }

  splinederiv->SetKnots(knotlist);
  splinederiv->SetControlPoints(cplist);

  return splinederiv;
}


NonUniformBSpline::Pointer splineFromBoundary( vtkPolyData *polyData, structuredPairBoundaryT boundary) {
  //NonUniformBSpline::PointType point ;
  unsigned numberoffaces = boundary.faceList.size();

  int facecontrolratio= FACECONTROLRATIO;
  int splineorder=4; //cubic
  // Due to cyclic points and anchor-end points this is not the final
  // number of control points
  int cyclic = boundary.boundaryType == structuredPairBoundaryT::cyclic;
  //cyclic=0;
  bool track=0;
  switch (boundary.faceList[0]) {
  case 198047:
  case 200222:
  case 200223:
    if (boundary.labels[0]==30 && boundary.labels[1]==35){
      track=1;
    }
    track=0;
  }
  NonUniformBSpline::PointListType midpointvals;

  for (unsigned ii=0; ii<numberoffaces; ii++ ) {
    NonUniformBSpline::PointType  midpointval;
    midpointval.Fill(0);
    vtkIdType face=boundary.faceList[ii];
    vtkCell *thisCell = polyData->GetCell(face);
    vtkIdList *thisCellPointIdList;
    vtkIdType pointCount = thisCell->GetNumberOfPoints();
    thisCellPointIdList = thisCell->GetPointIds();
    for ( vtkIdType jj=0; jj<pointCount; jj++ ) {
      double pointX[3];
      vtkIdType pointindex=thisCellPointIdList->GetId(jj);
      polyData->GetPoint(pointindex,pointX);
      for ( unsigned element=0 ; element<3 ; element++) {
	midpointval[element] += (pointX[element]-midpointval[element])/(jj+1);
      }
    }
    if(track)  std::cout << midpointval << std::endl;
    midpointvals.push_back(midpointval);
  }
  if (midpointvals.size()==2) {
    // End points only, introduce an intermediate point projected to the
    // sphere surface (if possible: but should only happen with adjacent input
    // triangles anyway).
    // If we don't do this then the end point support is sufficient to allow
    // middle control points to be zero, which takes the curve off the surface.
    midpointvals.push_back(midpointvals[1]); // Extend by one
    NonUniformBSpline::PointType  newmidpointval;
    double mag=0;
    std::cout<<"Adding intermediate point between adjacent end points" << std::endl;
    for (unsigned el=0 ; el < 3 ; el++ ) {
      newmidpointval[el] = midpointvals[0][el] + midpointvals[1][el];
      newmidpointval[el] /= 2.0;
      mag +=  newmidpointval[el]*newmidpointval[el];
    }
    mag=sqrt(mag);
    mag=1; // Don't do the projection as we'd need a more sophisticated model
    // to get it to work on hemisphere pairs.
    if (mag != 0) {
      for (unsigned el=0 ; el < 3 ; el++ ) {
	newmidpointval[el] = newmidpointval[el] / mag;
      }
    }
    midpointvals[1] = newmidpointval;
  }

  for (unsigned ii=0; track && ii<midpointvals.size() ; ii ++){
    std::cout << "ii" << ii << ": "<< midpointvals[ii] << std::endl;
  }

  int variablecontrolpoints = midpointvals.size() / facecontrolratio + 1;

  NonUniformBSpline::Pointer spline = NonUniformBSpline::New();

  NonUniformBSpline::KnotListType knots;

  // Cope if too few control points
  // Don't really need this, as anchoring spline for open and cycling knots
  // for closed will provide enough.
  int minvariablecontrolpoints = splineorder; // May need revision
  minvariablecontrolpoints = 2; // May need revision
  // Open splines will end up with variable controlpoints + 3 cps in
  // total, corresponds to order *2 + varcp + 1 knots

  if(variablecontrolpoints < minvariablecontrolpoints) {
    variablecontrolpoints = minvariablecontrolpoints;
  }

  // Or too many
  int maxvariablecontrolpoints = cyclic ?
    splineorder   + midpointvals.size() - 1 :
    midpointvals.size();
  if (track) {std::cout << maxvariablecontrolpoints << std::endl ;}
  if(variablecontrolpoints > maxvariablecontrolpoints) {
    variablecontrolpoints = maxvariablecontrolpoints;
  }

  if ( midpointvals.size() < variablecontrolpoints+3 ) {
    // Need to add more.
    NonUniformBSpline::PointListType newmidpointvals;
    unsigned ii;
    for (ii=0 ; ii+1<midpointvals.size(); ii++) {
      NonUniformBSpline::PointType  avgpoint;
      avgpoint.Fill(0);
      for (unsigned el=0; el<3 ; el++) {
	avgpoint[el] = (midpointvals[ii][el]+midpointvals[ii+1][el])/2;
      }
      newmidpointvals.push_back(midpointvals[ii]);
      newmidpointvals.push_back(avgpoint);
    }
    newmidpointvals.push_back(midpointvals[ii]);
    midpointvals=newmidpointvals;
  }
  for (unsigned ii=0; track && ii<midpointvals.size() ; ii ++){
    std::cout << "updii" << ii << ": "<< midpointvals[ii] << std::endl;
  }


  // Add anchor knots for open splines, for closed splines use uniform spacing
  // and don't include t=1
  for ( int ii=0 ; !cyclic && ii < splineorder-1; ii++) {
    knots.push_back(0);
  }
  for ( int ii=0 ; ii<=variablecontrolpoints ; ii++ ) {
    double knot;
    knot = 1.0/(variablecontrolpoints+cyclic) * ii;
    knots.push_back(knot);
  }
  for ( int ii=0 ; !cyclic && ii < splineorder-1; ii++) {
    knots.push_back(1);
  }

  for (unsigned ii=0; track && ii<knots.size(); ii++) {
    std::cout << "kt " << knots[ii] << std::endl;
  }
  
  spline->SetPoints(midpointvals);
  spline->SetSplineOrder(splineorder);

  // Must set cyclic before knots as SetKnots calculates the knots >=1 for us
  if ( cyclic ) spline->SetCyclic(1);
  spline->SetKnots(knots);

  spline->ComputeChordLengths();
  spline->ComputeControlPoints();

  if(1) { // Allow control point adjustments
    typedef NonUniformBSpline::ControlPointListType CPListType;
    CPListType cpoints = spline->GetControlPoints();
    if ( cyclic ) {
      //Nothing to do for cyclic.
    }
    else if (1) { // Fix end points
      //Due to knot definition end control points are already appropriately
      //duplicted, so only need to change the absolute end value.
      // Because of the way the end knots are handled there is only a single
      // control point for the duplicate knots.
      *(cpoints.begin()) = *(midpointvals.begin());
      *(cpoints.rbegin()) = *(midpointvals.rbegin());
      spline->SetControlPoints(cpoints);
      for (unsigned ii=0; track && ii<cpoints.size() ; ii ++){
	std::cout << "cpii" << ii << ": "<< cpoints[ii] << std::endl;
      }
    }
  }
  return spline;
};


unsortedBoundariesT getUnsortedBoundaries(vtkPolyData *polyData) {
    vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
    vtkIdType cellNumber = 0;
    unsortedBoundariesT unsortedBoundaries;

    for ( vtkIdType cell=0 ; cell < polyData->GetNumberOfCells() ;
	  cell++ ) {
      vtkCell *thisCell = polyData->GetCell(cell);
      vtkIdList *thisCellPointIdList;
      cellNumber = thisCell->GetNumberOfPoints();
      thisCellPointIdList = thisCell->GetPointIds();
      std::set <vtkIdType> labelsCount;
      if ( cellNumber == 3 ) {
	for ( unsigned ii=0 ; ii<3 ; ii++ ) {
	  labelsCount.insert((vtkIdType) scalarData->GetTuple1(thisCellPointIdList->GetId(ii)));
	}
	if (labelsCount.size() == 1 ) {
	  continue;
	}
	bool track=false;
	switch (cell) {
	case 506186:
	case 508008:
	case 504522:
	case 507973:
	case 506187:
	  track=1;
	  break;
	}
	if (track ) {
	  std::cout << "got " << cell << " " << labelsCount.size();
	    for (std::set<vtkIdType>::iterator thislabel=labelsCount.begin();
		 thislabel != labelsCount.end()  ; thislabel++)
	      std::cout << " " << *thislabel;
	  std::cout << std::endl;
	}



	vtkIdType labels[3];
      std::set<vtkIdType>::iterator thisLabel = labelsCount.begin();
	for ( unsigned ii=0 ; thisLabel != labelsCount.end() ; ii++, thisLabel++ ) {
	  labels[ii] = *thisLabel;
	}


	if (labelsCount.size() > 3) {
	  printf ("Error, %i labels for face %i\n", labelsCount.size(), cell);
	}
	if (labelsCount.size() == 2 ) {
	  unsortedBoundaries[labelpairT(labels[0],labels[1])].edgeFaces.insert(cell);
	}
	else if (labelsCount.size() == 3 ) {
	  if(track) std::cout << "inserting " << cell << " as junction" << std::endl;
	  unsortedBoundaries[labelpairT(labels[0],labels[1])].junctionFaces.insert(cell);
	  unsortedBoundaries[labelpairT(labels[0],labels[2])].junctionFaces.insert(cell);
	  unsortedBoundaries[labelpairT(labels[1],labels[2])].junctionFaces.insert(cell);
	}

      }

    }
    return unsortedBoundaries;
}



unsortedBoundariesT cleanUnsortedBoundaries
(
 vtkPolyData *polyData,
 unsortedBoundariesT unsortedBoundaries
 )
{
  unsortedBoundariesT newUnsortedBoundaries;
  vtkDataArray *scalarData =
    polyData->GetPointData()->GetScalars();
  for ( unsortedBoundariesT::iterator ii=unsortedBoundaries.begin() ;
	ii != unsortedBoundaries.end() ; ii++) {
    vtkIdType iilabel = ii->first.first;
    vtkIdType jjlabel = ii->first.second;
    std::set<vtkIdType> junctionFaces = ii->second.junctionFaces;
    for ( std::set<vtkIdType>::iterator startFace = junctionFaces.begin();
	  junctionFaces.size() > 0 && startFace != junctionFaces.end();
	  junctionFaces.erase(startFace), startFace=junctionFaces.begin()){
      bool bridgept=false;
      vtkIdList *collectJnNeighbours =
	getTriCellCrossBoundaryNeighbours( polyData, *startFace);
      for ( vtkIdType iineighbour=0 ;
	    iineighbour < collectJnNeighbours->GetNumberOfIds() ;
	    iineighbour++ ) {
	vtkIdType neighbour = collectJnNeighbours->GetId(iineighbour);
	std::set<vtkIdType>::iterator junctiontest =
	  junctionFaces.find(neighbour);
	
	if(junctiontest == junctionFaces.end() ) {
	  continue;
	}
	// First check against the other faces to see if this is a bridging
	// junction. We still allow it if it is (for now) but must detect
	// before attempting anything else.
	std::vector<vtkIdType> testEdge;
	testEdge = getEdgePointLeftRight(*startFace,*junctiontest,
					 polyData);
	if (testEdge.size()==2 || ( testEdge[0] >=0 && testEdge[1] >=0 )){
	  // Sharing an edge isn't enough, must contain common label on
	  // third point.
	  vtkIdType numberOfPointsInA;
	  vtkIdType numberOfPointsInB;
	  vtkIdType *ptsA, *ptsB;
	  polyData->GetCellPoints(*startFace, numberOfPointsInA, ptsA);
	  polyData->GetCellPoints(*junctiontest, numberOfPointsInB, ptsB);
	  vtkIdType otherA=-1;
	  vtkIdType otherB=-1;
	  for ( vtkIdType ptn = 0 ; ptn < numberOfPointsInA; ptn++) {
	    if ( ptsA[ptn] != testEdge[0] && ptsA[ptn] != testEdge[1] ) {
	      otherA=ptsA[ptn];
	      break;
	    }
	  }
	  for ( vtkIdType ptn = 0 ; ptn < numberOfPointsInB; ptn++) {
	    if ( ptsB[ptn] != testEdge[0] && ptsB[ptn] != testEdge[1] ) {
	      otherB=ptsB[ptn];
	      break;
	    }
	  }
	  if ( otherA < 0 || otherB < 0 ) {
	    std::cerr << "Failed to find non-edge point on cell in bridge"
		      << " detection, faces " << *startFace << " "
		      << *junctiontest << std::endl;
	  } else {
	    if ( scalarData->GetTuple1(otherA) ==
		 scalarData->GetTuple1(otherB) ) {
	      // Dilate this bridge
	      scalarData->SetTuple1(testEdge[0],scalarData->GetTuple1(otherA));
	      scalarData->SetTuple1(testEdge[1],scalarData->GetTuple1(otherA));
	      std::cout << "Clearing edge " << testEdge[0] << " "
			<<testEdge[1] << std::endl;
	      //scalarData->SetTuple1(testEdge[0],200);
	      //scalarData->SetTuple1(testEdge[1],200);
		bridgept = true;
	      break;
	    }
	  }
	}
      }
      collectJnNeighbours->Delete();
      if(bridgept) {
	continue;
      }
    }
  }
  polyData->GetPointData()->SetScalars(scalarData);
  //  scalarData->Delete();
  newUnsortedBoundaries = getUnsortedBoundaries(polyData);
  return newUnsortedBoundaries;
}




sortedBoundariesT sortUnsortedBoundaries(vtkPolyData *polyData, unsortedBoundariesT unsortedBoundaries) {
    sortedBoundariesT sortedBoundaries;
    for ( unsortedBoundariesT::iterator ii=unsortedBoundaries.begin() ;
	  ii != unsortedBoundaries.end() ; ii++) {
	vtkIdType iilabel = ii->first.first;
	vtkIdType jjlabel = ii->first.second;
	printf ("%i:%i %i edge faces, %i junctions\n",
		iilabel, jjlabel,
		ii->second.edgeFaces.size(), ii->second.junctionFaces.size());
        std::set<vtkIdType> edgeFaces = ii->second.edgeFaces;
        std::set<vtkIdType> junctionFaces = ii->second.junctionFaces;

	bool track =false;
	if (edgeFaces.find(506186) != edgeFaces.end() ) {
	  std::cout << "got 506186 on " << iilabel << " " << jjlabel << std::endl;
	  track=1;
	}
	if ( edgeFaces.find(508008) != edgeFaces.end() ) {
	  std::cout << "got 508008 on " << iilabel << " " << jjlabel << std::endl;
	  track=1;
	}
	
	// The actual sorted edges algorithm. This is destructive on the
	// unsortedBoundaries. We keep removing keys, so need to hold onto
	// a begin iterator until there are none left.
	for ( std::set<vtkIdType>::iterator startFace = junctionFaces.begin();
	      junctionFaces.size() > 0 && startFace != junctionFaces.end();
	      junctionFaces.erase(startFace), startFace=junctionFaces.begin()){
	  if (track) std::cout << *startFace << " start" << std::endl;
	  std::vector<vtkIdType> buildEdge;
	  // Need all cell neighbours.
	  vtkIdType thisFace = *startFace;
	  switch (thisFace) {
	  case 506186:
	  case 506187:
	  case 508008:
	    track=1;
	  }

	  bool foundnext=false;
	  bool hitjunction=false;
	  while (!hitjunction) {
	    if (track) std::cout << "push " << thisFace << std::endl;
	    buildEdge.push_back(thisFace);
	    foundnext=false;
	    vtkIdList *collectNeighbours = getTriCellCrossBoundaryNeighbours(
					     polyData, thisFace);
	    for ( vtkIdType iineighbour=0 ;
		  iineighbour < collectNeighbours->GetNumberOfIds() ;
		  iineighbour++ ) {
	      vtkIdType neighbour = collectNeighbours->GetId(iineighbour);
	      std::set<vtkIdType>::iterator locateNeighbour = edgeFaces.end();
	      locateNeighbour = edgeFaces.find(neighbour);
	      if ( locateNeighbour != edgeFaces.end() ) {
		thisFace = neighbour;
		edgeFaces.erase(locateNeighbour);
		foundnext = true;
		break;
	      }
	      locateNeighbour = junctionFaces.find(neighbour);
	      if ( locateNeighbour != junctionFaces.end() &&
		   locateNeighbour != junctionFaces.begin() ) {
		thisFace = neighbour;
		junctionFaces.erase(locateNeighbour);
		foundnext = true;
		hitjunction = true;
		if (track) std::cout << "push " << thisFace << std::endl;
		buildEdge.push_back(thisFace);
		break;
	      }
	    }
	    if ( ! foundnext ) {
	      vtkIdType problemface = thisFace; // If needed for debugging
	      std::cerr << "Failed to find next face, bug or problem data,"
			<< " checked " <<  collectNeighbours->GetNumberOfIds()
			<< " cross boundary neighbours, " << problemface
			<< std::endl;
	      for (unsigned nii=0 ; nii<collectNeighbours->GetNumberOfIds(); nii++) {
		std::cout << collectNeighbours->GetId(nii) << std::endl;
	      }
	      problemface += 0;
	    }
	  }
	  sortedBoundaries.addEdge(buildEdge, polyData,
				   structuredPairBoundaryT::noncyclic,
				   iilabel, jjlabel);
	}
	// No junctions left, only closed loops
	for ( std::set<vtkIdType>::iterator startFace = edgeFaces.begin();
	      edgeFaces.size() > 0 && startFace != edgeFaces.end();
	      edgeFaces.erase(startFace), startFace=edgeFaces.begin()){
	  std::vector<vtkIdType> buildEdge;

	  // Need all cell neighbours which share a boundary edge.

	  vtkIdType thisFace = *startFace;
	  std::set<vtkIdType>::iterator locateNeighbour = edgeFaces.begin();
	  bool foundnext=true;
	  bool sawEnd=0;
	  while (foundnext) {
	    buildEdge.push_back(thisFace);
	    foundnext=false;
	    vtkIdList *collectNeighbours = getTriCellCrossBoundaryNeighbours(
                                             polyData, thisFace);
	    sawEnd=0;
	    for ( vtkIdType iineighbour=0 ;
		  iineighbour < collectNeighbours->GetNumberOfIds() ;
		  iineighbour++ ) {
	      vtkIdType neighbour = collectNeighbours->GetId(iineighbour);
	      locateNeighbour = edgeFaces.find(neighbour);
	      if ( locateNeighbour != edgeFaces.end() &&
	           locateNeighbour != edgeFaces.begin() ) {
		thisFace = neighbour;
		edgeFaces.erase(locateNeighbour);
		foundnext = true;
		break;
	      }
	      if ( locateNeighbour == edgeFaces.begin() ) {
		sawEnd=1;
	      }
	    }
	  }
	  if ( !sawEnd ) {
	    printf("Failed to close boundary loop\n");
	  }
	  if (buildEdge.size() > dropcyclicsize) {
	    std::cout << "cyclic" << std::endl;
	    sortedBoundaries.addEdge(buildEdge, polyData,
				     structuredPairBoundaryT::cyclic,
				     iilabel, jjlabel);
	  }
	}
    }
    return sortedBoundaries;
}


void saveBoundaryPolyData (vtkPolyData *polyData, sortedBoundariesT sortedBoundaries) {
    vtkPoints *points = polyData->GetPoints();
    vtkIdType numberOfPoints = polyData->GetNumberOfPoints();
    
    vtkIdType *listOfPointIds;
    vtkIdType numberOfPointsInCurrentCell;
    double point[3];
    
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

    unsigned labelNonCyclic = 1;
    unsigned labelCyclic = 5;
    unsigned labelJunction = 10;

    std::cout << "all points" <<std::endl;
    for (vtkIdType pointNumber = 0; pointNumber < numberOfPoints;
	 pointNumber++) {
      points->GetPoint(pointNumber, point);
      outputPoints->InsertPoint(pointNumber, point);
      outputVectors->InsertTuple3(pointNumber, 0, 0, 0);
      outputLabels->SetValue(pointNumber, 0); // zero for region, 1 for edge
                                              // 2 for junction
    }
    std::cout << "through boundaries" <<std::endl;
    unsigned boundcount=0;
    unsigned boundlen=sortedBoundaries.structuredBoundaries.size();
    for ( sortedBoundariesT::iterator thisBound = sortedBoundaries.begin();
	  thisBound != sortedBoundaries.end() ; thisBound++) {
      boundcount++;
      std::vector<vtkIdType> faceList = thisBound->faceList;
      std::vector<vtkIdType> jnpts;
      structuredPairBoundaryT::boundaryT type = thisBound->boundaryType;
      for (unsigned face=0 ; face < faceList.size() ; face++ ) {
	// !face catches faceList.size == 1 and divide by zero here.
	// Not that that should happen...
	int junction = (type==structuredPairBoundaryT::noncyclic) &&
	  (!face || !(face%(faceList.size()-1)));
	int label =  type==structuredPairBoundaryT::noncyclic ? labelNonCyclic
	  : labelCyclic;
	polyData->GetCellPoints(faceList[face],
				numberOfPointsInCurrentCell, listOfPointIds);
	for (vtkIdType ii=0 ; ii<numberOfPointsInCurrentCell ; ii++) {
	  outputLabels->SetValue(listOfPointIds[ii], label);
	  if (junction) {
	    jnpts.push_back(listOfPointIds[ii]);
	  }
	  vtkIdType line[2];
	  line[0]=listOfPointIds[ii];
	  line[1]=listOfPointIds[(ii+1)%numberOfPointsInCurrentCell];
	  outputLines->InsertNextCell(2, line);
	}
	for (std::vector<vtkIdType>::iterator jnpt=jnpts.begin();
	     jnpt!=jnpts.end(); jnpt++){
	  outputLabels->SetValue(*jnpt, labelJunction);
	}
      }
    }
    std::cout << "newpoly data" <<std::endl;

    vtkPolyData *outputPolyData = vtkPolyData::New();
    outputPolyData->SetPoints(outputPoints);
    outputPolyData->GetPointData()->SetScalars(outputLabels);
    outputPolyData->GetPointData()->SetVectors(outputVectors);
    outputPolyData->SetLines(outputLines);
    outputPolyData->BuildCells();
    outputPolyData->BuildLinks();
    
    std::cout << "deleting support" <<std::endl;

    outputPoints->Delete();
    outputLabels->Delete();
    outputLines->Delete();
    outputVectors->Delete();

    std::cout << "writing" <<std::endl;
    
    vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
    writer->SetFileName("testnewboundarymodel.vtk");
    writer->SetInput(outputPolyData);
    writer->Write();
    writer->Delete();
    outputPolyData->Delete();

    std::cout << "writerfinished" <<std::endl;

    return;
}


void saveSplinePolyData(vtkPolyData *polyData, sortedBoundariesT sortedBoundaries, char *outname, int showcontrolpoints=0) {
    vtkPolyData *outputCurvePolyData = vtkPolyData::New();
    vtkPoints *outputCurvePoints = vtkPoints::New();
    vtkIntArray *outputCurveLabels = vtkIntArray::New();
    outputCurveLabels->SetNumberOfComponents(1);
    vtkCellArray *outputCurveLines = vtkCellArray::New();
    unsigned pointcount=0;
    for ( sortedBoundariesT::iterator thisBound = sortedBoundaries.begin();
	  thisBound != sortedBoundaries.end() ; thisBound++) {
      if (thisBound->spline.IsNull() ) continue;
      unsigned intervals=500;
      int cyclic = thisBound->boundaryType == structuredPairBoundaryT::cyclic;
      for (unsigned ii=0 ; ii<=intervals; ii++) {
	//std::cout<<"Point "<< ii << std::endl;

	double pos=ii/(double)intervals;

	NonUniformBSpline::PointType loc;
	loc = thisBound->spline->EvaluateSpline(pos);
	//std::cout << "Eval " << pos << " res " << loc << std::endl;

        double mag=sqrt(loc[0]*loc[0]+loc[1]*loc[1]+loc[2]*loc[2]);
	mag=1;
        double locX[]={loc[0]/mag,loc[1]/mag,loc[2]/mag};
        if (mag==0) for (unsigned ii=0 ; ii < 3 ; ii++) locX[ii]=0;

        outputCurvePoints->InsertNextPoint(locX);
	outputCurveLabels->InsertNextValue(cyclic*3);
        if ( ii != 0 ) {
	  vtkIdType line[] = {pointcount-1, pointcount};
          outputCurveLines->InsertNextCell(2,line);
	}
        pointcount++;
      }
      
      if(showcontrolpoints) {      
        typedef NonUniformBSpline::ControlPointListType CPListType;
        CPListType cpoints = thisBound->spline->GetControlPoints();
        unsigned cpsize = cpoints.size();
        for ( unsigned ii=0 ; ii< cpsize ; ii++ ) {
        //double mag=sqrt(loc[0]*loc[0]+loc[1]*loc[1]+loc[2]*loc[2]);
	double mag=1;
        double locX[]={cpoints[ii][0]/mag,cpoints[ii][1]/mag,cpoints[ii][2]/mag};
        outputCurvePoints->InsertNextPoint(locX);
	outputCurveLabels->InsertNextValue(6);
	
	if ( ii != 0 ) {
	  vtkIdType line[] = {pointcount-1, pointcount};
          outputCurveLines->InsertNextCell(2,line);
	}
        pointcount++;
        }
      }
      int showtangents=0;
      if(showtangents) {
	NonUniformBSpline::Pointer derivative =
	  derivativeOfSpline(thisBound->spline);
	unsigned tangentintervals = 10;
	for (unsigned ii=0 ; ii<=tangentintervals; ii++) {
	  double pos=ii/(double)tangentintervals;
	  NonUniformBSpline::PointType locstart, loctangent;
	  locstart = thisBound->spline->EvaluateSpline(pos);
	  loctangent = derivative->EvaluateSpline(pos);
	  //std::cout << "Eval " << pos << " res " << loc << std::endl;
	  double mag=sqrt(loctangent[0]*loctangent[0]+loctangent[1]*loctangent[1]+loctangent[2]*loctangent[2]);
	  double scale;
	  if (mag==0) {
	    scale=1;
	  } else {
	    scale=tanh(mag)/mag/4;
	  }
	  double loc1[3];
	  double loc2[3];
	  std::cout << "start " ;
	  for (unsigned el=0 ;el<3; el++) {
	    loc2[el]=locstart[el]+loctangent[el]*scale;
	    loc1[el]=locstart[el];
	    std::cout << loc1[el] << " ";
	  }
	  std::cout << std::endl;
	  std::cout << "end " ;
	  for (unsigned el=0 ;el<3; el++) {
	    std::cout << loc2[el] << " ";
	  }
	  std::cout << std::endl;
	  std::cout << "tangent " ;
	  for (unsigned el=0 ;el<3; el++) {
	    std::cout << loctangent[el] << " ";
	  }
	  std::cout << std::endl;
	  outputCurvePoints->InsertNextPoint(loc1);
	  outputCurvePoints->InsertNextPoint(loc2);
	  int tangentLabel=10;
	  outputCurveLabels->InsertNextValue(tangentLabel);
	  outputCurveLabels->InsertNextValue(tangentLabel);
	  vtkIdType line[] = {pointcount, pointcount+1};
          outputCurveLines->InsertNextCell(2,line);
	  pointcount += 2;
	}
      }
    }

    vtkPolyDataWriter *writer = vtkPolyDataWriter::New();

    outputCurvePolyData->SetPoints(outputCurvePoints);
    outputCurvePolyData->SetLines(outputCurveLines);
    outputCurvePolyData->GetPointData()->SetScalars(outputCurveLabels);
    writer->SetFileName(outname);
    writer->SetInput(outputCurvePolyData);
    writer->Write();
    outputCurvePoints->Delete();
    outputCurveLines->Delete();
    outputCurveLabels->Delete();
    outputCurvePolyData->Delete();
    writer->Delete();
    return;
}


void saveAnyPolyData(vtkPolyData *polyData, char *name) {
    vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
    writer->SetFileName(name);
    writer->SetInput(polyData);
    writer->Write();
    writer->Delete();
}


bool updateStructuredBoundaryFaces ( vtkPolyData *polyData,
				     structuredPairBoundaryT &boundary ) {

  bool result=1; // Okay
  NonUniformBSpline::Pointer spline = boundary.spline;
  std::vector <vtkIdType> newFaceList;
  std::vector <double> newTList;
  std::vector <std::vector<vtkIdType> > newEdges;
  std::vector <vtkIdType> testEdge;
  NonUniformBSpline::PointType tpoint;
  bool cyclic = boundary.boundaryType == structuredPairBoundaryT::cyclic;
  double t=0;
  // Rough estimate of the face spacing on the spline
  double interval = 1/( (double)FACECONTROLRATIO
			* spline->GetControlPoints().size() ) ;
  double step;
  vtkIdList *cellslist = vtkIdList::New();

  // Find our first face, easier after this as we have neighbours.
  tpoint = spline->EvaluateSpline(t);
  vtkIdType point = polyData->FindPoint(tpoint.GetDataPointer());

  if ( point < 0 ) {
    std::cerr << "Couldn't find nearest to start point, result "
	      << point << " " << tpoint <<std::endl;
    //TODO, implement an all-faces search.
    return false;
  }

  const unsigned NEIGHBOURSEARCHDIST = 5;
  std::set <vtkIdType> checkedcells;
  std::set <vtkIdType> realneighbours; //For reporting in errors.

  polyData->GetPointCells( point, cellslist );
  vtkIdType startcell = -1;
  // Becomes an all cells search if neighbourdist limt is replaced by
  // size of cells list test
  for ( unsigned neighbourdist = 0 ;
	startcell < 0 && neighbourdist < NEIGHBOURSEARCHDIST;
	neighbourdist++) {
    for ( vtkIdType ii=0 ; ii<cellslist->GetNumberOfIds(); ii++) {
      vtkIdType *tripoint;
      vtkIdType pointcount;
      vtkIdType thiscell = cellslist->GetId(ii);
      checkedcells.insert(thiscell);
      polyData->GetCellPoints( thiscell, pointcount, tripoint);
      if ( polyData->GetCellType(thiscell) != VTK_TRIANGLE ) {
	continue;
      }
      realneighbours.insert(thiscell);
      //    if (pointInsideMeshTri ( tpoint.GetDataPointer(), polyData, tripoint )) {
      if (pointInsideMeshFaceEdges( tpoint.GetDataPointer(), polyData, thiscell)){
	startcell = thiscell;
	break;
      }
    }
    if ( startcell < 0 ) {
      vtkIdList *nextneighbours = vtkIdList::New();
      for ( vtkIdType ii=0 ; ii<cellslist->GetNumberOfIds(); ii++) {
	vtkIdType thisFace = cellslist->GetId(ii);
	vtkIdList *theseneighbours =
	  getTriCellNeighbours (polyData, thisFace );
	for (unsigned ncount=0 ; ncount < theseneighbours->GetNumberOfIds();
	     ncount++) {
	  vtkIdType neighbour = theseneighbours->GetId(ncount);
	  if ( checkedcells.find(neighbour) == checkedcells.end() ) {
	    nextneighbours->InsertUniqueId(neighbour);
	  }
	}
	theseneighbours->Delete();
      }
      cellslist->Delete();
      cellslist=nextneighbours;
    }
  }
  cellslist->Delete();
  if ( startcell < 0 ) {
    std::cerr << "Start cell not found for point id " << point
	      << " after depth " << NEIGHBOURSEARCHDIST << " search, "
	      << realneighbours.size() << " cells checked of "
	      << checkedcells.size() << " potential."
	      << std::endl;
    return false;
  }

  // Now loop until t=1;
  vtkIdType lastface = startcell;
  newFaceList.push_back(lastface);
  newTList.push_back(t);
  while ( t<1 ) {
    //std::cout << "t " << t << std::endl;
    vtkIdList *neighbourlist =  getTriCellNeighbours (polyData, lastface);
    vtkIdType nextcell=-1;
    step = interval;
    vtkIdType testcell;
    vtkIdType *tripoint;
    vtkIdType pointcount;
    double tnext;
    double stepmax=1;
    double stepmin=0;
    while (nextcell<0) {
      tnext = t + step ;
      if (tnext > 1.0) {
	step = 1.0 - t;
	tnext=1.0;
      }
      //      std::cout << "tnext " << tnext << " lastface " << lastface << " #N " <<neighbourlist->GetNumberOfIds() << std::endl;
      tpoint = spline->EvaluateSpline(tnext);
      for ( vtkIdType ii=0 ; ii<neighbourlist->GetNumberOfIds(); ii++) {
	testcell = neighbourlist->GetId(ii);
	//	std::cout << "test " << testcell << std::endl;
	polyData->GetCellPoints( testcell, pointcount, tripoint);
	//	std::cout << "testinside (p=) " << pointcount << std::endl;
	if (pointcount == 3 &&
	    //	    pointInsideMeshTri(tpoint.GetDataPointer(), polyData, tripoint )) {
	    pointInsideMeshFaceEdges(tpoint.GetDataPointer(), polyData, testcell)){
	  nextcell = testcell;
	  //std::cout << "done (found next)" << std::endl;
	  break;
	}
	//std::cout << "done" << std::endl;
      }
      polyData->GetCellPoints( lastface, pointcount, tripoint);
      //      std::cout << "Didn't land in neighbour" << std::endl;
      // Constants for these step adjustments are balanced to avoid
      // oscillation, but might be better with a max-min scheme.
      //      if (pointInsideMeshTri(tpoint.GetDataPointer(), polyData, tripoint )) {
      if (pointInsideMeshFaceEdges(tpoint.GetDataPointer(), polyData, lastface)){
	stepmin=step;
	step= ((stepmax + stepmin) / 2) > (stepmin*1.5) ? 
	  (1.5 * stepmin) :
	  ((stepmax + stepmin) / 2);
	// Different from max as max is initialised at
	// 1 and we don't want to jump to a massive step
	//std::cout << "Min " << step << " " << stepmin << " " << stepmax << std::endl;
	if ( tnext >= 1.0 ) {
	  break;
	}
      } else {
	//std::cout << "Max " << step << " " << stepmin << " " << stepmax << std::endl;
	if (step < stepmax ) {
	  stepmax=step;
	}
	step=(stepmax+stepmin)/2;
      }
      if ( stepmax - stepmin < 1e-15 ) {
	// Aproximately the limit of double precision in 0,1, implies we're
	// probably trapped on a boundary.
	std::cerr << "Failed on a boundary traversal last face "
		  << lastface << std::endl;
	tnext=1;
	break;
      }
    }
    neighbourlist->Delete();
    if (nextcell >= 0 ) {
      newFaceList.push_back(nextcell);
      newTList.push_back(tnext);
      testEdge = getEdgePointLeftRight(lastface,nextcell,polyData);
      if (testEdge.size()!=2 || testEdge[0]==-1 || testEdge[1]==-1) {
	std::cerr
	  << "Was unable to resolve points left right when building edge list"
	  << std::endl;
	testEdge.resize(2,0);
      }
      newEdges.push_back(testEdge);
      lastface=nextcell;
    } else if ( tnext < 1 ) {
      std::cerr << "Lost next cell, but t<1: " << t << std::endl;
      result = false;
    }
    t = tnext;
    // Now reset any loops back over the same edge:
    if ( newFaceList.size()>2 &&
	 newFaceList[newFaceList.size()-1] ==
	 newFaceList[newFaceList.size()-3] ) {
      std::cout << "Culling re-cross " << newFaceList[newFaceList.size()-3] <<
	" " << newFaceList[newFaceList.size()-2] << " "
		<<newFaceList[newFaceList.size()-3] << std::endl;
      newFaceList.pop_back();
      newFaceList.pop_back();
      newTList.pop_back();
      newTList.pop_back();
      newEdges.pop_back();
      newEdges.pop_back();
    }
  }

  if (result) {
    if(cyclic) {
      // We should have wrapped into the start cell, but maybe not
      if ( lastface != startcell ) {
	/*
	testEdge = getEdgePointLeftRight(lastface,startcell,polyData);
	std::cout << "Extended cyclic edge, possibly problematic " << testEdge[0] << " " << testEdge[1] <<std::endl;
	if (testEdge.size()!=2) {
	  std::cerr
	    << "Was unable to resolve points left right when building edge list"
	    << std::endl;
	  testEdge.resize(2,0);
	}
	newEdges.push_back(testEdge);
	*/
      }
    }

    boundary.faceList = newFaceList;
    boundary.edges = newEdges;
    boundary.tList = newTList;
  }


  //int direction; // 0 unset, 1 1st label on left, 2 2nd label on left
  return result;
}



void paintSortedBoundariesToPolyData ( vtkPolyData *polyData,
			      sortedBoundariesT sortedBoundaries,
                              std::vector<vtkIdType> paintlabels) {
  std::map<vtkIdType,vtkIdType> nextPts;
  paintlabels.resize(0); // Disable selective redraw, still suspect.
  vtkDataArray *labels = polyData->GetPointData()->GetScalars();

  // If given paintlabels to redo we limit our points (and bounds) involved
  // to those affected. Otherwise redo all points.

  // Set up unlabelledPts
  std::set<vtkIdType> unlabelledPts;
  vtkIdType totalpoints = polyData->GetNumberOfPoints();
  if ( paintlabels.size() == 0 ) {
    for (vtkIdType ii=0; ii < totalpoints; ii++){
      unlabelledPts.insert(ii);
    }
  } else {
    for (vtkIdType ii=0; ii < totalpoints; ii++){
      vtkIdType label = labels->GetTuple1(ii);
      for (unsigned whichlabel=0; whichlabel < paintlabels.size() ;
	   whichlabel++) {
	if (paintlabels[whichlabel] == label) {
	  std::cout << whichlabel << " " << label << std::endl;
	  unlabelledPts.insert(ii);
	}
      }
    }
  }
  //  std::cout << "unlabelledPts.size " << unlabelledPts.size() << std::endl;

  // First paint the edges on

  for ( sortedBoundariesT::iterator thisBound = sortedBoundaries.begin();
	thisBound != sortedBoundaries.end() ; thisBound++) {
    structuredPairBoundaryT bound=*thisBound;
    bool cyclic = (bound.boundaryType == structuredPairBoundaryT::cyclic);
    vtkIdType labelleft, labelright;
    switch (bound.direction) {
    case LEFT1ST:
      labelleft=bound.labels[0];
      labelright=bound.labels[1];
      break;
    case RIGHT1ST:
      labelright=bound.labels[0];
      labelleft=bound.labels[1];
      break;
    default:
      std::cerr << "Can't paint directionless boundary, will be ignored"
		<< std::endl;
      continue;
    }

    if ( paintlabels.size() > 0 ) {
      bool doboundleft = 0, doboundright = 0;
      for (unsigned whichlabel=0; whichlabel < paintlabels.size() ;
	   whichlabel++) {
	if (labelleft==paintlabels[whichlabel]) {
	  doboundleft=1;
	}
	if (labelright==paintlabels[whichlabel]) {
	  doboundright=1;
	}
      }
      if ( ! (doboundleft && doboundright) ) {
	// Don't bother doing boundaries that wont change the labels.
	continue;
      }
    }

    unsigned faceii=0;

    vtkIdType firstface, lastface;
    std::vector<vtkIdType> faceList = bound.faceList;
    std::vector<vtkIdType>::iterator face = faceList.begin();
    if ( face != faceList.end() ) {
      firstface=*face;
      lastface=firstface;
      face++;
      faceii++;
    }
    vtkIdType faceminus1 = lastface;
    vtkIdType faceminus2 = lastface;
    // Setup left and right faces
    for ( ; face != faceList.end(); face++, faceii++ ) {
      std::vector<vtkIdType> edgeLeftRight =
	getEdgePointLeftRight(lastface, *face, polyData);
      if(edgeLeftRight[0] < 0 || edgeLeftRight[1]<0) {
	std::cerr << "Problem finding edge for faces " << lastface << " "
		  << *face << std::endl;
	if (lastface == *face ) {
	  std::cerr << "(" << lastface << " twice)" << std::endl;
	}
      } else {
	std::map<vtkIdType,vtkIdType>::iterator findleft, findright;
	findleft=nextPts.find(edgeLeftRight[0]);
	findright=nextPts.find(edgeLeftRight[1]);
	if ( findleft != nextPts.end() && findright != nextPts.end() ) {
	  // Already labelled a point on this edge, eliminate the common label.
	  vtkIdType alreadyleft = findleft->second;
	  vtkIdType alreadyright = findright->second;
	  if (alreadyleft == labelright && alreadyright == labelleft ) {
	    // Crossing back over edge. The common point for the last two edges
	    // crossed is the outside of the curve, so both points get that
	    // label. The last edge crossed is the same as the one we're
	    // crossing.
	    // The edge before that is the edge between the p
	    if (faceminus1 == faceminus2 ) {
	      std::cout << "Painting error, re-crossed edge too near start." << std::endl;
	      std::cout << firstface << " " << lastface << " " << faceminus1 << " " << faceminus2 << " " << *face << std::endl;	      
	      std::cout << "faceii " << faceii << std::endl;
	      std::cout << "Faces to ii:" << std::endl;
	      for (unsigned faceii_ii=0; faceii_ii <= faceii; faceii_ii++) {
		std::cout << bound.faceList[faceii_ii] << std::endl;
	      }
	      std::vector<vtkIdType> stepbackEdgeLeftRight = 
		getEdgePointLeftRight(faceminus2, lastface, polyData);
	      std::cout << "This edge ";
	      for (unsigned el=0; el<2 ; el++ ) std::cout << " " << edgeLeftRight[el];
	      std::cout << std::endl;

	    }
	    else {
	      std::vector<vtkIdType> stepbackEdgeLeftRight = 
		getEdgePointLeftRight(faceminus2, faceminus1, polyData);
	      int foundcommonpoint=0;
	      vtkIdType commonpoint=-1;
	      for (unsigned iilastedge=0; !foundcommonpoint &&
		     iilastedge<2; iilastedge++) {
		for (unsigned iiedge=0; !foundcommonpoint && 
		       iiedge<2; iiedge++) {
		  if ( stepbackEdgeLeftRight[iilastedge] == 
		       edgeLeftRight[iiedge] ) {
		    foundcommonpoint+=1;
		    commonpoint=edgeLeftRight[iiedge];
		  }
		}
	      }
	      if ( foundcommonpoint > 1 ) std::cout << "common point dupl." << std::endl;
	      if ( 0 && foundcommonpoint ) { // Try without
		std::cout << "common point " << commonpoint << ": label " <<
		  nextPts[commonpoint] << std::endl;
		nextPts[edgeLeftRight[0]]=nextPts[commonpoint];
		nextPts[edgeLeftRight[1]]=nextPts[commonpoint];
	      } else {
		if ( ! foundcommonpoint ) {
		  std::cout << "Painting error, re-crossed edge didn't find outside point" << std::endl;
		}
	      }
	    }
	  }
	  else if (alreadyleft==labelright) {
	    nextPts[edgeLeftRight[0]]=labelleft;
	  }
	  else if (alreadyright==labelleft) {
	    nextPts[edgeLeftRight[1]]=labelright;
	  }
	} else {
	  nextPts[edgeLeftRight[0]]=labelleft;
	  nextPts[edgeLeftRight[1]]=labelright;
	}
      }
      faceminus2 = faceminus1;
      faceminus1 = lastface;
      lastface = *face;
    }
    if ( cyclic ) {
      std::vector<vtkIdType> edgeLeftRight =
	getEdgePointLeftRight(lastface, firstface, polyData);
      for(unsigned ii=0; ii<2; ii++) {
	if (edgeLeftRight[ii] < 0 ) {
	  std::cerr << "Problem finding edge for faces " << lastface << " "
		    << *face << std::endl;
	} else {
	  switch (ii) {
	  case 0:
	    nextPts[edgeLeftRight[ii]]=labelleft;
	    break;
	  case 1:
	    nextPts[edgeLeftRight[ii]]=labelright;
	    break;
	  }
	}
      }
    }
  }

  // Face setup done, now do a loop where we paint the nextPts and eliminate
  // them from unlabelled, then evaluate a new set of nextPts;

  while ( nextPts.size() > 0 ) {
    std::map<vtkIdType,vtkIdType> updatePts = nextPts;
    nextPts.clear();
    for (std::map<vtkIdType,vtkIdType>::iterator pt = updatePts.begin();
	 pt != updatePts.end(); pt++) {
      labels->SetTuple1(pt->first, pt->second);
      unlabelledPts.erase(pt->first);
      vtkSmartPointer<vtkIdList> neighbourPts =
	vtkSmartPointer<vtkIdList>::New();
      getConnectedVertices(polyData, pt->first, neighbourPts);
      for (vtkIdType ii=0; ii<neighbourPts->GetNumberOfIds(); ii++) {
	vtkIdType testpt=neighbourPts->GetId(ii);
	if ( unlabelledPts.find(testpt) != unlabelledPts.end() &&
	     updatePts.find(testpt) == updatePts.end() ) {
	  //nextPts[testpt]=0;
	  nextPts[testpt]=pt->second;
	}
      }
    }
  }
  std::cout << "Painting done"<<std::endl;
  return;
}



void paintSortedBoundariesToPolyData ( vtkPolyData *polyData,
			      sortedBoundariesT sortedBoundaries) {
  std::vector<vtkIdType> emptylabels(0);
  paintSortedBoundariesToPolyData( polyData,
				   sortedBoundaries,
				   emptylabels );
  return;
}


sortedBoundariesT buildBoundaryModelKernel(vtkPolyData *polyData)
  {
    unsortedBoundariesT unsortedBoundaries = getUnsortedBoundaries(polyData);
    unsortedBoundaries = cleanUnsortedBoundaries(polyData, unsortedBoundaries);
    saveAnyPolyData(polyData,"cleaned.vtk");
    sortedBoundariesT sortedBoundaries = sortUnsortedBoundaries(polyData, unsortedBoundaries);

    bool saveboundaries=0;
    if (saveboundaries) saveBoundaryPolyData(polyData, sortedBoundaries);

    return sortedBoundaries;
  }



vtkPolyData* buildBoundaryModel(vtkPolyData *polyData) {
    sortedBoundariesT sortedBoundaries = buildBoundaryModelKernel(polyData);
    // Build the polydata that boundary model needs to return

    vtkPoints *points = polyData->GetPoints();
    vtkIdType numberOfPoints = polyData->GetNumberOfPoints();
    vtkIdType pointNumber = 0;
    vtkIdType *listOfPointIds;
    vtkIdType numberOfPointsInCurrentCell;
    double point[3];

    vtkCellArray *outputLines = vtkCellArray::New();

    // Not sure what the output vectors were for since they get set to zero,
    // the update vectors for the boundary model iterative loop?
    vtkFloatArray *outputVectors = vtkFloatArray::New();
    outputVectors->SetNumberOfComponents(3);
    outputVectors->SetNumberOfValues(numberOfPoints);

    vtkPoints *outputPoints = vtkPoints::New();
    outputPoints->SetDataTypeToFloat();
    outputPoints->Allocate(numberOfPoints);
    
    vtkIntArray *outputLabels = vtkIntArray::New();
    outputLabels->SetNumberOfComponents(1);
    outputLabels->SetNumberOfValues(numberOfPoints);

    for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++) {
      points->GetPoint(pointNumber, point);
      outputPoints->InsertPoint(pointNumber, point);
      outputVectors->InsertTuple3(pointNumber, 0, 0, 0);
      outputLabels->SetValue(pointNumber, 0); // zero for region, 1 for edge
                                              // 2 for junction
    }
    ////////XXXXXXXX
    for ( sortedBoundariesT::iterator thisBound = sortedBoundaries.begin();
	  thisBound != sortedBoundaries.end() ; thisBound++) {
      std::vector<vtkIdType> faceList = thisBound->faceList;
      std::vector<vtkIdType> jnpts;
      structuredPairBoundaryT::boundaryT type = thisBound->boundaryType;
      for (unsigned face=0 ; face < faceList.size() ; face++ ) {
	int junction = (type==structuredPairBoundaryT::noncyclic) &&
	  !(face%(faceList.size()-1));
	int label =  1;
	polyData->GetCellPoints(faceList[face],
				numberOfPointsInCurrentCell, listOfPointIds);
	for (vtkIdType ii=0 ; ii<numberOfPointsInCurrentCell ; ii++) {
	  outputLabels->SetValue(listOfPointIds[ii], label);
	  if (junction) {
	    jnpts.push_back(listOfPointIds[ii]);
	  }
	  vtkIdType line[2];
	  line[0]=listOfPointIds[ii];
	  line[1]=listOfPointIds[(ii+1)%numberOfPointsInCurrentCell];
	  outputLines->InsertNextCell(2, line);
	}
	for (std::vector<vtkIdType>::iterator jnpt=jnpts.begin();
	     jnpt!=jnpts.end(); jnpt++){
	  outputLabels->SetValue(*jnpt, 2);
	}
      }
    }
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



sortedBoundariesT  buildBoundarySplineModel(vtkPolyData *polyData) {
    sortedBoundariesT sortedBoundaries = buildBoundaryModelKernel(polyData);
    polyData->BuildLinks(); // Will be needed by updatestucturedboundary

    for ( sortedBoundariesT::iterator thisBound = sortedBoundaries.begin();
	  thisBound != sortedBoundaries.end() ; thisBound++) {
      printf("Spline for pair %i %i\n",(thisBound->labels[0]),
	     (thisBound->labels[1]));
      thisBound->spline = splineFromBoundary(polyData,*thisBound);
    }

    sortedBoundariesT::junctionListT::iterator jn;
    for ( jn = sortedBoundaries.junctionList.begin();
	  jn !=sortedBoundaries.junctionList.end();  jn++){
      std::cout << "jn " << jn->first ;
      if ( jn->second.size() != 3 ) {
	std::cout << "Junction without exactly 3 associated boundaries:"
		  << std::endl;
      }
      for (unsigned jj=0; jj< jn->second.size(); jj++) {
	structuredPairBoundaryT bound =
	  sortedBoundaries.structuredBoundaries[jn->second[jj]];
	std::cout << " " << jn->second[jj] << "(" << bound.labels[0]
		  << "," << bound.labels[1] << ")" ;
      }
      std::cout<<std::endl;
    }


    bool savesplines=1;
    if (savesplines) saveSplinePolyData(polyData, sortedBoundaries, "testnewboundarymodel-splines.vtk" ,1);

    ////////Following is update boundary and repainting code that we don't want
    ////////to use yet.
    unsigned ii=0;
    for ( sortedBoundariesT::iterator thisBound = sortedBoundaries.begin();
      thisBound != sortedBoundaries.end() ; thisBound++) {
	  std::cout << "Updating bound " << (ii++) << " of " << sortedBoundaries.structuredBoundaries.size() << std::endl;
      updateStructuredBoundaryFaces( polyData, *thisBound );
      //// Not yet: updateStructuredBoundaryFaces( polyData, *thisBound );
    }
    std::cout<<"Finished boundary model splines"<<std::endl;
    return sortedBoundaries;
}


typedef std::vector<std::map<vtkIdType,double> > pointConnectivityMapT;
pointConnectivityMapT
buildPointConnectivity(vtkPolyData *polyData,
		       vtkCellArray *lines,
		       neighbourMapT neighbours,
		       double cutoffthreshold) {
  pointConnectivityMapT connections;
  //  pointConnectivityMapT *connections = new pointConnectivityMapT;
  unsigned numberofpoints = polyData->GetNumberOfPoints();
  connections.resize(numberofpoints);
 
  vtkIdType numberOfPointsInCell;
  vtkIdType *pointIdsInLine = new vtkIdType[2];  
  double totalWeight = 0;
    
  // Get each line, plot label value in labelImage, building up a histogram.
  lines->InitTraversal();
  unsigned secondcount=0;
  while(lines->GetNextCell(numberOfPointsInCell, pointIdsInLine))
    {
      vtkIdType p1, p2;
      double connectweight;
      p1 = pointIdsInLine[0];
      p2 = pointIdsInLine[1];
      for ( pointAndWeightVectorT::iterator p1neighbour =
	      neighbours[p1].begin();
	    p1neighbour != neighbours[p1].end() ; p1neighbour++ ) {
	for ( pointAndWeightVectorT::iterator p2neighbour =
		neighbours[p2].begin();
	      p2neighbour != neighbours[p2].end() ; p2neighbour++ ) {
	  vtkIdType point1 = p1neighbour->m_point;
	  vtkIdType point2 = p2neighbour->m_point;
      if (point1>=numberofpoints || point1<0) std::cout << "point1 outside range " << point1 << std::endl;
      if (point2>=numberofpoints || point2<0) std::cout << "point2 outside range " << point2 << std::endl;
	  connectweight = p1neighbour->m_weight * p2neighbour->m_weight;
	  if (connectweight > cutoffthreshold ) {
	    // Not sure if the find tests are strictly necessary or we
	    // can supply a default value for map entries.
	    pointConnectivityMapT::value_type::iterator connectfind;
	    connectfind = (connections)[point1].find(point2);
	    if ( connectfind != (connections)[point1].end() ) {
	      connectfind->second = connectfind->second + connectweight;
	      secondcount++;
	    } else {
	      (connections)[point1][point2] = connectweight;
	    }
	    connectfind = (connections)[point2].find(point1);
	    if ( connectfind != (connections)[point2].end() ) {
	      connectfind->second += connectfind->second + connectweight;
	    } else {
	      (connections)[point2][point1] = connectweight;
	    }
	  }
	}
      }
    }
  unsigned count=0;
  for (pointConnectivityMapT::iterator point=connections.begin() ;
       point !=connections.end(); point++ ) {
    count += point->size();
  }
  std::cout << "size overall " << connections.size() << std::endl;
  std::cout << "count " << count << std::endl;
  std::cout << "secondcount " << secondcount << std::endl;
  return connections;
}


void setupBoundaryStrip (
			 vtkPolyData *polyData,
			 structuredPairBoundaryT bound,
			 std::set<vtkIdType> &pointsOnBound,
			 std::vector<double> &edgeTs,
			 NonUniformBSpline::ControlPointListType &edgeVectors
) {
  int cyclic = bound.boundaryType == structuredPairBoundaryT::cyclic;
  for ( unsigned edgenum = 0; edgenum < bound.edges.size(); edgenum++){
    NonUniformBSpline::ControlPointType RtoLvec;
    RtoLvec.Fill(0);
    std::vector<vtkIdType> thisedge = bound.edges[edgenum];
    double pointleft[3];
    polyData->GetPoint(thisedge[0],pointleft);
    double pointright[3];
    polyData->GetPoint(thisedge[1],pointright);
    for (unsigned ii=0 ; ii<3 ; ii++ ){
      RtoLvec[ii] = pointleft[ii] - pointright[ii];
    }
    edgeVectors.push_back(RtoLvec);
    vtkIdType nextface=edgenum+1;
    double nextT;
    if (cyclic && nextface==bound.edges.size()) {
      nextT=1;
    }else{
      nextT=bound.tList[nextface];
    }
    edgeTs.push_back( (nextT + bound.tList[edgenum])/2 ); // Approximation
    pointsOnBound.insert(thisedge[0]);
    pointsOnBound.insert(thisedge[1]);
  }
  return;
}



void setupPointProfiles(
			vtkDataArray *scalarData,
			const  std::set<vtkIdType> &pointsOnBound,
			unsigned int labelImageXSize,
			pointConnectivityMapT &connections,
			std::map<vtkIdType,std::vector<double> > &pointProfile
) {
  for ( std::set<vtkIdType>::iterator thispoint = pointsOnBound.begin();
	thispoint!= pointsOnBound.end() ; thispoint++ ) {
    //std::cout << "Point" << std::endl;
    pointProfile[*thispoint].resize(labelImageXSize,0);
    double profileTot = 0;
    //std::cout << "Map size " <<  connections[*thispoint].size() << std::endl;
    //std::cout << "Start point " <<  *thispoint << std::endl;
    //std::cout << "First end point " <<  connections[*thispoint].begin()->first << std::endl;
    //std::cout << "Map size " <<  connections[*thispoint].size() << std::endl;
    for (std::map<vtkIdType,double>::iterator endpoint =
	   connections[*thispoint].begin();
	 endpoint != connections[*thispoint].end() ; endpoint++ ) {
      vtkIdType endlabel = (int)scalarData->GetTuple1(endpoint->first);
      pointProfile[*thispoint][endlabel] += endpoint->second;
      profileTot += endpoint->second;
    }
    for ( unsigned ii=0 ; ii< pointProfile[*thispoint].size(); ii++) {
      if (profileTot == 0) {
	// Unconnected point, can't use, and  pointProfile[*thispoint]
	profileTot = 1;
      }
      pointProfile[*thispoint][ii] /= profileTot;
    }
  }
  return;
}



std::vector<double>  labelProfileFromImage(
					   double *labelImageArray,
					   unsigned labelImageXSize,
					   unsigned labelImageSize,
					   vtkIdType label)

{
  double profiletotal=0;
  std::vector<double> profile(labelImageXSize,1);
  if ( (label+1) * labelImageXSize <= labelImageSize ) {
    for (unsigned ii=0; ii<labelImageXSize; ii++) {
      profile[ii] = labelImageArray[ii+label*labelImageXSize];
      profiletotal+=profile[ii];
    }
  } else {
    std::cout << "Error getting label " << label 
	      << " connectivity profile, from mean histogram,"
	      << " guessing flat, probably wrong." << std::endl;
    profiletotal=labelImageXSize;
  }
  if (profiletotal == 0 ) {
    //std::cout << "Region label " << label 
    //      << " is unconnected, using zero connectivity profile"
    //      << std::endl;
    for (unsigned ii=0; ii<labelImageXSize; ii++) {
      profile[ii] = 0.0; // Zero
      //profile[ii] = 1.0 / labelImageXSize;  // Flat
    }
  }
  else {
    for (unsigned ii=0; ii<labelImageXSize; ii++) {
      profile[ii] =profile[ii] / profiletotal;
    }
  }
  return profile;
}



double boundaryScore(std::vector<std::vector<vtkIdType> > edges,
			std::map<vtkIdType,std::vector<double> > pointProfile,
			std::vector<double> profileLeft,
			std::vector<double> profileRight)
{
  double score = 0;
  double scoreleft = 0;
  double scoreright = 0;
  for ( unsigned edgenum = 0; edgenum < edges.size(); edgenum++){
    std::vector<vtkIdType> thisedge = edges[edgenum];

      if ( totalArray(pointProfile[thisedge[0]].size(),
		      &pointProfile[thisedge[0]][0]) != 0 ) {
	if (pprofilemode) {
	  scoreleft  = computeProfilePScore(profileLeft.size(),
					    pointProfile[thisedge[0]],
					    profileLeft,
					    profileRight);
	}
	else {
	  scoreleft  = computeProfileSSDScore(profileLeft.size(),
					    pointProfile[thisedge[0]],
					    profileLeft,
					    profileRight);
	}
      }
      if ( totalArray(pointProfile[thisedge[0]].size(),
		      &pointProfile[thisedge[1]][0]) != 0 ) {
	if (pprofilemode) {
	  scoreright = computeProfilePScore(profileRight.size(),
					    pointProfile[thisedge[1]],
					    profileRight,
					    profileLeft);
	} else {
	  scoreright = computeProfileSSDScore(profileRight.size(),
					      pointProfile[thisedge[1]],
					      profileRight,
					      profileLeft);
	}
      }
      score += scoreleft+scoreright;
  }
  return score;
}


NonUniformBSpline::ControlPointListType
calculateBoundaryFlipVectors(
		     std::vector<std::vector<vtkIdType> > edges,
		     std::map<vtkIdType,std::vector<double> > pointProfile,
		     std::vector<double> profileLeft,
		     std::vector<double> profileRight,
		     NonUniformBSpline::ControlPointListType edgeVectors)
{
  NonUniformBSpline::ControlPointListType flipVectors;
  flipVectors = edgeVectors;
  for ( unsigned edgenum = 0; edgenum < edges.size(); edgenum++){
    std::vector<vtkIdType> thisedge = edges[edgenum];
    std::vector<int> flip(2,0);
    for (unsigned side=0; side <2; side++ ) { //left 0
      double scoreleft;
      double scoreright;
      if ( totalArray(pointProfile[thisedge[side]].size(),
		      &pointProfile[thisedge[side]][0]) == 0 ) {
	// Unconnected point, can't use.
	continue;
      }
      std::vector<double> profile = pointProfile[thisedge[side]];

      if(pprofilemode) {
	scoreleft  = computeProfilePScore(profileLeft.size(),
					  pointProfile[thisedge[side]],
					  profileLeft,
					  profileRight);
	// Strictly, expect -scoreleft for ProfileP, but might swap back ssd
	scoreright = computeProfilePScore(profileRight.size(),
					  pointProfile[thisedge[side]],
					  profileRight,
					  profileLeft);
      } else {
	scoreleft  = computeProfileSSDScore(profileLeft.size(),
					  pointProfile[thisedge[side]],
					  profileLeft,
					  profileRight);
	// Strictly, expect -scoreleft for ProfileP, but might swap back ssd
	scoreright = computeProfileSSDScore(profileRight.size(),
					  pointProfile[thisedge[side]],
					  profileRight,
					  profileLeft);
      }
      // More like right.
      if ( scoreleft > scoreright && side==0 ) flip[0] = 1;
      // More like left
      if ( scoreleft < scoreright && side==1 ) flip[1] = 1;
    }
    // If left needs flipped then boundary moves left
    for (unsigned ii=0; ii<3 ; ii++){
      flipVectors[edgenum][ii] = flipVectors[edgenum][ii] * (flip[0]-flip[1]);
    }
  }
  return flipVectors;
}



double updateBoundaryInternalCPs ( vtkPolyData *polyData,
				   vtkDataArray *scalarData,
				   sortedBoundariesT &splineModel,
				   unsigned boundn,
				   unsigned int labelImageXSize,
				   unsigned int labelImageSize,
				   double* labelImageArray,
				   pointConnectivityMapT &connections,
				   vtkCellArray *lines,
				   neighbourMapT neighbours,
				   double cutoffthreshold
)
{
  structuredPairBoundaryT &bound = splineModel.structuredBoundaries[boundn];
  double startscore, endscore, scorechange;
  double startscoreperedge, endscoreperedge, scorechangeperedge;
  bool doupdate=false;
  int cyclic = bound.boundaryType == structuredPairBoundaryT::cyclic;
  vtkIdType labelleft;
  vtkIdType labelright;
  std::set<vtkIdType> pointsOnBound;
  int order = bound.spline->GetSplineOrder();
  NonUniformBSpline::ControlPointListType cplist =
    bound.spline->GetControlPoints();
  NonUniformBSpline::ControlPointListType newcplist;
  NonUniformBSpline::ControlPointListType updateVectors;
  NonUniformBSpline::ControlPointListType edgeVectors;
  NonUniformBSpline::ControlPointListType flipVectors;

  std::vector<double> edgeTs;
  std::map<vtkIdType,std::vector<double> > pointProfile;

  newcplist.resize(cplist.size());
  updateVectors.resize(cplist.size());

  switch (bound.direction) {
  case LEFT1ST:
    labelleft=bound.labels[0];
    labelright=bound.labels[1];
    break;
  case RIGHT1ST:
    labelright=bound.labels[0];
    labelleft=bound.labels[1];
    break;
  default:
    std::cerr << "Couldn't update directionless boundary, " <<
      bound.labels[0] << ":" << bound.labels[1] << std::endl;
    return 0;
  }

  // Update faces list, updates t list and edges as well
  updateStructuredBoundaryFaces( polyData, bound );

  std::cout << "edgevectors" << std::endl;
  // Vectors, could save time by only doing them for edges which flip,
  // but this way at least indices correspond to 
  setupBoundaryStrip ( polyData,
		       bound,
		       pointsOnBound,
		       edgeTs,
		       edgeVectors);

  setupPointProfiles(
		     scalarData,
		     pointsOnBound,
		     labelImageXSize,
		     connections,
		     pointProfile
		     );

  // This is going to mask errors if sizes are different.
  std::vector<double> profileLeft = labelProfileFromImage (labelImageArray,
							   labelImageXSize,
							   labelImageSize,
							   labelleft);
  std::vector<double> profileRight = labelProfileFromImage (labelImageArray,
							    labelImageXSize,
							    labelImageSize,
							    labelright);
  if (edgescoremode) {
    startscore= boundaryScore(bound.edges,
			      pointProfile,
			      profileLeft,
			      profileRight);
  } else {
    paintSortedBoundariesToPolyData(polyData, splineModel, bound.labels);
    //paintSortedBoundariesToPolyData(polyData, splineModel);
    double testLabelImageArray[labelImageSize];
    startscore = computeSSDOfLines(labelImageSize, labelImageXSize, testLabelImageArray, labelImageArray, scalarData, lines, neighbours, cutoffthreshold);
  }


  startscoreperedge = startscore/edgeTs.size();
  // With profiles set up, calculate the flip vectors.
  flipVectors = calculateBoundaryFlipVectors(bound.edges,
					     pointProfile,
					     profileLeft,
					     profileRight,
					     edgeVectors);

  // Whole spline update, except for end control points on closed splines.
  // Single cp for duplicate knots => 1, not order for open splines
  unsigned skipend = cyclic ? 0 : 1 ;
  //  unsigned skipend = cyclic ? 0 : order ;
  for ( unsigned cpnum = skipend ; cpnum + skipend < cplist.size(); cpnum++){
    NonUniformBSpline::ControlPointType updateVector;
    updateVector.Fill(0);
    double totalweight=0;
    // Could do some extra work and only calculate over supported region.
    for ( unsigned edgenum = 0 ; edgenum < edgeTs.size(); edgenum++){
      double support = bound.spline->
	NonUniformBSplineFunctionRecursive(order, cpnum, edgeTs[edgenum]);
      totalweight += support;
      for (unsigned ii=0 ; ii<3; ii++ ) {
	//	updateVector[ii] += support*flipVectors[edgenum][ii];
	updateVector[ii] += support*support*flipVectors[edgenum][ii];
      }
    }
    for (unsigned ii=0 ; ii<3; ii++ ) {
      updateVectors[cpnum][ii] = updateVector[ii]/totalweight;
    }
  }

  double stepsize;
  for ( stepsize=INITSTEP; stepsize > lowerstepsize ;	stepsize = stepsize * 0.75){
    newcplist = cplist;
    for ( unsigned cpnum = skipend ; cpnum + skipend < cplist.size(); cpnum++){
      //            std::cout << newcplist[cpnum] << " :: " << updateVectors[cpnum] << " :: ";
      for (unsigned ii=0 ; ii<3; ii++) {
	newcplist[cpnum][ii] += updateVectors[cpnum][ii] * stepsize;
      }
      //            std::cout << newcplist[cpnum] << std::endl ;
    }

    bound.spline->SetControlPoints(newcplist);
    
    updateStructuredBoundaryFaces( polyData, bound );

    edgeTs.clear();
    edgeVectors.clear();

    setupBoundaryStrip ( polyData,
			 bound,
			 pointsOnBound,
			 edgeTs,
			 edgeVectors);
    if (edgescoremode) {
      pointsOnBound.clear();

      pointProfile.clear();
      setupPointProfiles(
			 scalarData,
			 pointsOnBound,
			 labelImageXSize,
			 connections,
			 pointProfile
			 );
      endscore= boundaryScore(bound.edges,
			      pointProfile,
			      profileLeft,
			      profileRight);
    } else {
      paintSortedBoundariesToPolyData(polyData, splineModel, bound.labels);
      //paintSortedBoundariesToPolyData(polyData, splineModel);

      double testLabelImageArray[labelImageSize];

      vtkDataArray *newscalarData = polyData->GetPointData()->GetScalars();
      endscore = computeSSDOfLines(labelImageSize, labelImageXSize, testLabelImageArray, labelImageArray, newscalarData, lines, neighbours, cutoffthreshold);
      //      newscalarData->Delete();
    }
    std::cout << "After repaint start " << startscore << " end "
	      << endscore <<std::endl;

    endscoreperedge = endscore/edgeTs.size();

    //accept the update whatever:      endscore=0;
    scorechangeperedge = endscoreperedge - startscoreperedge;
    scorechange = endscore - startscore;
    doupdate = peredgecost ? scorechangeperedge < 0 : scorechange < 0;
    if ( doupdate || endscore==startscore ) {
      std::cout << "Stopped step size " << stepsize << std::endl;
      break;
    }
  }

  std::cout << "CP update Start " << startscore << "(" <<startscoreperedge<<")"
	      << " End " << endscore << "(" <<endscoreperedge<<")"
	      << " step " << stepsize;
  if (!doupdate) {
    std::cout << "reset cps" <<std::endl;
    bound.spline->SetControlPoints(cplist);
    std::cout << "update faces" <<std::endl;
    updateStructuredBoundaryFaces( polyData, bound );
    if (!edgescoremode) {
      // Need to return to vanilla boundary.
      std::cout << "repaint bound" <<std::endl;

      paintSortedBoundariesToPolyData(polyData, splineModel, bound.labels);
      std::cout << "okay" <<std::endl;
      //paintSortedBoundariesToPolyData(polyData, splineModel);
    }
    std::cout << " didn't update";
  }
    std::cout << std::endl;

  return peredgecost ? scorechangeperedge : scorechange;
}



NonUniformBSpline::ControlPointListType
regularisejncps(NonUniformBSpline::ControlPointType oldjunction,
		NonUniformBSpline::ControlPointType newjunction,
		NonUniformBSpline::ControlPointType p0,
		NonUniformBSpline::ControlPointType p1,
		NonUniformBSpline::ControlPointType p2,
		double equilibdist=0) {
  //Not using equilibrium distance yet, needs to be calculated correctly, just
  // preserve current distanc for the moment.

  std::vector<NonUniformBSpline::ControlPointType> cps;
  std::vector<NonUniformBSpline::ControlPointType> cpfromjn;
  std::vector<NonUniformBSpline::ControlPointType> cpfromjnunit;
  std::vector<NonUniformBSpline::ControlPointType> cpstep;
  NonUniformBSpline::ControlPointType step;
  std::vector<double> mag;

  cps.push_back(p0);  
  cps.push_back(p1);  
  cps.push_back(p2);

  cpfromjn=cps;
  mag.resize(cps.size());
  cpfromjnunit.resize(cps.size());
  cpstep.resize(cps.size());
  step.Fill(0);

  NonUniformBSpline::ControlPointType midjunction;
  bool stepadjust=1;

  if ( stepadjust ) {
    //Using oldjunction as the midpoint doesn't work as the centring needs more
    //careful handling than this, try midpoint instead.
    midjunction.Fill(0);
    for (unsigned ii=0; ii< cpfromjn.size(); ii++) {
      for (unsigned el=0 ; el<3 ; el++) {
	midjunction[el] += cps[ii][el] / cpfromjn.size();
      }
    }

    std::cout << midjunction << std::endl;

    for (unsigned ii=0; ii< cpfromjn.size(); ii++) {
      mag[ii]=0;
      for (unsigned el=0 ; el<3 ; el++) {
	cpfromjn[ii][el] -= midjunction[el];
	mag[ii] += cpfromjn[ii][el]*cpfromjn[ii][el];
      }
      mag[ii] = sqrt(mag[ii]);
      std::cout << "mag[" <<ii<<"] " << mag[ii] << std::endl;
      std::cout << "cpfromjnunit";
      for (unsigned el=0 ; el<3 ; el++) {
	cpfromjnunit[ii][el] = mag[ii] ? cpfromjn[ii][el]/mag[ii] : 0;
	std::cout << " " << cpfromjnunit[ii][el];
	step[el] -= cpfromjnunit[ii][el] / cpfromjn.size();
      }
      std::cout << std::endl;
    }
    std::cout << "step";
    for (unsigned el=0 ; el<3 ; el++) {
      std::cout << " " << step[el];
    }
    std::cout << std::endl;
    for (unsigned ii=0; ii< cpfromjn.size(); ii++) {
      double magtmp=0;
      std::cout << "cpfromjnunitpreadd";
      for (unsigned el=0 ; el<3 ; el++) {
	std::cout << " " << cpfromjnunit[ii][el];
      }
      std::cout << std::endl;
      std::cout << "cpfromjnunitpostadd";
      for (unsigned el=0 ; el<3 ; el++) {
	cpfromjnunit[ii][el] = cpfromjnunit[ii][el] + step[el];
	std::cout << " " << cpfromjnunit[ii][el];
	magtmp += cpfromjnunit[ii][el]*cpfromjnunit[ii][el];
      }
      std::cout << std::endl;
      magtmp = sqrt(magtmp);
      std::cout << "magtmp[" <<ii<<"] " << magtmp << std::endl;
      if (magtmp) {
	std::cout << "cpfromjnunit";
	for (unsigned el=0 ; el<3 ; el++) {
	  // Don't use magtmp ? X : 0 as we don't want to land points on origin
	  cpfromjnunit[ii][el] = cpfromjnunit[ii][el] / magtmp;
	  std::cout << " " << cpfromjnunit[ii][el];
	  cpfromjn[ii][el] = cpfromjnunit[ii][el]*mag[ii] ;
	  cpstep[ii][el] = newjunction[el]+cpfromjn[ii][el] - cps[ii][el];
	}
	std::cout << std::endl;
      }
    }
  }
  else {
    for (unsigned ii=0; ii< cpfromjn.size(); ii++) {
	for (unsigned el=0 ; el<3 ; el++) {
	  cpstep[ii][el] = newjunction[el]-oldjunction[el];
	}
    }
  }
  return cpstep;
}



double updateBoundaryJunctionCPs ( vtkPolyData *polyData,
				   vtkDataArray *scalarData,
				   sortedBoundariesT &bounds,
				   vtkIdType junctionId,
				   unsigned int labelImageXSize,
				   unsigned int labelImageSize,
				   double* labelImageArray,
				   pointConnectivityMapT &connections,
				   vtkCellArray *lines,
				   neighbourMapT neighbours,
				   double cutoffthreshold
)
{
  std::vector<NonUniformBSpline::ControlPointListType> cplistlist;
  std::vector<NonUniformBSpline::ControlPointListType> newcplistlist;
  std::vector<NonUniformBSpline::ControlPointListType> edgesVectors;
  std::vector<std::vector<vtkIdType> > labelsleftright;

  std::vector<std::set<vtkIdType> > pointsOnBounds;
  std::set<vtkIdType> pointsOnAllBounds;

  std::vector<std::vector<double> > edgesTs;
  std::set<vtkIdType> jnlabelsset;
  std::vector<vtkIdType> jnlabels;

  typedef std::map<vtkIdType,std::vector<double> > IDProfileMapT;

  IDProfileMapT pointProfile;
  IDProfileMapT profileForLabel;

  double startscore, endscore, scorechange;
  double startscoreperedge, endscoreperedge, scorechangeperedge;
  bool doupdate=false;
  std::vector<vtkIdType> boundaries = bounds.junctionList[junctionId];

  if ( boundaries.size() != 3 ) {
    std::cerr << "Junction id " << junctionId
	      << " in list without 3 attached boundaries" << std::endl;
    return 0;
  }

  std::vector<unsigned> orders;
  std::vector<std::vector <int> > endcps;
  bool problembound=false;

  // List the involved control points, do building steps.
  labelsleftright.resize(boundaries.size());
  edgesVectors.resize(boundaries.size());
  edgesTs.resize(boundaries.size());
  pointsOnBounds.resize(boundaries.size());
  for (unsigned ii=0 ; ii<boundaries.size(); ii++) {
    vtkIdType boundnum=boundaries[ii];
    structuredPairBoundaryT &bound = bounds.structuredBoundaries[boundnum];
    orders.push_back(bound.spline->GetSplineOrder());
    cplistlist.push_back(bound.spline->GetControlPoints());
    if ( bound.boundaryType == structuredPairBoundaryT::cyclic ) {
      problembound = true;
      std::cerr << "Junction point " << junctionId
		<< " references cyclic boundary" <<std::endl;
    } else {
      if (bound.junctions[0] == junctionId) {
	std::vector<int> endcplist;
	// Single cp for duplicate knots, need the extra cp for jn regularisation
	// Single cp for duplicate knots, but need all for smoothing
	//	for (unsigned pointn=0; pointn < 2; pointn++) {
	for (unsigned pointn=0; pointn < orders[ii]; pointn++) {
	  endcplist.push_back(pointn);
	}
	endcps.push_back(endcplist);
      }
      else if (bound.junctions[1] == junctionId) {
	std::vector<int> endcplist;
	unsigned cpcount = cplistlist[ii].size();
	// Single cp for duplicate knots, but need all for smoothing
	//	for (unsigned pointn=0; pointn < 2; pointn++) {
	for (unsigned pointn=0; pointn < orders[ii]; pointn++) {
	  endcplist.push_back(cpcount-1-pointn);
	}
	endcps.push_back(endcplist);
      }
      else {
	problembound = true;
	std::cerr << "Junction point " << junctionId
		  << " reference not returned by boundary, has "
		  << bound.junctions[0] << " and " << bound.junctions[1]
		  << std::endl;
      }
    }
    jnlabelsset.insert(bound.labels[0]);
    jnlabelsset.insert(bound.labels[1]);
    switch (bound.direction) {
    case LEFT1ST:
      labelsleftright[ii][0]=bound.labels[0];
      labelsleftright[ii][1]=bound.labels[1];
      break;
    case RIGHT1ST:
      labelsleftright[ii][1]=bound.labels[0];
      labelsleftright[ii][0]=bound.labels[1];
      break;
    default:
      std::cerr << "Couldn't update directionless boundary, "
		<< bound.labels[0] << ":"
		<< bound.labels[1] << std::endl;
      problembound=true;
    }
    setupBoundaryStrip ( polyData,
			 bound,
			 pointsOnBounds[ii],
			 edgesTs[ii],
			 edgesVectors[ii]);
    // For either side find the label profile if we don't already have it
    for (unsigned side=0; side<2; side++) {
      IDProfileMapT::iterator profilemaplookup =
	profileForLabel.find(bound.labels[side]);
      if (profilemaplookup == profileForLabel.end()) {
	profileForLabel[bound.labels[side]] =
	  labelProfileFromImage (labelImageArray,
				 labelImageXSize,
				 labelImageSize,
				 bound.labels[side]);
      }
    }
  }
  if (problembound) {
    return 0;
  }

  for (std::set<vtkIdType>::iterator jnlabelit=jnlabelsset.begin();
       jnlabelit != jnlabelsset.end(); jnlabelit++ ) {
    jnlabels.push_back(*jnlabelit);
  }


  for (unsigned ii=0; ii<pointsOnBounds.size(); ii++) {
    pointsOnAllBounds.insert(pointsOnBounds[ii].begin(),
			     pointsOnBounds[ii].end());
  }
  setupPointProfiles(
		     scalarData,
		     pointsOnAllBounds,
		     labelImageXSize,
		     connections,
		     pointProfile
		     );


  // Two loops:
  // Separately calculate the weighted total update vector for each
  // boundary at this junction and track the total weight. Calculate the 
  // overall step vector (which applies to all the duplicate points at this
  // junction). With this in posession:

  // Calculate the updated point (there's only one) and apply to all duplicate
  // points. Check for improvement, reducing step size if necessary.
  NonUniformBSpline::ControlPointType updateVector;
  updateVector.Fill(0);
  double updateVectorTotalWeight = 0;

  startscore=0;
  startscoreperedge=0;

  if (!edgescoremode) {
    paintSortedBoundariesToPolyData(polyData, bounds, jnlabels);
    //paintSortedBoundariesToPolyData(polyData, bounds);
    double testLabelImageArray[labelImageSize];
    startscore = computeSSDOfLines(labelImageSize, labelImageXSize, testLabelImageArray, labelImageArray, scalarData, lines, neighbours, cutoffthreshold);
  }


  for (unsigned ii=0; ii<boundaries.size(); ii++) { 
    vtkIdType boundnum = boundaries[ii];
    structuredPairBoundaryT &bound = bounds.structuredBoundaries[boundnum];
    double thisscore;
    if (edgescoremode) {
      thisscore = boundaryScore(bound.edges,
				pointProfile,
				profileForLabel[labelsleftright[ii][0]],
				profileForLabel[labelsleftright[ii][1]]
				);
      startscoreperedge += thisscore/edgesTs[ii].size();
      startscore += thisscore;
    }
    NonUniformBSpline::ControlPointListType flipVectors;
    flipVectors = calculateBoundaryFlipVectors(bound.edges,
                                   pointProfile,
				   profileForLabel[labelsleftright[ii][0]],
				   profileForLabel[labelsleftright[ii][1]],
				   edgesVectors[ii]);
    for ( unsigned edgenum=0 ; edgenum < edgesTs[ii].size(); edgenum++) {
      double edgeTotalWeight=0;
      NonUniformBSpline::ControlPointType edgeInfluenceVector;
      edgeInfluenceVector.Fill(0);
      //      for (unsigned cpii=0 ; cpii<endcps[ii].size(); cpii++) {
      // Justs the one cp
      for (unsigned cpii=0 ; cpii<1; cpii++) {
	vtkIdType cpnum=endcps[ii][cpii];
	double support;
        support = bound.spline->
	  NonUniformBSplineFunctionRecursive(orders[ii],
					     cpnum,
					     edgesTs[ii][edgenum]);
	edgeTotalWeight += support;
	for (unsigned el=0 ; el<3; el++ ) {
	  edgeInfluenceVector[el] +=
	    support*support*flipVectors[edgenum][el];
	}
      }
      for (unsigned el=0 ; edgeTotalWeight !=0 && el<3; el++ ) {
	updateVector[el] += edgeInfluenceVector[el]/edgeTotalWeight;
      }
      updateVectorTotalWeight += edgeTotalWeight;
    }
  }  
  for (unsigned el=0 ; updateVectorTotalWeight !=0 && el<3; el++ ) {
    updateVector[el] = updateVector[el]/updateVectorTotalWeight;
  }


  double stepsize;
  for ( stepsize=INITSTEP; stepsize > lowerstepsize ;	stepsize = stepsize * 0.75){

    newcplistlist = cplistlist;
    for (unsigned ii=0; ii<boundaries.size(); ii++) {
      vtkIdType boundnum = boundaries[ii];
      structuredPairBoundaryT &bound = bounds.structuredBoundaries[boundnum];
      //for (unsigned cpii=0 ; cpii<endcps[ii].size(); cpii++) {
      // Just the one cp to update
      for (unsigned cpii=0 ; cpii<1; cpii++) {
	unsigned cpnum=endcps[ii][cpii];
	for (unsigned el=0 ; el<3; el++ ) {
	  newcplistlist[ii][cpnum][el] += updateVector[el] * stepsize;
	}
      }
      bound.spline->SetControlPoints(newcplistlist[ii]);
    
      updateStructuredBoundaryFaces( polyData, bound );

      pointsOnBounds[ii].clear();
      edgesTs[ii].clear();
      edgesVectors[ii].clear();
      setupBoundaryStrip ( polyData,
			   bound,
			   pointsOnBounds[ii],
			   edgesTs[ii],
			   edgesVectors[ii]);
    }
    pointsOnAllBounds.clear();
    for (unsigned ii=0; ii<boundaries.size(); ii++) {
      pointsOnAllBounds.insert(pointsOnBounds[ii].begin(),
			       pointsOnBounds[ii].end());
    }
    pointProfile.clear();
    setupPointProfiles(
		       scalarData,
		       pointsOnAllBounds,
		       labelImageXSize,
		       connections,
		       pointProfile
		       );
    endscore=0;
    endscoreperedge=0;
    if (edgescoremode) {
      for (unsigned ii=0; ii<boundaries.size(); ii++) {
	vtkIdType boundnum = boundaries[ii];
	structuredPairBoundaryT &bound = bounds.structuredBoundaries[boundnum];
	double thisscore;
	thisscore = boundaryScore(bound.edges,
				  pointProfile,
				  profileForLabel[labelsleftright[ii][0]],
				  profileForLabel[labelsleftright[ii][1]]
				  );
	endscore += thisscore;
	endscoreperedge += thisscore/edgesTs[ii].size();
      }
    } else {
      paintSortedBoundariesToPolyData(polyData, bounds, jnlabels);
      //paintSortedBoundariesToPolyData(polyData, bounds);
      double testLabelImageArray[labelImageSize];
      vtkDataArray *newscalarData = polyData->GetPointData()->GetScalars();
      endscore = computeSSDOfLines(labelImageSize, labelImageXSize, testLabelImageArray, labelImageArray, newscalarData, lines, neighbours, cutoffthreshold);
      //      newscalarData->Delete();
      std::cout << "After repaint start " << startscore << " end "
		<< endscore <<std::endl;
    }

    scorechange = endscore - startscore;
    scorechangeperedge = endscoreperedge - startscoreperedge;
    doupdate = peredgecost ? scorechangeperedge < 0 : scorechange < 0;
    if ( doupdate  || endscore==startscore ) {
      std::cout << "Stopped step size " << stepsize << std::endl;
      break;
    }
  }

  // Regularise the junction to stop folding.
  bool regularise=0;
  if(regularise) {
    std::cout << "Points before reg" << std::endl;
    std::cout << cplistlist[0][endcps[0][0]]  << std::endl
	      << newcplistlist[0][endcps[0][0]] << std::endl
	      << newcplistlist[0][endcps[0][1]] << std::endl
	      << newcplistlist[1][endcps[1][1]] << std::endl
	      << newcplistlist[2][endcps[2][1]] << std::endl;
    
    NonUniformBSpline::ControlPointListType updatesteps;
    updatesteps=  regularisejncps(cplistlist[0][endcps[0][0]],
				  newcplistlist[0][endcps[0][0]],
				  newcplistlist[0][endcps[0][1]],
				  newcplistlist[1][endcps[1][1]],
				  newcplistlist[2][endcps[2][1]]);
    
    endscore=0;
    endscoreperedge=0;
    for (unsigned ii=0; ii<boundaries.size(); ii++) {
      vtkIdType boundnum = boundaries[ii];
      structuredPairBoundaryT &bound = bounds.structuredBoundaries[boundnum];
      for (unsigned cpnum=1; cpnum<endcps[ii].size(); cpnum++) {
	std::cout << "step for " << ii << " cpnum " << endcps[ii][cpnum] << "," ;
	for (unsigned el=0 ; el<3 ; el++) {
	  std::cout << " " << (1-cpnum/(double)(endcps[ii].size()))*updatesteps[ii][el];
	  newcplistlist[ii][endcps[ii][cpnum]][el] +=
	    (1-cpnum/(double)(endcps[ii].size()-1))*updatesteps[ii][el];
	}
	std::cout << std::endl;
      }
      
      bound.spline->SetControlPoints(newcplistlist[ii]);    
      updateStructuredBoundaryFaces( polyData, bound );
      if (edgescoremode) {
	double thisscore;
	thisscore = boundaryScore(bound.edges,
				  pointProfile,
				  profileForLabel[labelsleftright[ii][0]],
				  profileForLabel[labelsleftright[ii][1]]
				  );
	endscore += thisscore;
	endscoreperedge += thisscore/edgesTs[ii].size();
      }
    }
    if (!edgescoremode) {
      paintSortedBoundariesToPolyData(polyData, bounds, jnlabels);
      //paintSortedBoundariesToPolyData(polyData, bounds);
      double testLabelImageArray[labelImageSize];
      vtkDataArray *newscalarData = polyData->GetPointData()->GetScalars();
      endscore = computeSSDOfLines(labelImageSize, labelImageXSize, testLabelImageArray, labelImageArray, newscalarData, lines, neighbours, cutoffthreshold);
      //      newscalarData->Delete();
      std::cout << "After repaint start " << startscore << " end "
		<< endscore <<std::endl;
    }

    scorechange = endscore - startscore;
    scorechangeperedge = endscoreperedge - startscoreperedge;
    doupdate = peredgecost ? scorechangeperedge < 0 : scorechange < 0;
    std::cout << "Points after reg" << std::endl;
    std::cout << cplistlist[0][endcps[0][0]]  << std::endl
	      << newcplistlist[0][endcps[0][0]] << std::endl
	      << newcplistlist[0][endcps[0][1]] << std::endl
	      << newcplistlist[1][endcps[1][1]] << std::endl
	      << newcplistlist[2][endcps[2][1]] << std::endl;
  }

  // Should re-calc in case regularised. Regularisation above doesn't do it yet though.
  scorechange = endscore - startscore;
  scorechangeperedge = endscoreperedge - startscoreperedge;
  doupdate = peredgecost ? scorechangeperedge < 0 : scorechange < 0;

  std::cout << "JNCP update Start " << startscore <<"("<<startscoreperedge<<")"
	    << " End " << endscore <<"("<<endscoreperedge<<")"
	    << " step " << stepsize;
  if (!doupdate) {
    for (unsigned ii=0; ii<boundaries.size(); ii++) {
      vtkIdType boundnum=boundaries[ii];
      structuredPairBoundaryT &bound = bounds.structuredBoundaries[boundnum];
      bound.spline->SetControlPoints(cplistlist[ii]);
      updateStructuredBoundaryFaces( polyData, bound );
      if (!edgescoremode) {
	//paintSortedBoundariesToPolyData(polyData, bounds);
	paintSortedBoundariesToPolyData(polyData, bounds, jnlabels);
      }
    }
    std::cout << " didn't update";
  }
  std::cout << std::endl;

  return peredgecost ? scorechangeperedge : scorechange;
}


void smoothBoundaryCPChange (structuredPairBoundaryT oldbound,
			     structuredPairBoundaryT &newbound) {
  NonUniformBSpline::ControlPointListType cpchange, smoothcpchange,
    oldcp, newcp;
  bool cyclic = newbound.boundaryType == structuredPairBoundaryT::cyclic;
  //Approx Gaussian with 1 sd at adjacent cps.
  std::vector<double> weights;
  weights.push_back(0.367879441171442334);
  weights.push_back(1.0);
  weights.push_back(0.367879441171442334);
  double totalweight=weights[0]+weights[1]+weights[2];
  int skip = cyclic ? 0 : 1;
  oldcp = oldbound.spline->GetControlPoints();
  newcp = newbound.spline->GetControlPoints();
  cpchange.resize(oldcp.size());
  smoothcpchange.resize(oldcp.size());
  for (unsigned ii=0; ii< oldcp.size() ; ii++){
    
    for (unsigned el=0 ; el<3; el++){
      cpchange[ii][el] = newcp[ii][el]-oldcp[ii][el];
    }
    smoothcpchange[ii].Fill(0);
  }
  for ( unsigned ii=skip; ii+skip<cpchange.size(); ii++ ) {
    for (unsigned el=0 ; el<3; el++){
      smoothcpchange[ii][el] = (weights[0] * cpchange[ii-1][el] +
				weights[1] * cpchange[ii][el] +
				weights[2] * cpchange[ii+1][el]) / totalweight;
      newcp[ii][el] = oldcp[ii][el] + smoothcpchange[ii][el];
    }
  }
  newbound.spline->SetControlPoints(newcp);
  return;
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
      std::cout << "Set -i=" >> args.surfaceDataFile;
    }
    else if(strcmp(argv[i], "-m") == 0){
      args.meanMatrixFile=argv[++i];
      std::cout << "Set -o=" >> args.meanMatrixFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputSurfaceFile=argv[++i];
      std::cout << "Set -o=" >> args.outputSurfaceFile;
    }   
    else if(strcmp(argv[i], "-nl") == 0){
      args.numberOfLabels=atoi(argv[++i]);
      std::cout << "Set -nl=" >> ucltk::ConvertToString(args.numberOfLabels);
    }
    else if(strcmp(argv[i], "-ni") == 0){
      args.numberOfIterations=atoi(argv[++i]);
      std::cout << "Set -ni=" >> ucltk::ConvertToString(args.numberOfIterations);
    }        
    else if(strcmp(argv[i], "-si") == 0){
      args.numberOfSmoothingIterations=atoi(argv[++i]);
      std::cout << "Set -si=" >> ucltk::ConvertToString(args.numberOfIterations);
    }        
    else if(strcmp(argv[i], "-ga") == 0){
      args.gamma=atof(argv[++i]);
      std::cout << "Set -ga=" >> ucltk::ConvertToString(args.gamma);
    }
    else if(strcmp(argv[i], "-ss") == 0){
      args.stepSize=atof(argv[++i]);
      std::cout << "Set -ss=" >> ucltk::ConvertToString(args.stepSize);
    }
    else if(strcmp(argv[i], "-averageVec") == 0){
      args.useVectorAveraging=true;
      std::cout << "Set -averageVec=" >> ucltk::ConvertToString(args.useVectorAveraging);
    }
    else if(strcmp(argv[i], "-forceNoConnections") == 0){
      args.forceNoConnections=true;
      std::cout << "Set -forceNoConnections=" >> ucltk::ConvertToString(args.forceNoConnections);
    }
    else if(strcmp(argv[i], "-forceBoundary") == 0){
      args.forceBoundary=true;
      std::cout << "Set -forceBoundary=" >> ucltk::ConvertToString(args.forceBoundary);
    }
    else if(strcmp(argv[i], "-testLabel") == 0){
      args.testLabel=atoi(argv[++i]);
      allLabels = false;
      std::cout << "Set -testLabel=" >> ucltk::ConvertToString(args.testLabel);
    }
    else if(strcmp(argv[i], "-smooth") == 0){
      args.smooth=true;
      char *argopt = argv[++i];
      double argoptf;
      std::cout << "Set -smooth=" >> ucltk::ConvertToString(argopt);
      if ( sscanf(argopt,"%lf",&argoptf) != 1 ) {
	std::cerr << argv[0] << ":\tParameter " << argopt << " not allowed." << std::endl;
	return -1;
      }
      args.smoothvariance = argoptf * argoptf;
      args.smoothcutoff = CUTOFFRANGE * argoptf;
    }    
    else if(strcmp(argv[i], "-freesurfercheat") == 0){
      freesurfercheat=true;
      std::cout << "Set -freesurfercheat";
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
  vtkSmartPointer<vtkPolyDataReader> surfaceReader = vtkPolyDataReader::New();
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
  if (freesurfercheat) {
    std::cout <<"WARNING: running with freesurfer cheat mode for dual spheres"
	      <<std::endl;
  }
  if (peredgecost) {
    std::cout << "Using boundary cost per edge face" << std::endl;
  } else {
    std::cout << "Using total boundary cost (not per edge face)" << std::endl;
  }
  if (repaintateachupdate) {
    std::cout << "Repainting labels after each boundary update" << std::endl;
  } else {
    std::cout << "Repainting labels after all boundaries updated" << std::endl;
  }
  std::cout << "INITSTEP (faces): " << INITSTEP << std::endl;

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
  std::cout << "Declare map" << std::endl;
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
  std::cout << "Declare writer" << std::endl;
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
  
  // Build cells and links for quick lookup
  std::cout<<"Getting lines" << std::endl;
  vtkCellArray *lines = polyData->GetLines();
   std::cout << "Build cells" << std::endl;
   polyData->BuildCells();
   std::cout << "Build links2" << std::endl;
  polyData->BuildLinks();

  std::cout << "New arrays" << std::endl;
 
  // Create a mean label image and a test label image
  double* meanLabelImageArray = new double[arraySize];
  double* testLabelImageArray = new double[arraySize];
  double* nextLabelImageArray = new double[arraySize];
  
  double bestScore = 0;
  int bestLabel = 0;
  vtkIdType bestPointNumber = 0;
  
  // Copy mean image into its array, and initialize testLabelImageArray.
  std::cout << "Loading mean image" << std::endl;
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

  // Work out total number of label values
  std::cout << "Count labels in vtk" << std::endl;
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

  // Spline update loop
#if 1
  scalarData = (polyData->GetPointData()->GetScalars());
  double startScore = computeSSDOfLines(arraySize, meanLabelImageSize[0], testLabelImageArray, meanLabelImageArray, scalarData, lines, neighbours, cutoffthreshold);
  std::cout << "[Start]: score =" << startScore << std::endl;


  std::cout << "Building neighbours list" <<  std::endl;
  buildFaceNeighboursList (polyData);

      std::cout << "Building connections map" <<  std::endl;
      
      pointConnectivityMapT connectivitymap;
      connectivitymap = buildPointConnectivity(polyData, lines, neighbours,
					       cutoffthreshold);
      
      cleanLabels(polyData);

      sortedBoundariesT splineModel =  buildBoundarySplineModel(polyData);
      paintSortedBoundariesToPolyData(polyData, splineModel);
      saveAnyPolyData(polyData,"repaint1.vtk");
      scalarData = (polyData->GetPointData()->GetScalars());
      double currentScore = computeSSDOfLines(arraySize, meanLabelImageSize[0], testLabelImageArray, meanLabelImageArray, scalarData, lines, neighbours, cutoffthreshold);

      std::cout << "[Spline]: score =" << currentScore << std::endl;
      bool saveupdatedboundaries=1;
      if (saveupdatedboundaries) saveBoundaryPolyData(polyData, splineModel);
      bool saverepainted=1;
      if(saverepainted) saveAnyPolyData(polyData,"repainted.vtk");

      sortedBoundariesT origSplineModel = splineModel;
  do
    {
      atLeastOneLabelImproved = false;
      totalNumberChangedPerIteration = 0;
      
      // Extract boundary model.
      //      boundaryModel =  extractBoundaryModel(polyData);

      double scorechange=0;

      sortedBoundariesT oldSplineModel = splineModel;
      for (sortedBoundariesT::junctionListT::iterator jn =
	     splineModel.junctionList.begin();
	   jn != splineModel.junctionList.end(); jn++) {
	//std::cout << "jn update " << jn->first << std::endl;
	scorechange +=
	  updateBoundaryJunctionCPs (polyData, scalarData, splineModel,
				     jn->first,
				     meanLabelImageSize[0], arraySize,
				     meanLabelImageArray,
				     connectivitymap,
				     lines,
				     neighbours,
				     cutoffthreshold );
	if(repaintateachupdate) {
	  paintSortedBoundariesToPolyData(polyData, splineModel);
	}
      {
	std::cout << "Trying point 89646" << std::endl;
	vtkIdType *listOfPointIds;
	vtkIdType numberOfPointsInCurrentCell;
	polyData->GetCellPoints(89646,
				numberOfPointsInCurrentCell, listOfPointIds);
	std::cout << "got " << numberOfPointsInCurrentCell << std::endl;
      }
      }
      //      for (sortedBoundariesT::iterator bound = splineModel.begin();
      //	   bound != splineModel.end() ; bound++ ) {
      for (unsigned boundn=0 ; boundn<splineModel.structuredBoundaries.size();
	   boundn++) {
	scorechange +=
	  //	  updateBoundaryInternalCPs ( polyData, scalarData, *bound,
	  updateBoundaryInternalCPs ( polyData, scalarData, splineModel,
				      boundn,
				      meanLabelImageSize[0], arraySize,
				      meanLabelImageArray,
				      connectivitymap,
				      lines,
				      neighbours,
				      cutoffthreshold);
	std::cout << "Doing repaint" << std::endl;
	if(repaintateachupdate) {
	  paintSortedBoundariesToPolyData(polyData, splineModel);
	}
	std::cout << "CP update repaint done" << std::endl;
      {
	std::cout << "Trying point 89646" << std::endl;
	vtkIdType *listOfPointIds;
	vtkIdType numberOfPointsInCurrentCell;
	polyData->GetCellPoints(89646,
				numberOfPointsInCurrentCell, listOfPointIds);
	std::cout << "got " << numberOfPointsInCurrentCell << std::endl;
      }
      }
      bool smoothcpchange = 1;
      std::cout << "Starting smooth" << std::endl;
      for (unsigned ii=0; smoothcpchange &&
	     ii< splineModel.structuredBoundaries.size(); ii++) {
	std::cout << "smooth " << ii+1 << "/" << splineModel.structuredBoundaries.size() << std::endl;
	smoothBoundaryCPChange ( origSplineModel.structuredBoundaries[ii],
				 splineModel.structuredBoundaries[ii]);
      {
	std::cout << "Trying point 89646" << std::endl;
	vtkIdType *listOfPointIds;
	vtkIdType numberOfPointsInCurrentCell;
	polyData->GetCellPoints(89646,
				numberOfPointsInCurrentCell, listOfPointIds);
	std::cout << "got " << numberOfPointsInCurrentCell << std::endl;
      }
      }
      std::cout << "Smooth done" << std::endl;

      paintSortedBoundariesToPolyData(polyData, splineModel);

      std::cout << "saving boundaries" << std::endl;
      if (saveupdatedboundaries) saveBoundaryPolyData(polyData, splineModel);
      std::cout << "saving repainted" << std::endl;
      if(saverepainted) saveAnyPolyData(polyData,"repainted.vtk");
      bool savesplines=1;
      std::cout << "saving boundary splines" << std::endl;
      if (savesplines) saveSplinePolyData(polyData, splineModel, "testnewboundarymodel-splines-update.vtk", 1);

      std::cout << "computing ssd" << std::endl;
      currentScore = computeSSDOfLines(arraySize, meanLabelImageSize[0], testLabelImageArray, meanLabelImageArray, scalarData, lines, neighbours, cutoffthreshold);
      std::cout << "[" << iterationNumber << "]:After update score =" << currentScore << std::endl;
      iterationNumber++;
      
      atLeastOneLabelImproved = scorechange < -1e-5 * startScore;
      std::cout << "change " << scorechange << " " << atLeastOneLabelImproved << " " << (iterationNumber < args.numberOfIterations) << std::endl;
    } while (atLeastOneLabelImproved && iterationNumber < args.numberOfIterations);
      //Spline update loop
#endif

  //Matt's update loop
#if 0
  do
    {
      atLeastOneLabelImproved = false;
      totalNumberChangedPerIteration = 0;
      
      // Extract boundary model.
      //      boundaryModel =  extractBoundaryModel(polyData);
      boundaryModel =  buildBoundaryModel(polyData);
  
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
  //Matt's update loop end
#endif    
  // Write out the resulting file.
  writer->SetFileName(args.outputSurfaceFile.c_str());
  writer->SetInput(polyData);
  writer->Update();

  return EXIT_SUCCESS;
  
}
