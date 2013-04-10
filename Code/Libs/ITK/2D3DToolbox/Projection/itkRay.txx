/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRay_txx
#define __itkRay_txx

#include <iomanip>

#include "itkRay.h"

//#define DEEP_DEBUG_RAY
//#define DEBUG_RAY


using namespace std;

namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
Ray<TInputImage, TCoordRep>
::Ray()
{
  ZeroState();
}


/* -----------------------------------------------------------------------
   ZeroState() - Set the default (zero) state of the object
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::ZeroState()
{
  int i;

  m_ValidRay = false;

  m_RayPosition2Dmm[0] = 0.;
  m_RayPosition2Dmm[1] = 0.;

  m_ProjectionResolution2Dmm[0] = 1.;
  m_ProjectionResolution2Dmm[1] = 1.;

  m_NumberOfVoxelsInX = 0;
  m_NumberOfVoxelsInY = 0;
  m_NumberOfVoxelsInZ = 0;

  m_VoxelDimensionInX = 0;
  m_VoxelDimensionInY = 0;
  m_VoxelDimensionInZ = 0;

  for (i=0; i<3; i++) 

    m_CurrentRayPositionInMM[i] = 
      m_RayDirectionInMM[i] =
      m_RayVoxelStartPosition[i] =
      m_RayVoxelEndPosition[i] = 
      m_VoxelIncrement[i] =
      m_RayIntersectionVoxelIndex[i] = 0;

  m_TraversalDirection = UNDEFINED_DIRECTION;
  m_TotalRayVoxelPlanes = 0;
  m_NumVoxelPlanesTraversed = -1;

  for (i=0; i<4; i++)
    m_RayIntersectionVoxels[i] = 0;

  m_ProjectionMatrix.SetIdentity();
}


/* -----------------------------------------------------------------------
   Initialise() - Initialise the object
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::Initialise(void)
{
#ifdef DEEP_DEBUG_RAY
  std::cout << std::endl << "Initialise()" << std::endl;
#endif

  // Save the dimensions of the volume and calculate the bounding box
  this->RecordVolumeDimensions();

  // Calculate the planes and corners which define the volume.
  this->DefineCorners();
  this->CalcPlanesAndCorners();

}


/* -----------------------------------------------------------------------
   RecordVolumeDimensions() - Record volume dimensions and resolution
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::RecordVolumeDimensions(void)
{
  typename InputImageType::SpacingType spacing = this->m_Image->GetSpacing();
  SizeType dim=this->m_Image->GetLargestPossibleRegion().GetSize();

  m_NumberOfVoxelsInX = dim[0];
  m_NumberOfVoxelsInY = dim[1];
  m_NumberOfVoxelsInZ = dim[2];

  m_VoxelDimensionInX = spacing[0];
  m_VoxelDimensionInY = spacing[1];
  m_VoxelDimensionInZ = spacing[2];

#ifdef DEEP_DEBUG_RAY
  std::cout << std::endl 
            << " Volume dimensions: " 
            << m_NumberOfVoxelsInX << ", "
            << m_NumberOfVoxelsInY << ", "
            << m_NumberOfVoxelsInZ 
            << " resolution: "
            << m_VoxelDimensionInX << ", "
            << m_VoxelDimensionInY << ", "
            << m_VoxelDimensionInZ 
            << std::endl;
#endif
}


/* -----------------------------------------------------------------------
   DefineCorners() - Define the corners of the volume
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::DefineCorners(void)
{
  // Define corner positions as if at the origin

  m_BoundingCorner[0][0] =
    m_BoundingCorner[1][0] =
    m_BoundingCorner[2][0] =
    m_BoundingCorner[3][0] = 0;

  m_BoundingCorner[4][0] =
    m_BoundingCorner[5][0] =
    m_BoundingCorner[6][0] =
    m_BoundingCorner[7][0] = m_VoxelDimensionInX*m_NumberOfVoxelsInX;

  m_BoundingCorner[1][1] =
    m_BoundingCorner[3][1] =
    m_BoundingCorner[5][1] =
    m_BoundingCorner[7][1] = m_VoxelDimensionInY*m_NumberOfVoxelsInY;

  m_BoundingCorner[0][1] =
    m_BoundingCorner[2][1] =
    m_BoundingCorner[4][1] =
    m_BoundingCorner[6][1] = 0;

  m_BoundingCorner[0][2] =
    m_BoundingCorner[1][2] =
    m_BoundingCorner[4][2] =
    m_BoundingCorner[5][2] =
    m_VoxelDimensionInZ*m_NumberOfVoxelsInZ;

  m_BoundingCorner[2][2] =
    m_BoundingCorner[3][2] =
    m_BoundingCorner[6][2] =
    m_BoundingCorner[7][2] = 0;

}

/* -----------------------------------------------------------------------
   CalcPlanesAndCorners() - Calculate the planes and corners of the volume.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::CalcPlanesAndCorners(void)
{
  int j;


  // find the equations of the planes

  int c1=0, c2=0, c3=0;

  for (j=0; j<6; j++)
    {                                // loop around for planes
    switch (j)
      {                // which corners to take
      case 0:
        c1=1; c2=2; c3=3;
        break;
      case 1:
        c1=4; c2=5; c3=6;
        break;
      case 2:
        c1=5; c2=3; c3=7;
        break;
      case 3:
        c1=2; c2=4; c3=6;
        break;
      case 4:
        c1=1; c2=5; c3=0;
        break;
      case 5:
        c1=3; c2=7; c3=2;
        break;
      }


    double line1x, line1y, line1z;
    double line2x, line2y, line2z;

    // lines from one corner to another in x,y,z dirns
    line1x = m_BoundingCorner[c1][0] - m_BoundingCorner[c2][0];
    line2x = m_BoundingCorner[c1][0] - m_BoundingCorner[c3][0];

    line1y = m_BoundingCorner[c1][1] - m_BoundingCorner[c2][1];
    line2y = m_BoundingCorner[c1][1] - m_BoundingCorner[c3][1];

    line1z = m_BoundingCorner[c1][2] - m_BoundingCorner[c2][2];
    line2z = m_BoundingCorner[c1][2] - m_BoundingCorner[c3][2];

    double A, B, C, D;

    // take cross product
    A = line1y*line2z - line2y*line1z;
    B = line2x*line1z - line1x*line2z;
    C = line1x*line2y - line2x*line1y;

    // find constant
    D = -(   A*m_BoundingCorner[c1][0]
             + B*m_BoundingCorner[c1][1]
             + C*m_BoundingCorner[c1][2] );

    // initialise plane value and normalise
    m_BoundingPlane[j][0] = A/vcl_sqrt(A*A + B*B + C*C);
    m_BoundingPlane[j][1] = B/vcl_sqrt(A*A + B*B + C*C);
    m_BoundingPlane[j][2] = C/vcl_sqrt(A*A + B*B + C*C);
    m_BoundingPlane[j][3] = D/vcl_sqrt(A*A + B*B + C*C);

    if ( (A*A + B*B + C*C) == 0 )
      {
      itk::ExceptionObject err(__FILE__, __LINE__);
      err.SetLocation( ITK_LOCATION );
      err.SetDescription( "Division by zero (planes) "
                          "- CalcPlanesAndCorners().");
      throw err;
      }
    }

}


/* -----------------------------------------------------------------------
   Line3D() - Calculate the equation of the line that projects to (u, v)

   Solve:

     | x |          | u |
   M | y | = lambda | v |
     | z |          | 1 |
     | 1 |

   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::Line3D(double x2D, double y2D, double r[3], double u[3])
{
  double pos1[3];		// Coordinates of point 1
  double pos2[3];		// Coordinates of point 2

  double distance;

  double denom1, denom2, denom3;
  double a, b, c, d, e, f, g, h;


#ifdef DEEP_DEBUG_RAY
  std::cout << " 2D coordinate: " 
            << x2D << ", "
            << y2D << std::endl;
#endif

  a = m_ProjectionMatrix(0, 0) - m_ProjectionMatrix(2, 0)*x2D;
  b = m_ProjectionMatrix(0, 1) - m_ProjectionMatrix(2, 1)*x2D;
  c = m_ProjectionMatrix(0, 2) - m_ProjectionMatrix(2, 2)*x2D;
  d = m_ProjectionMatrix(0, 3) - m_ProjectionMatrix(2, 3)*x2D;

  e = m_ProjectionMatrix(1, 0) - m_ProjectionMatrix(2, 0)*y2D;
  f = m_ProjectionMatrix(1, 1) - m_ProjectionMatrix(2, 1)*y2D;
  g = m_ProjectionMatrix(1, 2) - m_ProjectionMatrix(2, 2)*y2D;
  h = m_ProjectionMatrix(1, 3) - m_ProjectionMatrix(2, 3)*y2D;

  denom1 = g*b - c*f;
  denom2 = c*e - g*a;
  denom3 = a*f - e*b;

  // Choose the largest denominator to prevent division by zero

  if ((fabs(denom1) > fabs(denom2)) && (fabs(denom1) > fabs(denom3))) {

    pos1[0] = 0;
    pos1[1] = (d*g - h*c) / (-denom1);
    pos1[2] = (d*f - h*b) / denom1;


    pos2[0] = 100;
    pos2[1] = (pos2[0]*( a*g - e*c ) + d*g - h*c)/ (-denom1);
    pos2[2] = (pos2[0]*( a*f - e*b ) + d*f - h*b)/ denom1; 	
  }

  else if ((fabs(denom2) > fabs(denom1)) && (fabs(denom2) > fabs(denom3))) {

    pos1[1] = 0;
    pos1[0] = (h*c - d*g) / (-denom2);
    pos1[2] = (h*a - d*e) / denom2;

    pos2[1] = 100;
    pos2[0] = (pos2[1]*( f*c - b*g ) + h*c - d*g)/ (-denom2);
    pos2[2] = (pos2[1]*( f*a - b*e ) + h*a - d*e)/ denom2;
  }

  else if (denom3 != 0.) {

    pos1[2] = 0;
    pos1[0] = (h*b - d*f) / denom3;
    pos1[1] = (h*a - e*d) / (-denom3);

    pos2[2] = 100;
    pos2[0] = (pos2[2]*( g*b - c*f ) + h*b - d*f)/ denom3;
    pos2[1] = (pos2[2]*( g*a - c*e ) + h*a - d*e)/ (-denom3);
  }

  else {
    std::cerr << "WARNING: No values for r and u found" << std::endl
	      << "         denom1 " << denom1 << " denom2 " << denom2  
	      << " denom3 " << denom3 << std::endl;
	  	  
    pos1[2] = 0;
    pos1[0] = 0;
    pos1[1] = 0;

    pos2[2] = 0;
    pos2[0] = 0;
    pos2[1] = 0;
  }

  
#if 0
  // Ensure that the direction of the ray is away from the origin by
  // calculating which of the points (pos1[0], pos1[1], pos1[2]) or 
  // (pos2[0], pos2[1], pos2[2]) is closer to the origin. To do this we have
  // to transform the points temporarily into the world coordinate frame.
	
  int row;			// The rows and columns of the matrix

  double temp[3];		// Temporary storage

  double wPos1[3];		// Cordinates of (xpos1,ypos1,ypos1) in world
  double wPos2[3];		// Cordinates of (xpos2,ypos2,ypos2) in world

  double dist1;			// Distance of point 1 from the origin in world
  double dist2;			// Distance of point 2 from the origin in world

  for (row=0; row<3; row++) {
    wPos1[row] = pos1[row];
    wPos2[row] = pos2[row];
  }

  image2world(wPos1);
  image2world(wPos2);

  dist1 = sqrt(wPos1[0]*wPos1[0] + wPos1[1]*wPos1[1] + wPos1[2]*wPos1[2]);
  dist2 = sqrt(wPos2[0]*wPos2[0] + wPos2[1]*wPos2[1] + wPos2[2]*wPos2[2]);

  if (dist2 < dist1) {
    for (row=0; row<3; row++) temp[row] = pos2[row];
    for (row=0; row<3; row++) pos2[row] = pos1[row];
    for (row=0; row<3; row++) pos1[row] = temp[row];
  }

#endif



  // Finally calculate the position and direction of the line between 
  // these two points.

  distance = sqrt(   (pos1[0] - pos2[0])*(pos1[0] - pos2[0])
		   + (pos1[1] - pos2[1])*(pos1[1] - pos2[1])
		   + (pos1[2] - pos2[2])*(pos1[2] - pos2[2]));

  u[0] = (pos2[0] - pos1[0])/distance;
  u[1] = (pos2[1] - pos1[1])/distance;
  u[2] = (pos2[2] - pos1[2])/distance;

  r[0] = pos1[0];
  r[1] = pos1[1];
  r[2] = pos1[2];

#ifdef DEEP_DEBUG_RAY
  std::cout << " Line equation: u=" 
            << u[0] << ", "
            << u[1] << ", "
            << u[2]
            << " r= "
            << r[0] << ", "
            << r[1] << ", "
            << r[2] 
            << std::endl;
#endif
}


/* -----------------------------------------------------------------------
   CalcRayIntercepts() - Calculate the ray intercepts with the volume.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
bool
Ray<TInputImage, TCoordRep>
::CalcRayIntercepts()
{
  double maxInterDist, interDist;
  double cornerVect[4][3];
  int cross[4][3], noInterFlag[6];
  int nSidesCrossed, crossFlag, c[4];
  double ax, ay, az, bx, by, bz;
  double cubeInter[6][3];
  double denom;

  int i,j, k;
  int NoSides = 6;  // =6 to allow truncation: =4 to remove truncated rays


  Line3D(m_RayPosition2Dmm[0], m_RayPosition2Dmm[1], m_CurrentRayPositionInMM, m_RayDirectionInMM );

  // Calculate intercept of ray with planes

  double interceptx[6], intercepty[6], interceptz[6];
  double d[6];

#ifdef DEEP_DEBUG_RAY
    std::cout << " Intercepts of ray with planes: " << " "; 
#endif

  for (j=0; j<NoSides; j++) {

    denom = (  m_BoundingPlane[j][0]*m_RayDirectionInMM[0]
             + m_BoundingPlane[j][1]*m_RayDirectionInMM[1]
             + m_BoundingPlane[j][2]*m_RayDirectionInMM[2]);

    if ((long)(denom*100) != 0) {

      d[j] = -(   m_BoundingPlane[j][3]
                + m_BoundingPlane[j][0]*m_CurrentRayPositionInMM[0]
                + m_BoundingPlane[j][1]*m_CurrentRayPositionInMM[1]
                + m_BoundingPlane[j][2]*m_CurrentRayPositionInMM[2] ) / denom;

      interceptx[j] = m_CurrentRayPositionInMM[0] + d[j]*m_RayDirectionInMM[0];
      intercepty[j] = m_CurrentRayPositionInMM[1] + d[j]*m_RayDirectionInMM[1];
      interceptz[j] = m_CurrentRayPositionInMM[2] + d[j]*m_RayDirectionInMM[2];

      noInterFlag[j] = 1;  //OK
    }

    else
      noInterFlag[j] = 0;  //NOT OK

#ifdef DEEP_DEBUG_RAY
    std::cout << noInterFlag[j] << " "; 
#endif
  }

#ifdef DEEP_DEBUG_RAY
  std::cout << std::endl ;
#endif

  nSidesCrossed = 0;
  for (j=0; j<NoSides; j++) {

    // Work out which corners to use

    if ( j==0 ) {
      c[0] = 0; c[1] = 1; c[2] = 3; c[3] = 2;
    }
    else if ( j==1 ) {
      c[0] = 4; c[1] = 5; c[2] = 7; c[3] = 6;
    }
    else if ( j==2 ) {
      c[0] = 1; c[1] = 5; c[2] = 7; c[3] = 3;
    }
    else if ( j==3 ) {
      c[0] = 0; c[1] = 2; c[2] = 6; c[3] = 4;
    }
    else if ( j==4 ) { //TOP
      c[0] = 0; c[1] = 1; c[2] = 5; c[3] = 4;
    }
    else if ( j==5 ) { //BOTTOM
      c[0] = 2; c[1] = 3; c[2] = 7; c[3] = 6;
    }

    // Calculate vectors from corner of volume to intercept.
    for ( i=0; i<4; i++ ) {
      if ( noInterFlag[j]==1 ) {

        cornerVect[i][0] = m_BoundingCorner[c[i]][0] - interceptx[j];
        cornerVect[i][1] = m_BoundingCorner[c[i]][1] - intercepty[j];
        cornerVect[i][2] = m_BoundingCorner[c[i]][2] - interceptz[j];
      }
      else {

        cornerVect[i][0] = 0;
        cornerVect[i][1] = 0;
        cornerVect[i][2] = 0;
      }
    }

    // Do cross product with these vectors
    for ( i=0; i<4; i++ ) {

      if ( i==3 )
        k = 0;
      else
        k = i+1;

      ax = cornerVect[i][0];
      ay = cornerVect[i][1];
      az = cornerVect[i][2];
      bx = cornerVect[k][0];
      by = cornerVect[k][1];
      bz = cornerVect[k][2];

      // The int and divide by 100 are to avoid rounding errors.  If
      // these are not included then you get values fluctuating around
      // zero and so in the subsequent check, all the values are not
      // above or below zero.  NB. If you "INT" by too much here though
      // you can get problems in the corners of your volume when rays
      // are allowed to go through more than one plane.
      cross[i][0] = (int)((ay*bz - az*by)/100);
      cross[i][1] = (int)((az*bx - ax*bz)/100);
      cross[i][2] = (int)((ax*by - ay*bx)/100);
    }

    // See if a sign change occured between all these cross products
    // if not, then the ray went through this plane

    crossFlag=0;
    for ( i=0; i<3; i++ ) {

      if ( (   cross[0][i]<=0
	    && cross[1][i]<=0
            && cross[2][i]<=0
            && cross[3][i]<=0)

          || (   cross[0][i]>=0
              && cross[1][i]>=0
              && cross[2][i]>=0
              && cross[3][i]>=0) )

        crossFlag++;

    }


    if ( crossFlag==3 && noInterFlag[j]==1 ) {

      cubeInter[nSidesCrossed][0] = interceptx[j];
      cubeInter[nSidesCrossed][1] = intercepty[j];
      cubeInter[nSidesCrossed][2] = interceptz[j];
      nSidesCrossed++;
    }

  } // End of loop over all four planes

  m_RayStartCoordInMM[0] = cubeInter[0][0];
  m_RayStartCoordInMM[1] = cubeInter[0][1];
  m_RayStartCoordInMM[2] = cubeInter[0][2];

  m_RayEndCoordInMM[0] = cubeInter[1][0];
  m_RayEndCoordInMM[1] = cubeInter[1][1];
  m_RayEndCoordInMM[2] = cubeInter[1][2];

#ifdef DEEP_DEBUG_RAY
  std::cout << " No. of sides crossed equals: " << nSidesCrossed << std::endl;
#else
  if ( nSidesCrossed >= 5 )
    std::cerr << "WARNING: No. of sides crossed equals: " << nSidesCrossed << std::endl;
#endif


  // If 'nSidesCrossed' is larger than 2, this means that the ray goes through
  // a corner of the volume and due to rounding errors, the ray is
  // deemed to go through more than two planes.  To obtain the correct
  // start and end positions we choose the two intercept values which
  // are furthest from each other.


  if ( nSidesCrossed >= 3 ) {

    maxInterDist = 0;
    for ( j=0; j<nSidesCrossed-1; j++ ) {
      for ( k=j+1; k<nSidesCrossed; k++ ) {

        interDist = 0;

        for ( i=0; i<3; i++ )
          interDist += (cubeInter[j][i] - cubeInter[k][i])*
            (cubeInter[j][i] - cubeInter[k][i]);

        if ( interDist > maxInterDist ) {

          maxInterDist = interDist;

          m_RayStartCoordInMM[0] = cubeInter[j][0];
          m_RayStartCoordInMM[1] = cubeInter[j][1];
          m_RayStartCoordInMM[2] = cubeInter[j][2];

          m_RayEndCoordInMM[0] = cubeInter[k][0];
          m_RayEndCoordInMM[1] = cubeInter[k][1];
          m_RayEndCoordInMM[2] = cubeInter[k][2];
	}
      }
    }
    nSidesCrossed = 2;
  }

#ifdef DEEP_DEBUG_RAY
  std::cout << " RayStartCoordInMM: " 
	    << setw(10) << m_RayStartCoordInMM[0] << ", " 
	    << setw(10) << m_RayStartCoordInMM[1] << ", " 
	    << setw(10) << m_RayStartCoordInMM[2] << " mm" 
	    << " RayEndCoordInMM: " 
	    << setw(10) << m_RayEndCoordInMM[0] << ", " 
	    << setw(10) << m_RayEndCoordInMM[1] << ", " 
	    << setw(10) << m_RayEndCoordInMM[2] << " mm" 
	    << std::endl;
#endif

  if (nSidesCrossed == 2 )
    return true;
  else
    return false;
}


/* -----------------------------------------------------------------------
   SetRay() - Set the position and direction of the ray
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
bool
Ray<TInputImage, TCoordRep>
::SetRay(OutputPointType RayPosn)
{

  // Compute the ray path for this coordinate in mm

  m_RayPosition2Dmm[0] = RayPosn[0];
  m_RayPosition2Dmm[1] = RayPosn[1];

  m_ValidRay = this->CalcRayIntercepts();

  if (! m_ValidRay) {
    Reset();
    return false;
  }


  // Convert the start and end coordinates of the ray to voxels

  this->EndPointsInVoxels();

  /* Calculate the ray direction vector in voxels and hence the voxel
     increment required to traverse the ray, and the number of
     interpolation points on the ray.

     This routine also shifts the coordinate frame by half a voxel for
     two of the directional components (i.e. those lying within the
     planes of voxels being traversed). */

  this->CalcDirnVector();


  /* Reduce the length of the ray until both start and end
     coordinates lie inside the volume. */

  m_ValidRay = this->AdjustRayLength();

  // Reset the iterator to the start of the ray.

  Reset();

  return m_ValidRay;
}


/* -----------------------------------------------------------------------
   EndPointsInVoxels() - Convert the endpoints to voxels
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::EndPointsInVoxels(void)
{
  m_RayVoxelStartPosition[0] = m_RayStartCoordInMM[0]/m_VoxelDimensionInX;
  m_RayVoxelStartPosition[1] = m_RayStartCoordInMM[1]/m_VoxelDimensionInY;
  m_RayVoxelStartPosition[2] = m_RayStartCoordInMM[2]/m_VoxelDimensionInZ;

  m_RayVoxelEndPosition[0] = m_RayEndCoordInMM[0]/m_VoxelDimensionInX;
  m_RayVoxelEndPosition[1] = m_RayEndCoordInMM[1]/m_VoxelDimensionInY;
  m_RayVoxelEndPosition[2] = m_RayEndCoordInMM[2]/m_VoxelDimensionInZ;

}


/* -----------------------------------------------------------------------
   CalcDirnVector() - Calculate the incremental direction vector in voxels.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::CalcDirnVector(void)
{
  double xNum, yNum, zNum;

  // Calculate the number of voxels in each direction

  xNum = vcl_fabs(m_RayVoxelStartPosition[0] - m_RayVoxelEndPosition[0]);
  yNum = vcl_fabs(m_RayVoxelStartPosition[1] - m_RayVoxelEndPosition[1]);
  zNum = vcl_fabs(m_RayVoxelStartPosition[2] - m_RayVoxelEndPosition[2]);

#ifdef DEEP_DEBUG_RAY
  std::cout << " RayVoxelStartPosition: " 
	    << setw(6) << m_RayVoxelStartPosition[0] << ", " 
	    << setw(6) << m_RayVoxelStartPosition[1] << ", " 
	    << setw(6) << m_RayVoxelStartPosition[2] << " voxels" 
	    << " RayVoxelEndPosition: " 
	    << setw(6) << m_RayVoxelEndPosition[0] << ", " 
	    << setw(6) << m_RayVoxelEndPosition[1] << ", " 
	    << setw(6) << m_RayVoxelEndPosition[2] << " voxels" 
	    << " Number of Voxels: " 
	    << setw(6) << xNum << ", " 
	    << setw(6) << yNum << ", " 
	    << setw(6) << zNum 
	    << std::endl;
#endif

  // The direction iterated in is that with the greatest number of voxels
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Iterate in X direction

  if ( (xNum >= yNum) && (xNum >= zNum) )
    {

    if ( m_RayVoxelStartPosition[0] < m_RayVoxelEndPosition[0] )
      {
      m_VoxelIncrement[0] = 1;

      m_VoxelIncrement[1]
        = (m_RayVoxelStartPosition[1]
           - m_RayVoxelEndPosition[1])/(m_RayVoxelStartPosition[0]
                                        - m_RayVoxelEndPosition[0]);

      m_VoxelIncrement[2]
        = (m_RayVoxelStartPosition[2]
           - m_RayVoxelEndPosition[2])/(m_RayVoxelStartPosition[0]
                                        - m_RayVoxelEndPosition[0]);
      }
    else
      {
      m_VoxelIncrement[0] = -1;

      m_VoxelIncrement[1]
        = -(m_RayVoxelStartPosition[1]
            - m_RayVoxelEndPosition[1])/(m_RayVoxelStartPosition[0]
                                         - m_RayVoxelEndPosition[0]);

      m_VoxelIncrement[2]
        = -(m_RayVoxelStartPosition[2]
            - m_RayVoxelEndPosition[2])/(m_RayVoxelStartPosition[0]
                                         - m_RayVoxelEndPosition[0]);
      }


    // This section is to alter the start position in order to
    // place the center of the voxels in there correct positions,
    // rather than placing them at the corner of voxels which is
    // what happens if this is not carried out.  The reason why
    // x has no -0.5 is because this is the direction we are going
    // to iterate in and therefore we wish to go from center to
    // center rather than finding the surrounding voxels.

    m_RayVoxelStartPosition[1] += ( (int)m_RayVoxelStartPosition[0]
        - m_RayVoxelStartPosition[0])*m_VoxelIncrement[1]*m_VoxelIncrement[0]
      + 0.5*m_VoxelIncrement[1] - 0.5;

    m_RayVoxelStartPosition[2] += ( (int)m_RayVoxelStartPosition[0]
           - m_RayVoxelStartPosition[0])*m_VoxelIncrement[2]*m_VoxelIncrement[0]
      + 0.5*m_VoxelIncrement[2] - 0.5;

    m_RayVoxelStartPosition[0] = (int)m_RayVoxelStartPosition[0] + 0.5*m_VoxelIncrement[0];

    m_TotalRayVoxelPlanes = (int)xNum;

    m_TraversalDirection = TRANSVERSE_IN_X;
    }

  // Iterate in Y direction

  else if ( (yNum >= xNum) && (yNum >= zNum) )
    {

    if ( m_RayVoxelStartPosition[1] < m_RayVoxelEndPosition[1] )
      {
      m_VoxelIncrement[1] = 1;

      m_VoxelIncrement[0]
        = (m_RayVoxelStartPosition[0]
           - m_RayVoxelEndPosition[0])/(m_RayVoxelStartPosition[1]
                                        - m_RayVoxelEndPosition[1]);

      m_VoxelIncrement[2]
        = (m_RayVoxelStartPosition[2]
           - m_RayVoxelEndPosition[2])/(m_RayVoxelStartPosition[1]
                                        - m_RayVoxelEndPosition[1]);
      }
    else
      {
      m_VoxelIncrement[1] = -1;

      m_VoxelIncrement[0]
        = -(m_RayVoxelStartPosition[0]
            - m_RayVoxelEndPosition[0])/(m_RayVoxelStartPosition[1]
                                         - m_RayVoxelEndPosition[1]);

      m_VoxelIncrement[2]
        = -(m_RayVoxelStartPosition[2]
            - m_RayVoxelEndPosition[2])/(m_RayVoxelStartPosition[1]
                                         - m_RayVoxelEndPosition[1]);
      }

    m_RayVoxelStartPosition[0] += ( (int)m_RayVoxelStartPosition[1]
                                    - m_RayVoxelStartPosition[1])*m_VoxelIncrement[0]*m_VoxelIncrement[1]
      + 0.5*m_VoxelIncrement[0] - 0.5;

    m_RayVoxelStartPosition[2] += ( (int)m_RayVoxelStartPosition[1]
                                    - m_RayVoxelStartPosition[1])*m_VoxelIncrement[2]*m_VoxelIncrement[1]
      + 0.5*m_VoxelIncrement[2] - 0.5;

    m_RayVoxelStartPosition[1] = (int)m_RayVoxelStartPosition[1] + 0.5*m_VoxelIncrement[1];

    m_TotalRayVoxelPlanes = (int)yNum;

    m_TraversalDirection = TRANSVERSE_IN_Y;
    }

  // Iterate in Z direction

  else {

    if ( m_RayVoxelStartPosition[2] < m_RayVoxelEndPosition[2] ) {

      m_VoxelIncrement[2] = 1;
      
      m_VoxelIncrement[0]
        = (m_RayVoxelStartPosition[0]
           - m_RayVoxelEndPosition[0])/(m_RayVoxelStartPosition[2]
                                        - m_RayVoxelEndPosition[2]);

      m_VoxelIncrement[1]
        = (m_RayVoxelStartPosition[1]
           - m_RayVoxelEndPosition[1])/(m_RayVoxelStartPosition[2]
                                        - m_RayVoxelEndPosition[2]);
    }

    else {

      m_VoxelIncrement[2] = -1;

      m_VoxelIncrement[0]
        = -(m_RayVoxelStartPosition[0]
            - m_RayVoxelEndPosition[0])/(m_RayVoxelStartPosition[2]
                                         - m_RayVoxelEndPosition[2]);

      m_VoxelIncrement[1]
        = -(m_RayVoxelStartPosition[1]
            - m_RayVoxelEndPosition[1])/(m_RayVoxelStartPosition[2]
                                         - m_RayVoxelEndPosition[2]);
    }

    m_RayVoxelStartPosition[0] += ( (int)m_RayVoxelStartPosition[2]
                                    - m_RayVoxelStartPosition[2])*m_VoxelIncrement[0]*m_VoxelIncrement[2]
      + 0.5*m_VoxelIncrement[0] - 0.5;

    m_RayVoxelStartPosition[1] += ( (int)m_RayVoxelStartPosition[2]
                                    - m_RayVoxelStartPosition[2])*m_VoxelIncrement[1]*m_VoxelIncrement[2]
      + 0.5*m_VoxelIncrement[1] - 0.5;

    m_RayVoxelStartPosition[2] = (int)m_RayVoxelStartPosition[2] + 0.5*m_VoxelIncrement[2];

    m_TotalRayVoxelPlanes = (int)zNum;

#ifdef DEEP_DEBUG_RAY
    std::cout << "TotalRayVoxelPlanes: " << m_TotalRayVoxelPlanes << endl;
#endif

    m_TraversalDirection = TRANSVERSE_IN_Z;
  }

}


/* -----------------------------------------------------------------------
   AdjustRayLength() - Ensure that the ray lies within the volume
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
bool
Ray<TInputImage, TCoordRep>
::AdjustRayLength(void)
{
  bool startOK, endOK;

  int Istart[3];
  int Idirn[3];


  if (m_TraversalDirection == TRANSVERSE_IN_X) {
    Idirn[0] = 0;
    Idirn[1] = 1;
    Idirn[2] = 1;
  }
  else if (m_TraversalDirection == TRANSVERSE_IN_Y) {
    Idirn[0] = 1;
    Idirn[1] = 0;
    Idirn[2] = 1;
  }
  else if (m_TraversalDirection == TRANSVERSE_IN_Z) {
    Idirn[0] = 1;
    Idirn[1] = 1;
    Idirn[2] = 0;
  }
  else {
    itk::ExceptionObject err(__FILE__, __LINE__);
    err.SetLocation( ITK_LOCATION );
    err.SetDescription( "The ray traversal direction is unset "
                        "- AdjustRayLength().");
    throw err;
    return false;
  }


  do {

    startOK = false;
    endOK = false;
    
    Istart[0] = (int) vcl_floor(m_RayVoxelStartPosition[0]);
    Istart[1] = (int) vcl_floor(m_RayVoxelStartPosition[1]);
    Istart[2] = (int) vcl_floor(m_RayVoxelStartPosition[2]);

    if ((Istart[0] >= 0) && (Istart[0] + Idirn[0] < m_NumberOfVoxelsInX) &&
        (Istart[1] >= 0) && (Istart[1] + Idirn[1] < m_NumberOfVoxelsInY) &&
        (Istart[2] >= 0) && (Istart[2] + Idirn[2] < m_NumberOfVoxelsInZ) ) {

      startOK = true;
    }
    else {
      m_RayVoxelStartPosition[0] += m_VoxelIncrement[0];
      m_RayVoxelStartPosition[1] += m_VoxelIncrement[1];
      m_RayVoxelStartPosition[2] += m_VoxelIncrement[2];

      m_TotalRayVoxelPlanes--;

    }

    Istart[0] = (int) vcl_floor(m_RayVoxelStartPosition[0]
                            + m_TotalRayVoxelPlanes*m_VoxelIncrement[0]);

    Istart[1] = (int) vcl_floor(m_RayVoxelStartPosition[1]
                            + m_TotalRayVoxelPlanes*m_VoxelIncrement[1]);

    Istart[2] = (int) vcl_floor(m_RayVoxelStartPosition[2]
                            + m_TotalRayVoxelPlanes*m_VoxelIncrement[2]);
  
    if ((Istart[0] >= 0) && (Istart[0] + Idirn[0] < m_NumberOfVoxelsInX) &&
	(Istart[1] >= 0) && (Istart[1] + Idirn[1] < m_NumberOfVoxelsInY) &&
        (Istart[2] >= 0) && (Istart[2] + Idirn[2] < m_NumberOfVoxelsInZ) ) {

      endOK = true;
    }
    else {

#ifdef DEEP_DEBUG_RAY
      cout <<  " m_TotalRayVoxelPlanes: " << m_TotalRayVoxelPlanes << endl
	   << "Istart[0]: " << Istart[0] << " m_RayVoxelStartPosition[0]: " << m_RayVoxelStartPosition[0] << " m_VoxelIncrement[0]: " << m_VoxelIncrement[0] << " Idirn[0]: " << Idirn[0] << " m_NumberOfVoxelsInX: " << m_NumberOfVoxelsInX << endl
	   << "Istart[1]: " << Istart[1] << " m_RayVoxelStartPosition[1]: " << m_RayVoxelStartPosition[1] << " m_VoxelIncrement[1]: " << m_VoxelIncrement[1] << " Idirn[1]: " << Idirn[1] << " m_NumberOfVoxelsInY: " << m_NumberOfVoxelsInY << endl
	   << "Istart[2]: " << Istart[2] << " m_RayVoxelStartPosition[2]: " << m_RayVoxelStartPosition[2] << " m_VoxelIncrement[2]: " << m_VoxelIncrement[2] << " Idirn[2]: " << Idirn[2] << " m_NumberOfVoxelsInZ: " << m_NumberOfVoxelsInZ << endl;
#endif
      m_TotalRayVoxelPlanes--;

    }

  } while ( (! (startOK && endOK)) && (m_TotalRayVoxelPlanes > 1) );


  return (startOK && endOK);
}


/* -----------------------------------------------------------------------
   Reset() - Reset the iterator to the start of the ray.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::Reset(void)
{
  int i;

  m_NumVoxelPlanesTraversed = -1;

  // If this is a valid ray...

  if (m_ValidRay) {

    for (i=0; i<3; i++) 
      m_Position3Dvox[i] = m_RayVoxelStartPosition[i];

    this->InitialiseVoxelPointers();
  }

  // otherwise set parameters to zero

  else {

    for (i=0; i<3; i++) {

      m_RayVoxelStartPosition[i] = 0.;
      m_RayVoxelEndPosition[i] = 0.;
      m_VoxelIncrement[i] = 0.;
      m_RayIntersectionVoxelIndex[i] = 0;
    }

    m_TraversalDirection = UNDEFINED_DIRECTION;
    m_TotalRayVoxelPlanes = 0;

    for (i=0; i<4; i++)
      m_RayIntersectionVoxels[i] = 0;
  }
}


/* -----------------------------------------------------------------------
   InitialiseVoxelPointers() - Obtain pointers to the first four voxels
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::InitialiseVoxelPointers(void)
{
  IndexType index;

  int Ix, Iy, Iz;

  Ix = (int) vcl_floor(m_RayVoxelStartPosition[0]);
  Iy = (int) vcl_floor(m_RayVoxelStartPosition[1]);
  Iz = (int) vcl_floor(m_RayVoxelStartPosition[2]);

#ifdef DEEP_DEBUG_RAY
  std::cout << " RayVoxelStartPosition: " 
	    << setw(6) << m_RayVoxelStartPosition[0] << ", " 
	    << setw(6) << m_RayVoxelStartPosition[1] << ", " 
	    << setw(6) << m_RayVoxelStartPosition[2] << " voxels" 
	    << " Ix,Iy,Iz: " 
	    << setw(6) << Ix << ", " 
	    << setw(6) << Iy << ", " 
	    << setw(6) << Iz << " voxels" << std::endl;
#endif

  m_RayIntersectionVoxelIndex[0] = Ix;
  m_RayIntersectionVoxelIndex[1] = Iy;
  m_RayIntersectionVoxelIndex[2] = Iz;

  switch( m_TraversalDirection )
    {
    case TRANSVERSE_IN_X:
      {

      if ((Ix >= 0) && (Ix     < m_NumberOfVoxelsInX) &&
          (Iy >= 0) && (Iy + 1 < m_NumberOfVoxelsInY) &&
          (Iz >= 0) && (Iz + 1 < m_NumberOfVoxelsInZ))
        {
        
        index[0]=Ix; index[1]=Iy; index[2]=Iz;
        m_RayIntersectionVoxels[0]
          = const_cast<PixelType *>((   this->m_Image->GetBufferPointer() 
				      + this->m_Image->ComputeOffset(index)) );
        
        index[0]=Ix; index[1]=Iy+1; index[2]=Iz;
        m_RayIntersectionVoxels[1]
          = const_cast<PixelType *>((   this->m_Image->GetBufferPointer() 
				      + this->m_Image->ComputeOffset(index)) );
        
        index[0]=Ix; index[1]=Iy; index[2]=Iz+1;
        m_RayIntersectionVoxels[2]
          = const_cast<PixelType *>((   this->m_Image->GetBufferPointer() 
				      + this->m_Image->ComputeOffset(index)) );
        
        index[0]=Ix; index[1]=Iy+1; index[2]=Iz+1;
        m_RayIntersectionVoxels[3]
          = const_cast<PixelType *>((   this->m_Image->GetBufferPointer() 
				      + this->m_Image->ComputeOffset(index)) );
        }
      else
        {
        m_RayIntersectionVoxels[0] =
          m_RayIntersectionVoxels[1] =
          m_RayIntersectionVoxels[2] =
          m_RayIntersectionVoxels[3] = NULL;
        }
      break;
      }

    case TRANSVERSE_IN_Y:
      {

      if ((Ix >= 0) && (Ix + 1 < m_NumberOfVoxelsInX) &&
          (Iy >= 0) && (Iy     < m_NumberOfVoxelsInY) &&
          (Iz >= 0) && (Iz + 1 < m_NumberOfVoxelsInZ))
        {
        
        index[0]=Ix; index[1]=Iy; index[2]=Iz;
        m_RayIntersectionVoxels[0] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        
        index[0]=Ix+1; index[1]=Iy; index[2]=Iz;
        m_RayIntersectionVoxels[1] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        
        index[0]=Ix; index[1]=Iy; index[2]=Iz+1;
        m_RayIntersectionVoxels[2] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        
        index[0]=Ix+1; index[1]=Iy; index[2]=Iz+1;
        m_RayIntersectionVoxels[3] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        }
      else
        {
        m_RayIntersectionVoxels[0]
        = m_RayIntersectionVoxels[1]
        = m_RayIntersectionVoxels[2]
        = m_RayIntersectionVoxels[3] = NULL;
        }
      break;
      }

    case TRANSVERSE_IN_Z:
      {

      if ((Ix >= 0) && (Ix + 1 < m_NumberOfVoxelsInX)   &&
          (Iy >= 0) && (Iy + 1 < m_NumberOfVoxelsInY) &&
          (Iz >= 0) && (Iz     < m_NumberOfVoxelsInZ))
        {
        
        index[0]=Ix; index[1]=Iy; index[2]=Iz;
        m_RayIntersectionVoxels[0] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        
        index[0]=Ix+1; index[1]=Iy; index[2]=Iz;
        m_RayIntersectionVoxels[1] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        
        index[0]=Ix; index[1]=Iy+1; index[2]=Iz;
        m_RayIntersectionVoxels[2] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        
        index[0]=Ix+1; index[1]=Iy+1; index[2]=Iz;
        m_RayIntersectionVoxels[3] 
	  = const_cast<PixelType *>(   this->m_Image->GetBufferPointer()
				     + this->m_Image->ComputeOffset(index) );
        
        }
      else
        {
        m_RayIntersectionVoxels[0]
        = m_RayIntersectionVoxels[1]
        = m_RayIntersectionVoxels[2]
        = m_RayIntersectionVoxels[3] = NULL;
        }
      break;
      }

    default:
      {
      itk::ExceptionObject err(__FILE__, __LINE__);
      err.SetLocation( ITK_LOCATION );
      err.SetDescription( "The ray traversal direction is unset "
                          "- InitialiseVoxelPointers().");
      throw err;
      return;
      }
    }
}

/* -----------------------------------------------------------------------
   IncrementVoxelPointers() - Increment the voxel pointers
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::IncrementVoxelPointers(void)
{
  double xBefore = m_Position3Dvox[0];
  double yBefore = m_Position3Dvox[1];
  double zBefore = m_Position3Dvox[2];

  m_Position3Dvox[0] += m_VoxelIncrement[0];
  m_Position3Dvox[1] += m_VoxelIncrement[1];
  m_Position3Dvox[2] += m_VoxelIncrement[2];

  int dx = ((int) m_Position3Dvox[0]) - ((int) xBefore);
  int dy = ((int) m_Position3Dvox[1]) - ((int) yBefore);
  int dz = ((int) m_Position3Dvox[2]) - ((int) zBefore);

  m_RayIntersectionVoxelIndex[0] += dx;
  m_RayIntersectionVoxelIndex[1] += dy;
  m_RayIntersectionVoxelIndex[2] += dz;

  int totalRayVoxelPlanes
    = dx + dy*m_NumberOfVoxelsInX + dz*m_NumberOfVoxelsInX*m_NumberOfVoxelsInY;

  m_RayIntersectionVoxels[0] += totalRayVoxelPlanes;
  m_RayIntersectionVoxels[1] += totalRayVoxelPlanes;
  m_RayIntersectionVoxels[2] += totalRayVoxelPlanes;
  m_RayIntersectionVoxels[3] += totalRayVoxelPlanes;
}


/* -----------------------------------------------------------------------
   GetBilinearCoefficients() - Get the the bilinear coefficient for the current ray point.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::GetBilinearCoefficients(double &y, double &z) const
{
  switch( m_TraversalDirection )
    {
    case TRANSVERSE_IN_X:
      {
      y = m_Position3Dvox[1] - vcl_floor(m_Position3Dvox[1]);
      z = m_Position3Dvox[2] - vcl_floor(m_Position3Dvox[2]);
      break;
      }
    case TRANSVERSE_IN_Y:
      {
      y = m_Position3Dvox[0] - vcl_floor(m_Position3Dvox[0]);
      z = m_Position3Dvox[2] - vcl_floor(m_Position3Dvox[2]);
      break;
      }
    case TRANSVERSE_IN_Z:
      {
      y = m_Position3Dvox[0] - vcl_floor(m_Position3Dvox[0]);
      z = m_Position3Dvox[1] - vcl_floor(m_Position3Dvox[1]);
      break;
      }
    default:
      {
      itk::ExceptionObject err(__FILE__, __LINE__);
      err.SetLocation( ITK_LOCATION );
      err.SetDescription( "The ray traversal direction is unset "
                          "- GetCurrentIntensity().");
      throw err;
      return;
      }
    }
}


/* -----------------------------------------------------------------------
   GetCurrentIntensity() - Get the intensity of the current ray point.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
double
Ray<TInputImage, TCoordRep>
::GetCurrentIntensity(void) const
{
  double a, b, c, d;
  double y, z;

  if (! m_ValidRay) return 0;

  a = (double) (*m_RayIntersectionVoxels[0]);
  b = (double) (*m_RayIntersectionVoxels[1] - a);
  c = (double) (*m_RayIntersectionVoxels[2] - a);
  d = (double) (*m_RayIntersectionVoxels[3] - a - b - c);

  GetBilinearCoefficients(y, z);

#ifdef DEBUG_RAY
  std::cout << " RayIntersectionVoxels: " 
	    << setw(6) << *m_RayIntersectionVoxels[0] << ", " 
	    << setw(6) << *m_RayIntersectionVoxels[1] << ", " 
	    << setw(6) << *m_RayIntersectionVoxels[2] << ", " 
	    << setw(6) << *m_RayIntersectionVoxels[3];
#endif

  return a + b*y + c*z + d*y*z;
}


/* -----------------------------------------------------------------------
   NextPoint() - Step along the ray.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
bool
Ray<TInputImage, TCoordRep>
::NextPoint(void)
{
  if (! m_ValidRay) 
    return false;

  // The first time this routine is called 'm_NumVoxelPlanesTraversed' should equal -1.

  m_NumVoxelPlanesTraversed++;

#ifdef DEEP_DEBUG_RAY
  std::cout << "NumVoxelPlanesTraversed: " << m_NumVoxelPlanesTraversed << std::endl;
#endif

  // Have we finished stepping along the ray?

  if (m_NumVoxelPlanesTraversed > m_TotalRayVoxelPlanes)
    return false;

  // Are we trying to step beyond the end of the ray?

  if (m_NumVoxelPlanesTraversed > m_TotalRayVoxelPlanes) {
    std::cerr << "ERROR: The end of the ray has already been reached, voxel " 
	 << m_NumVoxelPlanesTraversed << " of " << m_TotalRayVoxelPlanes
	 << ", position (" << m_RayPosition2Dmm[0] << ", " << m_RayPosition2Dmm[1] << ") mm."
	 << std::endl << "       In routine: Ray::NextPoint()" << std::endl;
    exit(1);
    return false;
  }


  /* If 'iVoxel' is greater than zero then this isn't the first voxel and
     we need to increment the voxel pointers. This means that each time this
     this routine exits the four voxel pointers will be pointing at the 
     correct voxels surrounding the current position on the ray. */

  if (m_NumVoxelPlanesTraversed > 0) 
    IncrementVoxelPointers();

  return true;
}


/* -----------------------------------------------------------------------
   IncrementIntensities() - Increment the intensities of the current ray point
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::IncrementIntensities(double increment)
{
  double y, z, yz;

  if (! m_ValidRay) return;
  
#ifdef DEEP_DEBUG_RAY
  std::cout << " Position3Dvox: " 
	    << setw(6) << m_Position3Dvox[0] << ", " 
	    << setw(6) << m_Position3Dvox[1] << ", " 
	    << setw(6) << m_Position3Dvox[2] << " voxels" << std::endl;
#endif

  GetBilinearCoefficients(y, z);

  yz = y*z;

  *m_RayIntersectionVoxels[0] += increment*(1. - y - z + yz);
  *m_RayIntersectionVoxels[1] += increment*(y - yz);
  *m_RayIntersectionVoxels[2] += increment*(z - yz);
  *m_RayIntersectionVoxels[3] += increment*yz;

#ifdef DEBUG_RAY
  std::cout << "Image: " << m_Image << " Increment: " 
	    << setw(10) << increment << ", " 
	    << " Coefficients: " 
	    << setw(10) << (1. - y - z + yz) << ", " 
	    << setw(10) << (y - yz)          << ", " 
	    << setw(10) << (z - yz)          << ", " 
	    << setw(10) << yz
	    << " Voxels: " 
	    << setw(10) << *m_RayIntersectionVoxels[0] << ", " 
	    << setw(10) << *m_RayIntersectionVoxels[1] << ", " 
	    << setw(10) << *m_RayIntersectionVoxels[2] << ", " 
	    << setw(10) << *m_RayIntersectionVoxels[3]
	    << std::endl;
#endif

}



/* -----------------------------------------------------------------------
   IncrementIntensities() - Increment the intensities of the current
   ray point by one.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
void
Ray<TInputImage, TCoordRep>
::IncrementIntensities(void)
{
  (*m_RayIntersectionVoxels[0])++;
  (*m_RayIntersectionVoxels[1])++;
  (*m_RayIntersectionVoxels[2])++;
  (*m_RayIntersectionVoxels[3])++;
}


/* -----------------------------------------------------------------------
   IntegrateAboveThreshold() - Integrate intensities above a threshold.
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
bool
Ray<TInputImage, TCoordRep>
::IntegrateAboveThreshold(double &integral, double threshold)
{
  double intensity;

  integral = 0.;

  // Check if this is a valid ray

  if (! m_ValidRay)
    {
    return false;
    }


  /* Step along the ray as quickly as possible
     integrating the interpolated intensities. */

  for (m_NumVoxelPlanesTraversed=0;
       m_NumVoxelPlanesTraversed<=m_TotalRayVoxelPlanes;
       m_NumVoxelPlanesTraversed++) {

#ifdef DEBUG_RAY
    std::cout << "Image: " << m_Image << " Ray point: " << setw(6) << m_NumVoxelPlanesTraversed << " ";
#endif

    intensity = this->GetCurrentIntensity();

    if (threshold) {
      if (intensity > threshold)
	integral += intensity - threshold;
    }
    else
      integral += intensity;

#ifdef DEBUG_RAY
    std::cout << " Interpolated intensity: " << setw(10) << intensity << " "
	      << " Integral: " << setw(10) << integral << std::endl;
#endif

    this->IncrementVoxelPointers();
  }

#ifdef DEBUG_RAY
  std::cout << std::endl;
#endif

  /* The ray passes through the volume one plane of voxels at a time,
     however, if its moving diagonally the ray points will be further
     apart so account for this by scaling by the ratio of sampled
     volume to the voxel volume. */

  integral *= this->GetRayPointSpacing()*this->m_ProjectionResolution2Dmm[0]*m_ProjectionResolution2Dmm[1]
    / (this->m_VoxelDimensionInX*this->m_VoxelDimensionInY*this->m_VoxelDimensionInZ);

  return true;
}


/* -----------------------------------------------------------------------
   IncrementRayVoxelIntensities() - Increment all the voxels along a ray using
   bilinear interpolation
   ----------------------------------------------------------------------- */

template<class TInputImage, class TCoordRep>
bool
Ray<TInputImage, TCoordRep>
::IncrementRayVoxelIntensities(double increment)
{
  // Check if this is a valid ray

  if (! m_ValidRay)
    return false;

#ifdef DEBUG_RAY
  std::cout << "Back projecting intensity: " << increment << " / " << this->GetNumberOfRayPoints() + 1 << std::endl;
#endif
  increment /= this->GetNumberOfRayPoints() + 1;

  /* Step along the ray as quickly as possible
     setting the interpolated intensities. */

  for (m_NumVoxelPlanesTraversed=0;
       m_NumVoxelPlanesTraversed<=m_TotalRayVoxelPlanes;
       m_NumVoxelPlanesTraversed++) {

    this->IncrementIntensities(increment);
    this->IncrementVoxelPointers();
  }

  return true;
}

} // namespace itk

#endif
