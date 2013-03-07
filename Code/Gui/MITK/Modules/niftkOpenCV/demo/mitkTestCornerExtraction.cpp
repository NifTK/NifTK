/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTestCornerExtraction.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include <vtkPolyDataWriter.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>

using namespace std;
using namespace cv;

namespace mitk {

int width = 1920;
int height = 1080;

int cornerGaussianBlur = 3;
int meanWindow = 47;
int meanC = 5;
int dilations = 4;

int cornerMaxCorners = 100;
int qualityLevel = 10;
int minDistance = 1;
int averagingDistance = 10;
int epiPolarDistance = 2;
int lineChoice = 0;

cv::Mat leftIntrinsic, leftDistortion, leftRotation, leftTranslation;
cv::Mat rightIntrinsic, rightDistortion, rightRotation, rightTranslation;
cv::Mat fundamental;
cv::Mat r2lRotation;
cv::Mat r2lTranslation;

cv::Mat leftMapX, leftMapY;
cv::Mat rightMapX, rightMapY;

cv::Mat R1, R2, P1, P2, Q;

cv::Mat cornerSrcLeft,  cornerSrcLeftUndistorted,  cornerSrcLeftRectified, cornerSrcLeftRectifiedGrey, cornerSrcGreyLeft, cornerSrcGreyBlurLeft, cornerSrcGreyBlurThreshLeft, cornersSrcGreyBlurThreshDilateLeft, cornerDstLeft, cornerDetectedEdgesLeft;
cv::Mat cornerSrcRight, cornerSrcRightUndistorted, cornerSrcRightRectified, cornerSrcRightRectifiedGrey, cornerSrcGreyRight, cornerSrcGreyBlurRight, cornerSrcGreyBlurThreshRight, cornersSrcGreyBlurThreshDilateRight, cornerDstRight, cornerDetectedEdgesRight;

char* leftWindowName = "Left Window";
char* rightWindowName = "Right Window";

#define MAX_CONTOUR_APPROX  7

struct CvContourEx
{
    CV_CONTOUR_FIELDS()
    int counter;
};

struct CvCBCorner
{
    CvPoint2D32f pt; // Coordinates of the corner
    int row;         // Board row index
    int count;       // Number of neighbor corners
    struct CvCBCorner* neighbors[4]; // Neighbor corners

    float meanDist(int *_n) const
    {
        float sum = 0;
        int n = 0;
        for( int i = 0; i < 4; i++ )
        {
            if( neighbors[i] )
            {
                float dx = neighbors[i]->pt.x - pt.x;
                float dy = neighbors[i]->pt.y - pt.y;
                sum += sqrt(dx*dx + dy*dy);
                n++;
            }
        }
        if(_n)
            *_n = n;
        return sum/MAX(n,1);
    }
};

struct CvCBQuad
{
    int count;      // Number of quad neighbors
    int group_idx;  // quad group ID
    int row, col;   // row and column of this quad
    bool ordered;   // true if corners/neighbors are ordered counter-clockwise
    float edge_len; // quad edge len, in pix^2
    // neighbors and corners are synced, i.e., neighbor 0 shares corner 0
    CvCBCorner *corners[4]; // Coordinates of quad corners
    struct CvCBQuad *neighbors[4]; // Pointers of quad neighbors
};

int icvGenerateQuads( CvCBQuad **out_quads, CvCBCorner **out_corners,
    CvMemStorage *storage, CvMat *image, int flags )
{
  int quad_count = 0;
  cv::Ptr<CvMemStorage> temp_storage;

  if( out_quads )
      *out_quads = 0;

  if( out_corners )
      *out_corners = 0;

  CvSeq *src_contour = 0;
  CvSeq *root;
  CvContourEx* board = 0;
  CvContourScanner scanner;
  int i, idx, min_size;

  CV_Assert( out_corners && out_quads );

  // empiric bound for minimal allowed perimeter for squares
  min_size = 25; //cvRound( image->cols * image->rows * .03 * 0.01 * 0.92 );

  // create temporary storage for contours and the sequence of pointers to found quadrangles
  temp_storage = cvCreateChildMemStorage( storage );
  root = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSeq*), temp_storage );

  // initialize contour retrieving routine
  scanner = cvStartFindContours( image, temp_storage, sizeof(CvContourEx),
                                 CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

  // get all the contours one by one
  while( (src_contour = cvFindNextContour( scanner )) != 0 )
  {
      CvSeq *dst_contour = 0;
      CvRect rect = ((CvContour*)src_contour)->rect;

      // reject contours with too small perimeter
      if( CV_IS_SEQ_HOLE(src_contour) && rect.width*rect.height >= min_size )
      {
          const int min_approx_level = 1, max_approx_level = MAX_CONTOUR_APPROX;
          int approx_level;
          for( approx_level = min_approx_level; approx_level <= max_approx_level; approx_level++ )
          {
              dst_contour = cvApproxPoly( src_contour, sizeof(CvContour), temp_storage,
                                          CV_POLY_APPROX_DP, (float)approx_level );
              if( dst_contour->total == 4 )
                  break;

              // we call this again on its own output, because sometimes
              // cvApproxPoly() does not simplify as much as it should.
              dst_contour = cvApproxPoly( dst_contour, sizeof(CvContour), temp_storage,
                                          CV_POLY_APPROX_DP, (float)approx_level );

              if( dst_contour->total == 4 )
                  break;
          }

          // reject non-quadrangles
          if( dst_contour->total == 4 && cvCheckContourConvexity(dst_contour) )
          {
              CvPoint pt[4];
              double d1, d2, p = cvContourPerimeter(dst_contour);
              double area = fabs(cvContourArea(dst_contour, CV_WHOLE_SEQ));
              double dx, dy;

              for( i = 0; i < 4; i++ )
                  pt[i] = *(CvPoint*)cvGetSeqElem(dst_contour, i);

              dx = pt[0].x - pt[2].x;
              dy = pt[0].y - pt[2].y;
              d1 = sqrt(dx*dx + dy*dy);

              dx = pt[1].x - pt[3].x;
              dy = pt[1].y - pt[3].y;
              d2 = sqrt(dx*dx + dy*dy);

              // philipg.  Only accept those quadrangles which are more square
              // than rectangular and which are big enough
              double d3, d4;
              dx = pt[0].x - pt[1].x;
              dy = pt[0].y - pt[1].y;
              d3 = sqrt(dx*dx + dy*dy);
              dx = pt[1].x - pt[2].x;
              dy = pt[1].y - pt[2].y;
              d4 = sqrt(dx*dx + dy*dy);
              if( !(flags & CV_CALIB_CB_FILTER_QUADS) ||
                  (d3*4 > d4 && d4*4 > d3 && d3*d4 < area*1.5 && area > min_size &&
                  d1 >= 0.15 * p && d2 >= 0.15 * p) )
              {
                  CvContourEx* parent = (CvContourEx*)(src_contour->v_prev);
                  parent->counter++;
                  if( !board || board->counter < parent->counter )
                      board = parent;
                  dst_contour->v_prev = (CvSeq*)parent;
                  //for( i = 0; i < 4; i++ ) cvLine( debug_img, pt[i], pt[(i+1)&3], cvScalar(200,255,255), 1, CV_AA, 0 );
                  cvSeqPush( root, &dst_contour );
              }
          }
      }
  }

  // finish contour retrieving
  cvEndFindContours( &scanner );

  // allocate quad & corner buffers
  *out_quads = (CvCBQuad*)cvAlloc((root->total+root->total / 2) * sizeof((*out_quads)[0]));
  *out_corners = (CvCBCorner*)cvAlloc((root->total+root->total / 2) * 4 * sizeof((*out_corners)[0]));

  // Create array of quads structures
  for( idx = 0; idx < root->total; idx++ )
  {
      CvCBQuad* q = &(*out_quads)[quad_count];
      src_contour = *(CvSeq**)cvGetSeqElem( root, idx );
      if( (flags & CV_CALIB_CB_FILTER_QUADS) && src_contour->v_prev != (CvSeq*)board )
          continue;

      // reset group ID
      memset( q, 0, sizeof(*q) );
      q->group_idx = -1;
      assert( src_contour->total == 4 );
      for( i = 0; i < 4; i++ )
      {
          CvPoint2D32f pt = cvPointTo32f(*(CvPoint*)cvGetSeqElem(src_contour, i));
          CvCBCorner* corner = &(*out_corners)[quad_count*4 + i];

          memset( corner, 0, sizeof(*corner) );
          corner->pt = pt;
          q->corners[i] = corner;
      }
      q->edge_len = FLT_MAX;
      for( i = 0; i < 4; i++ )
      {
          float dx = q->corners[i]->pt.x - q->corners[(i+1)&3]->pt.x;
          float dy = q->corners[i]->pt.y - q->corners[(i+1)&3]->pt.y;
          float d = dx*dx + dy*dy;
          if( q->edge_len > d )
              q->edge_len = d;
      }
      quad_count++;
  }

  return quad_count;
}

static void icvFindQuadNeighbors( CvCBQuad *quads, int quad_count )
{
    const float thresh_scale = 1.f;
    int idx, i, k, j;
    float dx, dy, dist;

    // find quad neighbors
    for( idx = 0; idx < quad_count; idx++ )
    {
        CvCBQuad* cur_quad = &quads[idx];

        // choose the points of the current quadrangle that are close to
        // some points of the other quadrangles
        // (it can happen for split corners (due to dilation) of the
        // checker board). Search only in other quadrangles!

        // for each corner of this quadrangle
        for( i = 0; i < 4; i++ )
        {
            CvPoint2D32f pt;
            float min_dist = FLT_MAX;
            int closest_corner_idx = -1;
            CvCBQuad *closest_quad = 0;
            CvCBCorner *closest_corner = 0;

            if( cur_quad->neighbors[i] )
                continue;

            pt = cur_quad->corners[i]->pt;

            // find the closest corner in all other quadrangles
            for( k = 0; k < quad_count; k++ )
            {
                if( k == idx )
                    continue;

                for( j = 0; j < 4; j++ )
                {
                    if( quads[k].neighbors[j] )
                        continue;

                    dx = pt.x - quads[k].corners[j]->pt.x;
                    dy = pt.y - quads[k].corners[j]->pt.y;
                    dist = dx * dx + dy * dy;

                    if( dist < min_dist &&
                        dist <= cur_quad->edge_len*thresh_scale &&
                        dist <= quads[k].edge_len*thresh_scale )
                    {
                        // check edge lengths, make sure they're compatible
                        // edges that are different by more than 1:4 are rejected
                        float ediff = cur_quad->edge_len - quads[k].edge_len;
                        if (ediff > 32*cur_quad->edge_len ||
                            ediff > 32*quads[k].edge_len)
                        {
                            std::cerr << "Incompatible edge lengths" << std::endl;
                            continue;
                        }
                        closest_corner_idx = j;
                        closest_quad = &quads[k];
                        min_dist = dist;
                    }
                }
            }

            // we found a matching corner point?
            if( closest_corner_idx >= 0 && min_dist < FLT_MAX )
            {
                // If another point from our current quad is closer to the found corner
                // than the current one, then we don't count this one after all.
                // This is necessary to support small squares where otherwise the wrong
                // corner will get matched to closest_quad;
                closest_corner = closest_quad->corners[closest_corner_idx];

                for( j = 0; j < 4; j++ )
                {
                    if( cur_quad->neighbors[j] == closest_quad )
                        break;

                    dx = closest_corner->pt.x - cur_quad->corners[j]->pt.x;
                    dy = closest_corner->pt.y - cur_quad->corners[j]->pt.y;

                    if( dx * dx + dy * dy < min_dist )
                        break;
                }

                if( j < 4 || cur_quad->count >= 4 || closest_quad->count >= 4 )
                    continue;

                // Check that each corner is a neighbor of different quads
                for( j = 0; j < closest_quad->count; j++ )
                {
                    if( closest_quad->neighbors[j] == cur_quad )
                        break;
                }
                if( j < closest_quad->count )
                    continue;

                // check whether the closest corner to closest_corner
                // is different from cur_quad->corners[i]->pt
                for( k = 0; k < quad_count; k++ )
                {
                    CvCBQuad* q = &quads[k];
                    if( k == idx || q == closest_quad )
                        continue;

                    for( j = 0; j < 4; j++ )
                        if( !q->neighbors[j] )
                        {
                            dx = closest_corner->pt.x - q->corners[j]->pt.x;
                            dy = closest_corner->pt.y - q->corners[j]->pt.y;
                            dist = dx*dx + dy*dy;
                            if( dist < min_dist )
                                break;
                        }
                    if( j < 4 )
                        break;
                }

                if( k < quad_count )
                    continue;

                closest_corner->pt.x = (pt.x + closest_corner->pt.x) * 0.5f;
                closest_corner->pt.y = (pt.y + closest_corner->pt.y) * 0.5f;

                // We've found one more corner - remember it
                cur_quad->count++;
                cur_quad->neighbors[i] = closest_quad;
                cur_quad->corners[i] = closest_corner;

                closest_quad->count++;
                closest_quad->neighbors[closest_corner_idx] = cur_quad;
            }
        }
    }
}

//-----------------------------------------------------------------------------
std::vector<Point2f> RemoveClosePoints(const std::vector<Point2f> &input)
{
  std::vector<Point2f> output;
  std::vector<Point2f> pointsToAverage;

  std::vector<Point2f> copyOfInput = input;

  std::vector<Point2f>::iterator iter1;
  std::vector<Point2f>::iterator iter2;

  iter1 = copyOfInput.begin();
  while (iter1 != copyOfInput.end() && copyOfInput.size() > 0)
  {
    Point2f testPoint = *iter1;

    pointsToAverage.clear();
    pointsToAverage.push_back(testPoint);
    iter1 = copyOfInput.erase(iter1);

    int pointCounter = 0;
    for (iter2 = iter1; iter2 != copyOfInput.end() && copyOfInput.size() > 0; iter2++)
    {
      Point2f otherPoint = *iter2;
      double distance = sqrt((testPoint.x - otherPoint.x)*(testPoint.x - otherPoint.x) + (testPoint.y - otherPoint.y)*(testPoint.y - otherPoint.y));

      if (distance < averagingDistance)
      {
        iter2 = copyOfInput.erase(iter2);
        pointsToAverage.push_back(otherPoint);
        if (iter2 == copyOfInput.end())
        {
          break;
        }
      }
      pointCounter++;
    }
    Point2f averagePoint;
    averagePoint.x = 0;
    averagePoint.y = 0;
    for (unsigned int i = 0; i < pointsToAverage.size(); i++)
    {
      averagePoint.x += pointsToAverage[i].x;
      averagePoint.y += pointsToAverage[i].y;
    }
    averagePoint.x /= (double)pointsToAverage.size();
    averagePoint.y /= (double)pointsToAverage.size();

    output.push_back(averagePoint);
  }
  return output;
}


//-----------------------------------------------------------------------------
void ExtractCorners(int, void*)
{
  std::cerr << "Matt, ExtractCorners start" << std::endl;

  std::vector<Point2f> featuresLeft;
  std::vector<Point2f> featuresRight;
  std::vector<Point3f> features3D;

  // Output visualised on undistorted colour images (colour, so we can draw coloured lines on it).
  cornerDstLeft = cornerSrcLeftUndistorted.clone();
  cornerDstRight = cornerSrcRightUndistorted.clone();

  cv::blur( cornerSrcGreyLeft, cornerSrcGreyBlurLeft, Size(cornerGaussianBlur,cornerGaussianBlur) );
  cv::goodFeaturesToTrack(cornerSrcGreyBlurLeft, featuresLeft, cornerMaxCorners, qualityLevel/100.0, minDistance);

  std::vector<Point2f> filteredFeaturesLeft = RemoveClosePoints(featuresLeft);
  cv::cornerSubPix(cornerSrcGreyBlurLeft, filteredFeaturesLeft, Size(5,5), Size(2,2), cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 ));

  int numberOfFilteredPointsLeft = filteredFeaturesLeft.size();
  for (unsigned int i = 0; i < numberOfFilteredPointsLeft; i++)
  {
    cv::circle(cornerDstLeft, filteredFeaturesLeft[i], 8, CV_RGB(255, 0, 0), 1);
  }

  cv::blur( cornerSrcGreyRight, cornerSrcGreyBlurRight, Size(cornerGaussianBlur,cornerGaussianBlur) );
  cv::goodFeaturesToTrack(cornerSrcGreyBlurRight, featuresRight, cornerMaxCorners, qualityLevel/100.0, minDistance);

  std::vector<Point2f> filteredFeaturesRight = RemoveClosePoints(featuresRight);
  cv::cornerSubPix(cornerSrcGreyBlurRight, filteredFeaturesRight, Size(5,5), Size(2,2), cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 ));

  int numberOfFilteredPointsRight = filteredFeaturesRight.size();
  for (unsigned int i = 0; i < numberOfFilteredPointsRight; i++)
  {
    cv::circle(cornerDstRight, filteredFeaturesRight[i], 8, CV_RGB(255, 0, 0), 1);
  }

  std::cerr << "Matt, feature count, left=" << numberOfFilteredPointsLeft << ", right=" << numberOfFilteredPointsRight << std::endl;

  cv::Mat leftPointsInMatrix = cv::Mat(numberOfFilteredPointsLeft, 2, CV_32FC1);
  for (unsigned int i = 0; i < numberOfFilteredPointsLeft; i++)
  {
    leftPointsInMatrix.at<float>(i,0) = filteredFeaturesLeft[i].x;
    leftPointsInMatrix.at<float>(i,1) = filteredFeaturesLeft[i].y;
  }

  cv::Mat epiLines = cv::Mat(numberOfFilteredPointsLeft, 3, CV_32FC1);
  cv::computeCorrespondEpilines(leftPointsInMatrix, 1, fundamental, epiLines);

  std::cerr << "Matt, lineChoice=" << lineChoice << std::endl;

  // Plot lines in right window, corresponding to a chosen point in left window
  cv::circle(cornerDstLeft, filteredFeaturesLeft[lineChoice], 8, CV_RGB(0, 255, 0), 1);

  Point2f intersections[4];
  Point2f endpoints[2];

  intersections[0].x = 0;
  intersections[0].y = -epiLines.at<float>(lineChoice,2) / epiLines.at<float>(lineChoice,1);

  intersections[1].y = 0;
  intersections[1].x = -epiLines.at<float>(lineChoice,2) / epiLines.at<float>(lineChoice,0);

  intersections[2].x = cornerDstLeft.cols - 1;
  intersections[2].y = (-epiLines.at<float>(lineChoice,2) - epiLines.at<float>(lineChoice,0)*intersections[2].x) / epiLines.at<float>(lineChoice,1);

  intersections[3].y = cornerDstLeft.rows - 1;
  intersections[3].x = (-epiLines.at<float>(lineChoice,2) - epiLines.at<float>(lineChoice,1)*intersections[3].y) / epiLines.at<float>(lineChoice,0);

  int pointIndex = 0;
  for (int i = 0; i < 4; i++)
  {
    if (intersections[i].x >= 0 && intersections[i].x < cornerDstLeft.cols
        && intersections[i].y >= 0 && intersections[i].y < cornerDstLeft.rows
        && pointIndex < 2
        )
    {
      endpoints[pointIndex].x = intersections[i].x;
      endpoints[pointIndex].y = intersections[i].y;
      pointIndex++;
    }
  }

  if (pointIndex == 2)
  {
    cv::line( cornerDstRight, endpoints[0], endpoints[1], Scalar(255,0,0), 1, CV_AA);
  }


  // For each point in left image, try to find point in right image
  typedef std::vector< Point2f > MatchingPointsType;
  typedef std::pair<Point2f, MatchingPointsType > LeftPointToRightPointPairsType;
  typedef std::vector<LeftPointToRightPointPairsType> LeftPointToRightPointMatchesType;

  typedef std::pair<Point2f, Point2f> PointPairType;
  typedef std::vector<PointPairType> PointPairsType;

  LeftPointToRightPointMatchesType matches;
  PointPairsType pairs;

  // At the moment, compute a list of candidate matches
  for (int i = 0; i < numberOfFilteredPointsLeft; i++)
  {
    Point2f leftPoint = filteredFeaturesLeft[i];
    MatchingPointsType rightHandMatches;

    double a = epiLines.at<float>(i,0);
    double b = epiLines.at<float>(i,1);
    double c = epiLines.at<float>(i,2);

    for (int j = 0; j < numberOfFilteredPointsRight; j++)
    {
      Point2f rightPoint = filteredFeaturesRight[j];
      double x = rightPoint.x;
      double y = rightPoint.y;

      double distanceRightPointToLine = fabs(a*x + b*y + c) \
                                              / (sqrt(a*a + b*b))   \
                                              ;

      if ( distanceRightPointToLine < 5 )
      {
        rightHandMatches.push_back(rightPoint);
        pairs.push_back(PointPairType(leftPoint, rightPoint));

        std::cerr << "Matt, l=" << leftPoint.x << ", " << leftPoint.y << ", r=" << rightPoint.x << ", " << rightPoint.y << std::endl;
      }
    }
    if (rightHandMatches.size() > 0)
    {
      matches.push_back(LeftPointToRightPointPairsType(leftPoint, rightHandMatches));
    }
  }
  std::cerr << "Matt, numberOfFilteredPointsLeft=" << numberOfFilteredPointsLeft << ", pairs=" << pairs.size() << std::endl;
  std::cerr << "Matt, matched " << matches.size() << std::endl;

  //features3D = TriangulatePointPairs(pairs, leftIntrinsic, rightIntrinsic, r2lRotation, r2lTranslation);
  features3D = TriangulatePointPairs(pairs, leftIntrinsic, leftRotation, leftTranslation, rightIntrinsic, rightRotation, rightTranslation);

  // Dump out VTK file.
  vtkPolyData *polyData = vtkPolyData::New();

  vtkPoints *points = vtkPoints::New();
  points->SetDataTypeToDouble();
  points->Initialize();

  for (int i = 0; i < features3D.size(); i++)
  {
    points->InsertNextPoint(features3D[i].x, features3D[i].y, features3D[i].z);
  }
  polyData->SetPoints(points);

  vtkPolyDataWriter *polyWriter = vtkPolyDataWriter::New();
  polyWriter->SetFileName("/tmp/tmp.vtk");
  polyWriter->SetInput(polyData);
  polyWriter->SetFileTypeToASCII();
  polyWriter->Write();


  imshow( leftWindowName, cornerDstLeft );
  imshow( rightWindowName, cornerDstRight );


  /*
  CvCBQuad *quads = 0, **quad_group = 0;
  CvCBCorner *corners = 0, **corner_group = 0;
  cv::Ptr<CvMemStorage> storage;
  storage = cvCreateMemStorage(0);
  int flags = 0;
  int group_idx = 0;


  cv::adaptiveThreshold(cornerSrcGreyBlur, cornerSrcGreyBlurThresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, meanWindow, -meanC);
  cv::dilate( cornerSrcGreyBlurThresh, cornersSrcGreyBlurThreshDilate, Mat(), cv::Point(-1, -1), dilations-1 );
  */

  /*
  CvMat oldmat = cornersSrcGreyBlurThreshDilate;

  int quad_count = 0;

  quad_count = icvGenerateQuads( &quads, &corners, storage, &oldmat, flags );

  if( quad_count <= 0 )
      return;

  // Find quad's neighbors
  icvFindQuadNeighbors( quads, quad_count );


  for(int i = 0; i < quad_count; i++ )
  {
      for (int k=0; k<4; k++)
      {
          CvPoint2D32f pt1, pt2;
          CvScalar color = CV_RGB(30,255,30);
          pt1 = quads[i].corners[k]->pt;
          pt2 = quads[i].corners[(k+1)%4]->pt;
          pt2.x = (pt1.x + pt2.x)/2;
          pt2.y = (pt1.y + pt2.y)/2;
          if (k>0)
              color = CV_RGB(200,200,0);
          line( cornerDst, Point(pt1.x, pt1.y), Point(pt2.x, pt2.y), Scalar(255,0,0), 1, CV_AA);

      }
  }
  std::cerr << "Matt, first part chessboard found " << quad_count << " quads " << std::endl;

  CvMemStorage* storage2;
  storage2 = cvCreateMemStorage(0);

  CvSeq* contours;
  CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvSeq), storage2);
  CvSeq* square_contours = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvSeq), storage2);

  cvFindContours(&oldmat, storage2, &contours, sizeof(CvContour),
      CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0,0));

  int contourCounter = 0;
  int squaresCounter = 0;
  while(contours)
  {
      if(contours->total < 4)
      {
          contours = contours->h_next;
          contourCounter++;
          continue;
      }

      CvSeq* result = cvApproxPoly(contours, sizeof(CvContour), storage,
                                   CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.035, 0 ); // TODO: Parameters?

      if( result->total == 4
          && fabs(cvContourArea(result,CV_WHOLE_SEQ)) > 50
          && cvCheckContourConvexity(result) )
      {
              cvSeqPush(squares, result);
              cvSeqPush(square_contours, contours);
              squaresCounter++;
      }
      contours = contours->h_next;
      contourCounter++;
  }

  std::cerr << "Matt, cvFindContours then filtering polys found " << contourCounter << " contours and " << squaresCounter << " squares " << std::endl;

  int _n_blobs = squares->total;

  // For every detected 4-corner blob
  for(int i = 0; i < _n_blobs; ++i)
  {
      CvSeq* sq = (CvSeq*)cvGetSeqElem(squares, i);
      CvSeq* square_contour = (CvSeq*)cvGetSeqElem(square_contours, i);

      for(int j = 0; j < 4; ++j)
      {
          CvPoint* pt0 = (CvPoint*)cvGetSeqElem(sq, j);
          CvPoint* pt1 = (CvPoint*)cvGetSeqElem(sq, (j+1)%4);
          line( cornerDst, Point(pt0->x, pt0->y), Point(pt1->x, pt1->y), Scalar(0,0,255), 1, CV_AA);
      }
  }
*/

  std::cerr << "Matt, ExtractCorners finish" << std::endl;
}


//-----------------------------------------------------------------------------
TestCornerExtraction::TestCornerExtraction()
{
}


//-----------------------------------------------------------------------------
TestCornerExtraction::~TestCornerExtraction()
{

}


//-----------------------------------------------------------------------------
void TestCornerExtraction::Run(const std::string& fileNameLeft, const std::string& fileNameRight)
{
  /// Load data.
  cornerSrcLeft = imread( fileNameLeft );
  if( !cornerSrcLeft.data )
  {
    return;
  }
  cornerSrcRight = imread( fileNameRight );
  if( !cornerSrcRight.data )
  {
    return;
  }

  // Fake some matrices, from a previous calibration.
  leftIntrinsic = cv::Mat(3,3, CV_64FC1);
  leftIntrinsic.at<double>(0,0) = 2.00732715e+03;
  leftIntrinsic.at<double>(0,1) = 0;
  leftIntrinsic.at<double>(0,2) = 9.45485901e+02;
  leftIntrinsic.at<double>(1,0) = 0;
  leftIntrinsic.at<double>(1,1) = 2.01372095e+03;
  leftIntrinsic.at<double>(1,2) = 6.19069702e+02;
  leftIntrinsic.at<double>(2,0) = 0;
  leftIntrinsic.at<double>(2,1) = 0;
  leftIntrinsic.at<double>(2,2) = 1;

  leftDistortion = cv::Mat(1,5, CV_64FC1);
  leftDistortion.at<double>(0,0) = -2.37351358e-01;
  leftDistortion.at<double>(0,1) = -2.44029015e-01;
  leftDistortion.at<double>(0,2) = 2.99157645e-03;
  leftDistortion.at<double>(0,3) = -1.41141098e-03;
  leftDistortion.at<double>(0,4) = 1.35881567e+00;

  leftRotation = cv::Mat(1,3, CV_64FC1);
  leftRotation.at<double>(0,0) = 2.01146626e+00;
  leftRotation.at<double>(0,1) = 2.12203932e+00;
  leftRotation.at<double>(0,2) = -7.33722985e-01;

  leftTranslation = cv::Mat(1,3, CV_64FC1);
  leftTranslation.at<double>(0,0) = 9.79845142e+00;
  leftTranslation.at<double>(0,1) = -2.25804710e+01;
  leftTranslation.at<double>(0,2) = 1.27580696e+02;

  rightIntrinsic = cv::Mat(3,3, CV_64FC1);
  rightIntrinsic.at<double>(0,0) = 2.03054236e+03;
  rightIntrinsic.at<double>(0,1) = 0;
  rightIntrinsic.at<double>(0,2) = 1.04627478e+03;
  rightIntrinsic.at<double>(1,0) = 0;
  rightIntrinsic.at<double>(1,1) = 2.04631201e+03;
  rightIntrinsic.at<double>(1,2) = 5.52110168e+02;
  rightIntrinsic.at<double>(2,0) = 0;
  rightIntrinsic.at<double>(2,1) = 0;
  rightIntrinsic.at<double>(2,2) = 1;

  rightDistortion = cv::Mat(1,5, CV_64FC1);
  rightDistortion.at<double>(0,0) = -2.08332404e-01;
  rightDistortion.at<double>(0,1) = -4.93973404e-01;
  rightDistortion.at<double>(0,2) = -6.53384300e-03;
  rightDistortion.at<double>(0,3) = 3.38898215e-04;
  rightDistortion.at<double>(0,4) = 2.31051683e+00;

  rightRotation = cv::Mat(1,3, CV_64FC1);
  rightRotation.at<double>(0,0) = 2.00048733e+00;
  rightRotation.at<double>(0,1) = 2.12012887e+00;
  rightRotation.at<double>(0,2) = -7.91201830e-01;

  rightTranslation = cv::Mat(1,3, CV_64FC1);
  rightTranslation.at<double>(0,0) = 8.76964664e+00;
  rightTranslation.at<double>(0,1) = -1.95697060e+01;
  rightTranslation.at<double>(0,2) = 1.28789444e+02;

  fundamental = cv::Mat(3, 3, CV_64FC1);
  fundamental.at<double>(0,0) = 2.29537292e-07;
  fundamental.at<double>(0,1) = 4.25394537e-05;
  fundamental.at<double>(0,2) = -4.30139564e-02;
  fundamental.at<double>(1,0) = -3.76993048e-05;
  fundamental.at<double>(1,1) = 3.36927155e-06;
  fundamental.at<double>(1,2) = -2.85123259e-01;
  fundamental.at<double>(2,0) = 3.88712324e-02;
  fundamental.at<double>(2,1) = 2.75174260e-01;
  fundamental.at<double>(2,2) = 1;

  r2lRotation = cv::Mat(1, 3, CV_64FC1);
  r2lRotation.at<double>(0,0) = 2.36975513e-02;
  r2lRotation.at<double>(0,1) = -3.09497118e-02 ;
  r2lRotation.at<double>(0,2) = -4.73383843e-04;

  r2lTranslation = cv::Mat(1, 3, CV_64FC1);
  r2lTranslation.at<double>(0,0) = 5.02080059e+00;
  r2lTranslation.at<double>(1,0) = 4.13856506e-02;
  r2lTranslation.at<double>(2,0) = -9.18434143e-01;

  // Quick hack to work round interlacing, on colour images, before we undistort them.
  uint8_t* pixelPtr = (uint8_t*)cornerSrcLeft.data;
  for(int i = 0; i < cornerSrcLeft.rows; i+=2)
  {
      for(int j = 0; j < cornerSrcLeft.cols; j++)
      {
          pixelPtr[(i+1)*cornerSrcLeft.cols*3 + j*3 + 0] = pixelPtr[i*cornerSrcLeft.cols*3 + j*3 + 0];
          pixelPtr[(i+1)*cornerSrcLeft.cols*3 + j*3 + 1] = pixelPtr[i*cornerSrcLeft.cols*3 + j*3 + 1];
          pixelPtr[(i+1)*cornerSrcLeft.cols*3 + j*3 + 2] = pixelPtr[i*cornerSrcLeft.cols*3 + j*3 + 2];
      }
  }
  pixelPtr = (uint8_t*)cornerSrcRight.data;
  for(int i = 0; i < cornerSrcRight.rows; i+=2)
  {
      for(int j = 0; j < cornerSrcRight.cols; j++)
      {
          pixelPtr[(i+1)*cornerSrcRight.cols*3 + j*3 + 0] = pixelPtr[i*cornerSrcRight.cols*3 + j*3 + 0];
          pixelPtr[(i+1)*cornerSrcRight.cols*3 + j*3 + 1] = pixelPtr[i*cornerSrcRight.cols*3 + j*3 + 1];
          pixelPtr[(i+1)*cornerSrcRight.cols*3 + j*3 + 2] = pixelPtr[i*cornerSrcRight.cols*3 + j*3 + 2];
      }
  }

  // Create some images
  cornerSrcLeftUndistorted.create( cornerSrcLeft.size(), cornerSrcLeft.type() );
  cornerSrcRightUndistorted.create( cornerSrcRight.size(), cornerSrcRight.type() );

  cornerSrcLeftRectified.create( cornerSrcLeft.size(), cornerSrcLeft.type() );
  cornerSrcRightRectified.create( cornerSrcRight.size(), cornerSrcRight.type() );

  // Undistort images.
  cv::undistort(cornerSrcLeft, cornerSrcLeftUndistorted, leftIntrinsic, leftDistortion);
  cv::undistort(cornerSrcRight, cornerSrcRightUndistorted, rightIntrinsic, rightDistortion);

  // Calculate Rectification Transform, to get out R1, R2, P1, P2, Q etc.
  /*
  cv::stereoRectify(
      leftIntrinsic, leftDistortion,
      rightIntrinsic, rightDistortion,
      cv::Size(width,height),
      r2lRotation, r2lTranslation,
      R1, R2, P1, P2, Q, -1, 0
      );
  */

  // Rectify images
  /*
  cv::initUndistortRectifyMap(leftIntrinsic, leftDistortion, R1, P1, cv::Size(width,height), CV_16SC2, leftMapX, leftMapY);
  cv::initUndistortRectifyMap(rightIntrinsic, rightDistortion, R2, P2, cv::Size(width,height), CV_16SC2, rightMapX, rightMapY);
  cv::remap(cornerSrcLeft, cornerSrcLeftRectified, leftMapX, leftMapY, INTER_LINEAR);
  cv::remap(cornerSrcRight, cornerSrcRightRectified, rightMapX, rightMapY, INTER_LINEAR);
  */

  // Convert both to grey
  cvtColor( cornerSrcLeftUndistorted, cornerSrcGreyLeft, CV_RGB2GRAY );
  cvtColor( cornerSrcRightUndistorted, cornerSrcGreyRight, CV_RGB2GRAY );
  /*
  cvtColor( cornerSrcLeftRectified, cornerSrcLeftRectifiedGrey, CV_RGB2GRAY );
  cvtColor( cornerSrcRightRectified, cornerSrcRightRectifiedGrey, CV_RGB2GRAY );
  */

  // Create the windows
  namedWindow( leftWindowName );
  namedWindow( rightWindowName );

  // These create sliders.

  //createTrackbar( "Blur", cornerWindowName, &cornerGaussianBlur, 255, ExtractCorners );
  //createTrackbar( "Mean Window", cornerWindowName, &meanWindow, 255, ExtractCorners );
  //createTrackbar( "Mean C", cornerWindowName, &meanC, 255, ExtractCorners );
  //createTrackbar( "Dilations", cornerWindowName, &dilations, 255, ExtractCorners );
  createTrackbar( "Max corners", leftWindowName, &cornerMaxCorners, 255, ExtractCorners );
  createTrackbar( "Quality Level", leftWindowName, &qualityLevel, 100, ExtractCorners );
  createTrackbar( "Min Distance", leftWindowName, &minDistance, 255, ExtractCorners );
  createTrackbar( "Average Distance", leftWindowName, &averagingDistance, 255, ExtractCorners );
  createTrackbar( "Epi Distance", leftWindowName, &epiPolarDistance, 255, ExtractCorners );
  createTrackbar( "Line choice", leftWindowName, &lineChoice, 255, ExtractCorners );

  // Call this at least once to get an initial rendering.
  ExtractCorners(0,0);

  // Wait until user exit program by pressing a key
  waitKey(0);
}

//-----------------------------------------------------------------------------
} // end namespace
