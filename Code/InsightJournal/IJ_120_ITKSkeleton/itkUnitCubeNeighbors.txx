#ifndef itkUnitCubeNeighbors_txx
#define itkUnitCubeNeighbors_txx

#include "itkUnitCubeNeighbors.h"

namespace itk
{

template<typename Connectivity, typename NeighborhoodConnectivity>
UnitCubeNeighbors<Connectivity, NeighborhoodConnectivity>
::UnitCubeNeighbors()
: neighborhoodSize(
    static_cast<int>(std::pow(3.f, static_cast<float>(Connectivity::Dimension)))), 
  neighborsInUnitCube(
    std::vector<std::vector<bool> >(
      neighborhoodSize , std::vector<bool>(neighborhoodSize, false) ) )
  {
  assert(Connectivity::Dimension == NeighborhoodConnectivity::Dimension);
  Connectivity const & connectivity = Connectivity::GetInstance();
  NeighborhoodConnectivity const & neighborhoodConnectivity = 
    NeighborhoodConnectivity::GetInstance();
  
  for(int neighbor1 = 0; neighbor1 < neighborhoodSize; ++neighbor1)
    {
    // convert i to Connectivity::OffsetType
    Point const p1 = connectivity.OffsetToPoint(neighbor1);
            
    if(neighborhoodConnectivity.IsInNeighborhood(p1) )
      {
      for(int neighbor2 = 0; neighbor2 < neighborhoodSize; ++neighbor2)
        {
        // convert i to Connectivity::OffsetType
        Point const p2 = connectivity.OffsetToPoint(neighbor2);
        
        Point sum;
        for(unsigned int i=0; i<Connectivity::Dimension; ++i)
          {
          sum[i] = p1[i] + p2[i];
          }
        unsigned int const sumOffset = connectivity.PointToOffset(sum);
        
        bool inUnitCube = true;
        for(unsigned int dim = 0; 
            dim < Connectivity::Dimension && inUnitCube; ++dim)
          {
          if(sum[dim] < -1 || sum[dim] > +1) 
            {
            inUnitCube = false;
            }
          }
        
        if(inUnitCube && connectivity.AreNeighbors(p1, sum) )
          {
          neighborsInUnitCube[neighbor1][sumOffset] = true;
          }
        }
      }
    }
  }


template<typename Connectivity, typename NeighborhoodConnectivity>
bool 
UnitCubeNeighbors<Connectivity, NeighborhoodConnectivity>
::operator()(typename UnitCubeNeighbors<Connectivity, 
               NeighborhoodConnectivity>::Offset const o1, 
             typename UnitCubeNeighbors<Connectivity, 
               NeighborhoodConnectivity>::Offset const o2) const
  {
  return neighborsInUnitCube[o1][o2];
  }


template<typename Connectivity, typename NeighborhoodConnectivity>
bool 
UnitCubeNeighbors<Connectivity, NeighborhoodConnectivity>
::operator()(typename UnitCubeNeighbors<Connectivity, 
               NeighborhoodConnectivity>::Point const p1, 
             typename UnitCubeNeighbors<Connectivity, 
               NeighborhoodConnectivity>::Point const p2) const
  {
  Connectivity const & connectivity = Connectivity::getInstance();
  Offset const o1 = connectivity.pointToOffset(p1);
  Offset const o2 = connectivity.pointToOffset(p2);
  return operator()(o1, o2);
  }

}

#endif // itkUnitCubeNeighbors_txx
