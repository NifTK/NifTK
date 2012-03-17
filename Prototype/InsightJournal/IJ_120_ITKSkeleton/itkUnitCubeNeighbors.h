#ifndef itkUnitCubeNeighbors_h
#define itkUnitCubeNeighbors_h

#include <vector>

#include "itkNeighborhoodConnectivity.h"

namespace itk
{

/**
 * @brief For each point in the n'-neighborhood of 0, characterizes which of 
 * its surrounding points are n-neighbors and belong to the [-1, 1] cube.
 *
 * @param Connectivity : the image connectivity, noted n in the description
 * @param NeighborhoodConnectivity : the neighborhood connectivity, noted n' 
 * in the description.
 *
 * In 3D, use (6,18) or (26,26) for (n, n') to get significant result for 
 * topological numbers.
 *
 * The criterion returns true for (p1, p2) if p1 is n'-adjacent to 0 and 
 * (p1, p1+p2) are n-adjacent and (p1+p2) is in [-1, 1]^3.
 */
template< typename Connectivity, 
          typename NeighborhoodConnectivity = 
            typename NeighborhoodConnectivity<Connectivity>::Type >
class ITK_EXPORT UnitCubeNeighbors
  {
  public :
    typedef typename Connectivity::Point Point;
    typedef typename Connectivity::Offset Offset;
    
    UnitCubeNeighbors();
    
    bool operator()(Point const p1, Point const p2) const;
    bool operator()(Offset const p1, Offset const p2) const;
      
  private :
    int const neighborhoodSize;    
    std::vector<std::vector<bool> > neighborsInUnitCube;
  };

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkUnitCubeNeighbors.txx"
#endif

#endif // itkUnitCubeNeighbors_h
