#ifndef itkNeighborhoodConnectivity_h
#define itkNeighborhoodConnectivity_h

#include "itkConnectivity.h"

namespace itk
{

/**
 * @brief Helper to determine the neighborhood connectivity to use when 
 * computing topological numbers.
 * @param Connectivity : the image connectivity
 */
template<typename TConnectivity>
struct ITK_EXPORT NeighborhoodConnectivity
  {
  typedef TConnectivity Type;
  };


template<>
struct ITK_EXPORT NeighborhoodConnectivity< Connectivity<2,1> >
  {
  typedef Connectivity<2,0> Type;
  };


template<>
struct ITK_EXPORT NeighborhoodConnectivity< Connectivity<3,2> >
  {
  typedef Connectivity<3,1> Type;
  };

}

#endif // itkNeighborhoodConnectivity_h
