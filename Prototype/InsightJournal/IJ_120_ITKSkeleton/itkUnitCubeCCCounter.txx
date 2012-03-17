#ifndef itkUnitCubeCCCounter_txx
#define itkUnitCubeCCCounter_txx

#include "itkUnitCubeCCCounter.h"

#include <list>
#include <queue>

namespace itk
{

template<typename TConnectivity, typename TNeighborhoodConnectivity>
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::UnitCubeCCCounter()
: m_Image(new char[TConnectivity::GetInstance().GetNeighborhoodSize()])
  {
  }


template<typename TConnectivity, typename TNeighborhoodConnectivity>
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::~UnitCubeCCCounter()
  {
  delete[] m_Image;
  }


template<typename TConnectivity, typename TNeighborhoodConnectivity>
template<typename Iterator>
void
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::SetImage(Iterator imageBegin, Iterator imageEnd)
  {
  std::copy(imageBegin, imageEnd, m_Image);
  }

template<typename TConnectivity, typename TNeighborhoodConnectivity>
unsigned int 
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::operator()() const
  {
  unsigned int const neighborhoodSize = 
    TConnectivity::GetInstance().GetNeighborhoodSize();
  unsigned int seed=0;
  // Find first seed
  while(seed != neighborhoodSize && 
         (m_Image[seed] == 0 || !m_ConnectivityTest[seed] ) )
    {
    ++seed;
    }

  std::vector<bool> processed(neighborhoodSize, false);
  unsigned int nbCC=0;
  while(seed != neighborhoodSize)
    {
    ++nbCC;
    processed[seed] = true;
    
    std::queue<unsigned int, std::list<unsigned int> > q;
    q.push(seed);
    
    while(!q.empty())
      {
      unsigned int const current = q.front();
      q.pop();
      
      // For each neighbor check if m_UnitCubeNeighbors is true.
      for(unsigned int neighbor = 0; neighbor < neighborhoodSize; ++neighbor)
        {
        if(!processed[neighbor] && m_Image[neighbor] !=0 && 
           m_UnitCubeNeighbors(current, neighbor))
          {
          q.push(neighbor);
          processed[neighbor] = true;
          }
        }
      }
    
    // Look for next seed
    while(seed != neighborhoodSize && 
          ( processed[seed] || m_Image[seed] == 0 || !m_ConnectivityTest[seed] ) 
         )
      {
      ++seed;
      }
    }
  return nbCC;
  }


template<typename TConnectivity, typename TNeighborhoodConnectivity>
template<typename C>
std::vector<bool>
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::CreateConnectivityTest()
  {
  C const & connectivity = C::GetInstance();
  unsigned int const neighborhoodSize = connectivity.GetNeighborhoodSize();
  std::vector<bool> test(neighborhoodSize, false);
  for(unsigned int i=0; i<neighborhoodSize; ++i)
    {
    test[i] = connectivity.IsInNeighborhood(i);
    }
  
  return test;
  }


template<typename TConnectivity, typename TNeighborhoodConnectivity>
std::vector<bool> const
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::m_NeighborhoodConnectivityTest = 
    UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
        ::template CreateConnectivityTest< TNeighborhoodConnectivity >();


template<typename TConnectivity, typename TNeighborhoodConnectivity>
std::vector<bool> const
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::m_ConnectivityTest = 
    UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
        ::template CreateConnectivityTest<TConnectivity>();


template<typename TConnectivity, typename TNeighborhoodConnectivity>
UnitCubeNeighbors<TConnectivity, TNeighborhoodConnectivity> const
UnitCubeCCCounter<TConnectivity, TNeighborhoodConnectivity>
::m_UnitCubeNeighbors = 
  UnitCubeNeighbors<TConnectivity, TNeighborhoodConnectivity>();

}

#endif // itkUnitCubeCCCounter_txx
