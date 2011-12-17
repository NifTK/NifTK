#ifndef itkHierarchicalQueue_txx
#define itkHierarchicalQueue_txx

#include "itkHierarchicalQueue.h"

namespace itk
{

template<typename TPriority, typename TValue, typename TCompare>
HierarchicalQueue<TPriority, TValue, TCompare>
::HierarchicalQueue() : 
    m_Compare(), m_Container(), m_Size(0)
  {
  }


template<typename TPriority, typename TValue, typename TCompare>
HierarchicalQueue<TPriority, TValue, TCompare>
::HierarchicalQueue(HierarchicalQueue<TPriority, TValue, TCompare> const & src) :
    m_Compare(src.m_Compare), m_Container(src.m_Container),  
    m_CurrentPriority(src.m_CurrentPriority), m_Size(src.m_Size)
  {
  }


template<typename TPriority, typename TValue, typename TCompare>
HierarchicalQueue<TPriority, TValue, TCompare>
::~HierarchicalQueue()
  {
  }


template<typename TPriority, typename TValue, typename TCompare>
void 
HierarchicalQueue<TPriority, TValue, TCompare>
::Push(PriorityType p, ValueType v)
  {            
  if(m_Size == 0)
    {
    m_CurrentPriority = p;
    }
  else if(m_Compare(p, m_CurrentPriority))
    {
    m_CurrentPriority = p;
    }
  // else : don't change the current priority
    
  m_Container[p].push(v);
  ++m_Size;
  }


template<typename TPriority, typename TValue, typename TCompare>
void 
HierarchicalQueue<TPriority, TValue, TCompare>
::Pop()
  {
  m_Container.begin()->second.pop();
  --m_Size;
  if(m_Container.begin()->second.empty())
    {
    m_Container.erase(m_Container.begin());
    if(m_Size != 0) m_CurrentPriority = m_Container.begin()->first;
    }
  }


template<typename TPriority, typename TValue, typename TCompare>
bool 
HierarchicalQueue<TPriority, TValue, TCompare>
::IsEmpty()
  {
  return (m_Size == 0);
  }


template<typename TPriority, typename TValue, typename TCompare>
typename HierarchicalQueue<TPriority, TValue, TCompare>::SizeType
HierarchicalQueue<TPriority, TValue, TCompare>
::GetSize() const
  {
  return m_Size;
  }


template<typename TPriority, typename TValue, typename TCompare>
typename HierarchicalQueue<TPriority, TValue, TCompare>::ValueType const & 
HierarchicalQueue<TPriority, TValue, TCompare>
::GetFront() const
  {
  return m_Container.begin()->second.front();
  }


template<typename TPriority, typename TValue, typename TCompare>
typename HierarchicalQueue<TPriority, TValue, TCompare>::PriorityType const 
HierarchicalQueue<TPriority, TValue, TCompare>
::GetPriority() const
  {
  return m_CurrentPriority;
  }
    
}    

#endif // itkHierarchicalQueue_txx
