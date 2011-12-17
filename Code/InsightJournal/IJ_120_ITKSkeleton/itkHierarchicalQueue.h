#ifndef itkHierarchicalQueue_h
#define itkHierarchicalQueue_h

#include <queue>
#include <map>
#include <functional>

namespace itk
{

/**
 *  @brief Hierarchical queue.
 *
 *  The hierarchical queue can be seen as a multiple regular queue. Each
 *  element in the queue has a priority and a value. Elements of higher
 *  priority are served before those of lower priority. Elements of same
 *  priority are served in FIFO order.
 *
 *  The ``higher priority'' relation is not necessarily ``greater than'' : it
 *  depends on the given comparison criterion. The default behaviour is as such,
 *  but you could choose to use std::less<TPriority> to have point with a lower
 *  priority value being served first.
 */
template<typename TPriority, typename TValue, 
         typename TCompare = std::greater<TPriority> >
class ITK_EXPORT HierarchicalQueue
  {
  public:

    typedef HierarchicalQueue<TPriority, TValue, TCompare> Self;

    /**
     *  @brief Type of the queue's values.
     */
    typedef TValue ValueType;

    /**
     *  @brief Type of the queue's priority.
     */
    typedef TPriority PriorityType;

    /**
     *  @brief Type of the queue's size (number of elements).
     */
    typedef unsigned long int SizeType;

    /**
     *  @brief Create an empty hierarchical queue.
     */
    HierarchicalQueue();
    
    /**
     *  @brief Copy constructor
     */
    HierarchicalQueue(Self const & src);
    
    /** 
     *  @brief Destructor
     */
    ~HierarchicalQueue();
    
    /**
     *  @brief Add an element to the queue at given priority
     *  @param p : priority of the element to add
     *  @param v : value of the element
     */
    void Push(PriorityType p, ValueType v);
    
    /**
     *  @brief Remove the front element of the queue with the highest priority.
     *  @pre The queue must not be empty.
     */
    void Pop();
    
    /**
     *  @brief Return whether the hierarchical queue is empty.
     */
    bool IsEmpty();
    
    /**
     *  @brief Number of elements in the hierarchical queue.
     */
    SizeType GetSize() const;
    
    /**
     *  @brief Front element of the queue with the highest priority.
     *  @pre The queue must not be empty.
     */
    ValueType const & GetFront() const;
    
    /**
     *  @brief Highest (current) priority in the queue.
     *  @pre The queue must not be empty.
     */
    PriorityType const GetPriority() const;

  private:
    /**
     *  @brief Container's type.
     */
    typedef std::map<TPriority, std::queue<TValue>, TCompare>  Container;
    
    /**
     *  @brief Comparison criterion.
     */
    TCompare m_Compare;
    
    /**
     *  @brief Internal representation of data.
     */
    Container m_Container;
    
    /**
     *  @brief Current priority.
     */
    PriorityType m_CurrentPriority;
    
    /**
     *  @brief Size of the queue (number of elements).
     */
    SizeType m_Size; 
  };

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHierarchicalQueue.txx"
#endif

#endif // hierarchical_queue_h
