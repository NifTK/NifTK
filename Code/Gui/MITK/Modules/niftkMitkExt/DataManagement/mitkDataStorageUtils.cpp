/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-04 08:05:37 +0100 (Thu, 04 Aug 2011) $
 Revision          : $Revision: 6968 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkDataStorageUtils.h"
#include "mitkDataStorage.h"
#include "mitkMIDASTool.h"

namespace mitk
{
  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindFirstParentImage(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary)
  {
    mitk::DataNode::Pointer result = NULL;

    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    mitk::DataStorage::SetOfObjects::ConstPointer possibleParents = storage->GetSources( node, isImage );

    for (unsigned int i = 0; i < possibleParents->size(); i++)
    {

      mitk::DataNode* possibleNode = (*possibleParents)[i];

      bool isBinary;
      possibleNode->GetBoolProperty("binary", isBinary);

      if (isBinary == lookForBinary)
      {
        result = possibleNode;
      }
    }
    return result;
  }


  //-----------------------------------------------------------------------------
  mitk::DataStorage::SetOfObjects::Pointer FindDerivedVisibleNonHelperChildren(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node)
  {
    mitk::DataStorage::SetOfObjects::Pointer results = mitk::DataStorage::SetOfObjects::New();

    unsigned int counter = 0;
    mitk::DataStorage::SetOfObjects::ConstPointer possibleChildren = storage->GetDerivations( node, NULL, false);
    for (unsigned int i = 0; i < possibleChildren->size(); i++)
    {
      mitk::DataNode* possibleNode = (*possibleChildren)[i];

      bool isVisible = false;
      possibleNode->GetBoolProperty("visible", isVisible);

      bool isHelper = false;
      possibleNode->GetBoolProperty("helper object", isHelper);

      if (isVisible && !isHelper)
      {
        results->InsertElement(counter, possibleNode);
        counter++;
      }
    }
    return results;
  }


  //-----------------------------------------------------------------------------
  mitk::DataStorage::SetOfObjects::Pointer FindDerivedImages(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary )
  {
    mitk::DataStorage::SetOfObjects::Pointer results = mitk::DataStorage::SetOfObjects::New();

    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    mitk::DataStorage::SetOfObjects::ConstPointer possibleChildren = storage->GetDerivations( node, isImage, true);

    unsigned int counter = 0;
    for (unsigned int i = 0; i < possibleChildren->size(); i++)
    {

      mitk::DataNode* possibleNode = (*possibleChildren)[i];

      bool isBinary;
      possibleNode->GetBoolProperty("binary", isBinary);

      if (isBinary == lookForBinary)
      {
        results->InsertElement(counter, possibleNode);
        counter++;
      }
    }

    return results;
  }


  //-----------------------------------------------------------------------------
  bool IsNodeAHelperObject(const mitk::DataNode* node)
  {
    bool result = false;

    if (node != NULL)
    {
      bool isHelper(false);
      if (node->GetBoolProperty("helper object", isHelper) && isHelper)
      {
        result = true;
      }
    }

    return result;
  }


  //-----------------------------------------------------------------------------
  bool IsNodeAGreyScaleImage(const mitk::DataNode::Pointer node)
  {
    bool result = false;

    if (node.IsNotNull())
    {
      mitk::Image::Pointer image = static_cast<mitk::Image*>(node->GetData());
      if (image.IsNotNull())
      {
        bool isBinary(false);
        bool foundBinaryProperty = node->GetBoolProperty("binary", isBinary);

        if ((foundBinaryProperty && !isBinary) || !foundBinaryProperty)
        {
          result = true;
        }
      }
    }

    return result;
  }


  //-----------------------------------------------------------------------------
  bool IsNodeABinaryImage(const mitk::DataNode::Pointer node)
  {
    bool result = false;

    if (node.IsNotNull())
    {
      mitk::Image::Pointer image = static_cast<mitk::Image*>(node->GetData());
      if (image.IsNotNull())
      {
        bool isBinary(false);
        bool foundBinaryProperty = node->GetBoolProperty("binary", isBinary);

        if(foundBinaryProperty && isBinary)
        {
          result = true;
        }
      }
    }

    return result;
  }


  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindNthImage(const std::vector<mitk::DataNode*> &nodes, int n, bool lookForBinary)
  {
    if (nodes.empty()) return NULL;

    int numberOfMatchingNodesFound = 0;

    for(unsigned int i = 0; i < nodes.size(); ++i)
    {
        bool isImage(false);
        if (nodes.at(i)->GetData())
        {
          isImage = dynamic_cast<mitk::Image*>(nodes.at(i)->GetData()) != NULL;
        }

        bool isBinary;
        nodes.at(i)->GetBoolProperty("binary", isBinary);

        if (isImage && isBinary == lookForBinary)
        {
          numberOfMatchingNodesFound++;
          if (numberOfMatchingNodesFound == n)
          {
            return nodes.at(i);
          }
        }
    }
    return NULL;
  }


  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindNthGreyScaleImage(const std::vector<mitk::DataNode*> &nodes, int n )
  {
    return FindNthImage(nodes, n, false);
  }


  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindNthBinaryImage(const std::vector<mitk::DataNode*> &nodes, int n )
  {
    return FindNthImage(nodes, n, true);
  }


  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindFirstGreyScaleImage(const std::vector<mitk::DataNode*> &nodes )
  {
    return FindNthGreyScaleImage(nodes, 1);
  }


  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindFirstBinaryImage(const std::vector<mitk::DataNode*> &nodes )
  {
    return FindNthBinaryImage(nodes, 1);
  }


  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindFirstParent(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node)
  {
    mitk::DataNode::Pointer result = NULL;

    mitk::DataStorage::SetOfObjects::ConstPointer possibleParents = storage->GetSources(node);
    if (possibleParents->size() > 0)
    {
      result = (*possibleParents)[0];
    }
    return result;
  }


  //-----------------------------------------------------------------------------
  mitk::DataNode::Pointer FindParentGreyScaleImage(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node)
  {
    mitk::DataNode::Pointer result = NULL;

    if (node.IsNotNull())
    {
      mitk::DataNode::Pointer nodeToCheck = node;
      mitk::DataNode::Pointer parent = NULL;
      do
      {
        parent = FindFirstParent(storage, nodeToCheck);
        if (parent.IsNotNull())
        {
          if (IsNodeAGreyScaleImage(parent))
          {
            result = parent;
            break;
          }
          else
          {
            nodeToCheck = parent;
          }
        }
      } while (parent.IsNotNull());
    }
    return result;
  }


  //-----------------------------------------------------------------------------
  mitk::TimeSlicedGeometry::Pointer GetPreferredGeometry(const mitk::DataStorage* dataStorage, const std::vector<mitk::DataNode*>& nodes, const int& nodeIndex)
  {
    mitk::TimeSlicedGeometry::Pointer geometry = NULL;

    int indexThatWeActuallyUsed = -1;

    // If nodeIndex < 0, we are choosing the best geometry from all available nodes.
    if (nodeIndex < 0)
    {

      // First try to find an image geometry, and if so, use the first one.
      mitk::Image::Pointer image = NULL;
      for (unsigned int i = 0; i < nodes.size(); i++)
      {
        image = dynamic_cast<mitk::Image*>(nodes[i]->GetData());
        if (image.IsNotNull())
        {
          geometry = image->GetTimeSlicedGeometry();
          indexThatWeActuallyUsed = i;
          break;
        }
      }

      // Failing that, use the first geometry available.
      if (geometry.IsNull())
      {
        for (unsigned int i = 0; i < nodes.size(); i++)
        {
          mitk::BaseData::Pointer data = nodes[i]->GetData();
          if (data.IsNotNull())
          {
            geometry = data->GetTimeSlicedGeometry();
            indexThatWeActuallyUsed = i;
            break;
          }
        }
      }
    }
    // So, the caller has nominated a specific node, lets just use that one.
    else if (nodeIndex >= 0 && nodeIndex < (int)nodes.size())
    {
      mitk::BaseData::Pointer data = nodes[nodeIndex]->GetData();
      if (data.IsNotNull())
      {
        geometry = data->GetTimeSlicedGeometry();
        indexThatWeActuallyUsed = nodeIndex;
      }
    }
    // Essentially, the nodeIndex is garbage, so just pick the first one.
    else
    {
      mitk::BaseData::Pointer data = nodes[0]->GetData();
      if (data.IsNotNull())
      {
        geometry = data->GetTimeSlicedGeometry();
        indexThatWeActuallyUsed = 0;
      }
    }

    // In addition, (as MIDAS is an image based viewer), if the node is NOT a greyscale image,
    // we try and search the parents of the node to find a greyscale image and use that one in preference.
    // This assumes that derived datasets, such as point sets, surfaces, segmented volumes are correctly assigned to parents.
    if (indexThatWeActuallyUsed != -1)
    {
      if (!mitk::IsNodeAGreyScaleImage(nodes[indexThatWeActuallyUsed]))
      {
        mitk::DataNode::Pointer node = FindParentGreyScaleImage(dataStorage, nodes[indexThatWeActuallyUsed]);
        if (node.IsNotNull())
        {
          mitk::BaseData::Pointer data = nodes[0]->GetData();
          if (data.IsNotNull())
          {
            geometry = data->GetTimeSlicedGeometry();
          }
        }
      }
    }
    return geometry;
  }

} // end namespace


