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

  bool IsNodeAGreyScaleImage(const mitk::DataNode::Pointer node)
  {
    bool result = false;

    if (node.IsNotNull())
    {
      mitk::Image::Pointer image = static_cast<mitk::Image*>(node->GetData());
      if (image.IsNotNull())
      {
        bool isBinary;
        node->GetBoolProperty("binary", isBinary);

        if (!isBinary)
        {
          result = true;
        }
      }
    }

    return result;
  }

  bool IsNodeABinaryImage(const mitk::DataNode::Pointer node)
  {
    bool result = false;

    if (node.IsNotNull())
    {
      mitk::Image::Pointer image = static_cast<mitk::Image*>(node->GetData());
      if (image.IsNotNull())
      {
        node->GetBoolProperty("binary", result);
      }
    }

    return result;
  }

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


  mitk::DataNode::Pointer FindNthGreyScaleImage(const std::vector<mitk::DataNode*> &nodes, int n )
  {
    return FindNthImage(nodes, n, false);
  }

  mitk::DataNode::Pointer FindNthBinaryImage(const std::vector<mitk::DataNode*> &nodes, int n )
  {
    return FindNthImage(nodes, n, true);
  }

  mitk::DataNode::Pointer FindFirstGreyScaleImage(const std::vector<mitk::DataNode*> &nodes )
  {
    return FindNthGreyScaleImage(nodes, 1);
  }

  mitk::DataNode::Pointer FindFirstBinaryImage(const std::vector<mitk::DataNode*> &nodes )
  {
    return FindNthBinaryImage(nodes, 1);
  }

}
