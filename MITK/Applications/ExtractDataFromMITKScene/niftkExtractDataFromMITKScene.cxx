/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <iostream>
#include <fstream>
#include <typeinfo>
#include <mitkVector.h>

#include <mitkIOUtil.h>
#include <mitkStandaloneDataStorage.h>
#include <niftkExtractDataFromMITKSceneCLP.h>

int main(int argc, char **argv)
{
  PARSE_ARGS;

  int returnStatus = EXIT_FAILURE;

  try
  {
    if ( input.length() != 0 )
    {
      // Read the specified MITK scene
      mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
      mitk::IOUtil::Load(input, *static_cast<mitk::DataStorage*>(dataStorage.GetPointer()));

      // Extract and save the point nodes
      mitk::DataStorage::SetOfObjects::ConstPointer nodes = dataStorage->GetAll();
      for (unsigned int i = 0; i < nodes->size(); i++)
      {
         if(datatype.compare("pointset")==0)
         {
            mitk::PointSet::Pointer ps = dynamic_cast<mitk::PointSet *>(nodes->at(i)->GetData());
            if (ps.IsNull())
            {
              continue;
            }
            std::string nodeName = nodes->at(i)->GetName();
            std::cout << "Saving the point set named: " << nodeName << std::endl;

            mitk::IOUtil::Save(ps,  output + "/" + prefix + "_" + nodeName + ".mps");
         }
         else if(datatype.compare("image")==0)
         {
            mitk::Image::Pointer img = dynamic_cast<mitk::Image *>(nodes->at(i)->GetData());
            if (img.IsNull())
            {
               continue;
            }
            std::string nodeName = nodes->at(i)->GetName();
            std::cout << "Saving the image named: " << nodeName << std::endl;

            mitk::IOUtil::Save(img, output + "/" + prefix + "_" + nodeName + ".nii.gz");
         }
         else
         {
            MITK_ERROR << "Unsupported datatype: " << datatype;
            returnStatus = -3;
         }
      }
      returnStatus = EXIT_SUCCESS;
    }
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = -2;
  }
  return returnStatus;
}
