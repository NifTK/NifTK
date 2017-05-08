#include <iostream>
#include <fstream>
#include <typeinfo>

#include <mitkIOUtil.h>
#include <mitkSceneIO.h>
#include <niftkExtractLandmarkFromMITKSceneCLP.h>

int main(int argc, char **argv)
{
   PARSE_ARGS;
   int returnStatus = EXIT_FAILURE;

   try
   {
      if ( input.length() != 0 )
      {
         // Read the specified MITK scene
         mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();
         mitk::DataStorage::Pointer dataStorage = sceneIO->LoadScene(input);

         // Extract and save the point nodes
         mitk::DataStorage::SetOfObjects::ConstPointer nodes = dataStorage->GetAll();
         for (unsigned int i = 0; i < nodes->size(); i++)
         {
            mitk::PointSet::Pointer ps = dynamic_cast<mitk::PointSet*>(nodes->at(i)->GetData());
            if (ps.IsNull())
            {
               continue;
            }
            std::string nodeName = nodes->at(i)->GetName();
            std::cout << "Saving the point set named: " << nodeName << std::endl;

            mitk::IOUtil::Save(ps, output + "_" + nodeName + ".mps");
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
