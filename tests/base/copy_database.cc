#include <fiddle/base/samrai_utilities.h>

#include <tbox/Database.h>
#include <tbox/InputManager.h>
#include <tbox/SAMRAIManager.h>

#include <fstream>

#include <mpi.h>

int
main(int argc, char **argv)
{
  using namespace SAMRAI;
  MPI_Init(&argc, &argv);
  tbox::SAMRAIManager::startup();

  {
    tbox::Pointer<tbox::Database> input_db =
      new tbox::InputDatabase("input_db");
    tbox::InputManager::getManager()->parseInputFile(argv[1], input_db);
    auto copy = fdl::copy_database(input_db);

    // Modify the input
    input_db->putInteger("XX", 12345);
    input_db->putDatabase("Main");

    std::ofstream out("output");
    copy->printClassData(out);
  }

  tbox::SAMRAIManager::shutdown();
  MPI_Finalize();
}
