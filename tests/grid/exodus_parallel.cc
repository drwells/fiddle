#include <fiddle/base/exceptions.h>

#include <fiddle/grid/data_in.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "../tests.h"

int
main(int argc, char **argv)
{
  using namespace dealii;

  const MPI_Comm                   comm = MPI_COMM_WORLD;
  Utilities::MPI::MPI_InitFinalize mpi_environment(argc, argv);

  const auto partitioner =
    parallel::shared::Triangulation<2>::Settings::partition_zorder;
  parallel::shared::Triangulation<2> tria(comm, {}, true, partitioner);

  int         degree = 1;
  std::string exodus_prefix;
  if (argc > 1)
    {
      const std::string input_file(argv[1]);

      for (const char e : {'p', 'q'})
        for (const int i : {1, 2})
          {
            const std::string in_str =
              "." + std::string(std::size_t(1), e) + std::to_string(i) + ".";
            if (std::search(input_file.begin(),
                            input_file.end(),
                            in_str.begin(),
                            in_str.end()) != input_file.end())
              {
                degree        = i;
                exodus_prefix = in_str.substr(1, 2);
              }
          }

      AssertThrow(exodus_prefix.size() > 0, fdl::ExcFDLNotImplemented());
    }
  else
    {
      exodus_prefix = "p1";
    }

  const std::string test_file = SOURCE_DIR + exodus_prefix + ".ex2";
  std::cout << "test file = " << test_file << '\n';
  std::cout << "degree = " << degree << '\n';

  GridIn<2> grid_in(tria);
  auto      result = grid_in.read_exodusii(test_file);
  (void)result;

  Vector<double> libmesh_cell_data(tria.n_active_cells());
  fdl::read_elemental_data(test_file, tria, 1, "M", libmesh_cell_data);

  std::unique_ptr<FiniteElement<2>> scalar_fe;
  std::unique_ptr<FiniteElement<2>> constant_fe;
  if (tria.all_reference_cells_are_hyper_cube())
    {
      scalar_fe.reset(new FE_Q<2>(degree));
      constant_fe.reset(new FE_DGQ<2>(0));
    }
  else
    {
      scalar_fe.reset(new FE_SimplexP<2>(degree));
      constant_fe.reset(new FE_SimplexDGP<2>(0));
    }

  std::ostringstream local_out;
  local_out << "scalar DoFHandler\n\n";
  DataOutBase::VtkFlags flags;
  const unsigned int    n_subdivisions = degree;
  flags.print_date_and_time            = false;

  // Test a scalar DoFHandler
  {
    DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(*scalar_fe);

    LinearAlgebra::distributed::Vector<double> position_x(
      dof_handler.locally_owned_dofs(), comm);
    std::vector<std::string> var_names{std::string("X_0")};
    fdl::read_dof_data(test_file, dof_handler, 1, var_names, position_x);

    DataOut<2> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(libmesh_cell_data, "M");
    data_out.add_data_vector(position_x, "X_0");

    data_out.build_patches(n_subdivisions);
    data_out.write_vtu_with_pvtu_record("./", "grid-2", 1, comm, 5);
    data_out.write_vtk(local_out);
  }

  // Test a vector DoFHandler
  local_out << "\n\nvector DoFHandler\n\n";
  {
    FESystem<2>   fe(*scalar_fe, 2);
    DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    LinearAlgebra::distributed::Vector<double> position(
      dof_handler.locally_owned_dofs(), comm);
    std::vector<std::string> var_names{std::string("X_0"), std::string("X_1")};
    fdl::read_dof_data(test_file, dof_handler, 1, var_names, position);

    DataOut<2> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(libmesh_cell_data, "M");
    data_out.add_data_vector(position, var_names);

    data_out.build_patches(n_subdivisions);
    data_out.write_vtu_with_pvtu_record("./", "grid-3", 1, comm, 5);
    data_out.write_vtk(local_out);
  }

  // mixed DoFHandler doesn't work yet
#if 0
  local_out << "\n\nmixed DoFHandler\n\n";
  {
    FESystem<2> fe(*scalar_fe, 2, *constant_fe, 2);
    DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::block_wise(dof_handler);
    std::vector<IndexSet> dofs_per_block =
      DoFTools::locally_owned_dofs_per_component(dof_handler);
    dofs_per_block[0].add_indices(dofs_per_block[1]);
    dofs_per_block[2].add_indices(dofs_per_block[3]);
    dofs_per_block.erase(dofs_per_block.begin() + 1);
    dofs_per_block.erase(dofs_per_block.begin() + 2);

    LinearAlgebra::distributed::BlockVector<double> position(dofs_per_block, comm);
    std::vector<std::string> var_names
      {"X_0", "X_1", "M", "FF_11"};
    fdl::read_dof_data(test_file, dof_handler, 1, var_names, position);

    DataOut<2> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(position, var_names);

    data_out.build_patches(n_subdivisions);
    data_out.write_vtu_with_pvtu_record(
      "./", "grid-4", 1, comm, 5);
  }
#endif

  std::ofstream output;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    output.open("output");

  print_strings_on_0(local_out.str(), comm, output);
}
