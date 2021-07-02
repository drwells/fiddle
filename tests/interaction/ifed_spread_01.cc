// ---------------------------------------------------------------------
//
// Copyright (c) 2019 - 2019 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#include <fiddle/base/exceptions.h>
#include <fiddle/base/samrai_utilities.h>

#include <fiddle/interaction/ifed_method.h>

#include <fiddle/mechanics/part.h>

#include <deal.II/base/function_parser.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/muParserCartGridFunction.h>

#include <CellVariable.h>
#include <SideVariable.h>

#include <string>
#include <vector>

#include "../tests.h"

using namespace dealii;
using namespace SAMRAI;

template <int dim, int spacedim = dim>
class IFEDMethod2 : public fdl::IFEDMethod<dim, spacedim>
{
public:
  IFEDMethod2(tbox::Pointer<tbox::Database>           test_db,
              tbox::Pointer<tbox::Database>           input_db,
              std::vector<fdl::Part<dim, spacedim>> &&input_parts)
    : fdl::IFEDMethod<dim, spacedim>("ifed_method",
                                     input_db,
                                     std::move(input_parts))
    , test_db(test_db)
  {}

  virtual void
  computeLagrangianForce(const double /*time*/) override
  {
    // everything else is at the current time so just roll with that
    const auto &                               part = this->parts[0];
    LinearAlgebra::distributed::Vector<double> current_force(
      part.get_partitioner());

    FunctionParser<spacedim> fp(extract_fp_string(
                                  test_db->getDatabase("f_exact")),
                                "PI=" + std::to_string(numbers::PI),
                                dim == 2 ? "X_0,X_1" : "X_0,X_1,X_2");
    VectorTools::interpolate(part.get_mapping(),
                             part.get_dof_handler(),
                             fp,
                             current_force);

    current_force.update_ghost_values();
    Assert(current_force.has_ghost_elements(),
           ExcMessage("Should have ghosts"));
    this->current_force_vectors.emplace_back(std::move(current_force));
    // a proper move ctor for LA::d::V is not yet merged into deal.II (but
    // should be in 10.0)
    this->current_force_vectors.back().update_ghost_values();
    Assert(this->current_force_vectors.back().has_ghost_elements(),
           ExcMessage("Should have ghosts"));
  }

  // just for testing - plot the force
  const LinearAlgebra::distributed::Vector<double> &
  get_force() const
  {
    return this->current_force_vectors.back();
  }

protected:
  tbox::Pointer<tbox::Database> test_db;
};

template <int dim, int spacedim = dim>
void
test(tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto       input_db = app_initializer->getInputDatabase();
  auto       test_db  = input_db->getDatabase("test");
  const auto mpi_comm = MPI_COMM_WORLD;

  // setup deal.II stuff:
  const double dx = input_db->getDouble("DX");
  const double ds = input_db->getDouble("MFAC") * dx;
  const double L  = input_db->getDouble("L");
  parallel::shared::Triangulation<dim, spacedim> tria(
    mpi_comm, {}, test_db->getBoolWithDefault("use_artificial_cells", false));
  // ensure that all points are actually inside the Eulerian domain
  GridGenerator::subdivided_hyper_cube(tria,
                                       std::ceil(L / ds),
                                       std::nexttoward(0.0, L),
                                       std::nexttoward(L, 0.0),
                                       L);
  tbox::plog << "Number of elements: " << tria.n_active_cells() << '\n';

  // fiddle:
  FESystem<dim> fe(FE_Q<dim>(test_db->getIntegerWithDefault("fe_degree", 1)),
                   dim);
  std::vector<fdl::Part<dim>> parts;
  parts.emplace_back(tria, fe);
  IFEDMethod2<dim, spacedim> ifed_method(test_db,
                                         input_db->getDatabase("IFEDMethod"),
                                         std::move(parts));

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_index         = std::get<5>(tuple);

  ifed_method.initializePatchHierarchy(
    patch_hierarchy, std::get<4>(tuple), -1, {}, {}, 0, 0.0, true);

  // Set up visualization plot file writers.
  tbox::Pointer<appu::VisItDataWriter<spacedim>> visit_data_writer =
    app_initializer->getVisItDataWriter();

  // Actual test:
  {
    const double data_time = 0.0;
    ifed_method.preprocessIntegrateData(0.0, 1.0, 0);
    ifed_method.computeLagrangianForce(data_time);
    // We can skip boundary accumulation since there are no physical boundaries
    ifed_method.spreadForce(f_index, nullptr, {}, data_time);

    auto *var_db = hier::VariableDatabase<spacedim>::getDatabase();
    tbox::Pointer<hier::VariableContext> ctx = var_db->getContext("context");

    tbox::Pointer<hier::Variable<spacedim>> f_exact_var;
    tbox::Pointer<hier::Variable<spacedim>> f_error_var;
    int                                     f_exact_index = -1;
    int                                     f_error_index = -1;
    const std::string                       f_data_type =
      input_db->getDatabase("test")->getStringWithDefault("f_data_type",
                                                          "CELL");
    const int n_components = get_n_f_components(input_db);
    if (f_data_type == "CELL")
      {
        f_exact_var =
          new pdat::CellVariable<spacedim, double>("f_exact", n_components);
        f_error_var =
          new pdat::CellVariable<spacedim, double>("f_error", n_components);
      }
    else if (f_data_type == "SIDE")
      {
        f_exact_var = new pdat::SideVariable<spacedim, double>("f_exact");
        f_error_var = new pdat::SideVariable<spacedim, double>("f_error");
      }
    f_exact_index =
      var_db->registerVariableAndContext(f_exact_var,
                                         ctx,
                                         hier::IntVector<spacedim>(1));
    f_error_index =
      var_db->registerVariableAndContext(f_error_var,
                                         ctx,
                                         hier::IntVector<spacedim>(1));
    for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
      {
        tbox::Pointer<hier::PatchLevel<spacedim>> level =
          patch_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(f_exact_index, 0.0);
        level->allocatePatchData(f_error_index, 0.0);
      }

    IBTK::muParserCartGridFunction f_fcn(
      "f",
      input_db->getDatabase("test")->getDatabase("f_exact"),
      patch_hierarchy->getGridGeometry());
    f_fcn.setDataOnPatchHierarchy(f_exact_index,
                                  f_exact_var,
                                  patch_hierarchy,
                                  0.0);

    // Compute error and interpolate output.
    auto hier_data_ops =
      fdl::extract_hierarchy_data_ops(f_exact_var, patch_hierarchy);
    hier_data_ops->subtract(f_error_index, f_index, f_exact_index);

    const double error = hier_data_ops->maxNorm(f_error_index);
    tbox::pout << "max norm error = " << error << '\n';
    if (IBTK::IBTK_MPI::getRank() == 0)
      {
        std::ofstream output("output");
        output << "Number of elements: " << tria.n_active_cells() << '\n';
        output << "max norm error = " << error << '\n';
      }

    // plot data:
    {
      if (f_data_type == "CELL")
        {
          for (int d = 0; d < n_components; ++d)
            {
              visit_data_writer->registerPlotQuantity(f_exact_var->getName() +
                                                        std::to_string(d),
                                                      "SCALAR",
                                                      f_exact_index,
                                                      d);
              visit_data_writer->registerPlotQuantity(f_error_var->getName() +
                                                        std::to_string(d),
                                                      "SCALAR",
                                                      f_error_index,
                                                      d);
            }
        }
      else if (f_data_type == "SIDE")
        {
          tbox::Pointer<pdat::CellVariable<spacedim, double>> f_exact_cc_var =
            new pdat::CellVariable<spacedim, double>("f_exact_cc",
                                                     n_components);
          tbox::Pointer<pdat::CellVariable<spacedim, double>> f_error_cc_var =
            new pdat::CellVariable<spacedim, double>("f_error_cc",
                                                     n_components);

          auto f_exact_cc_index =
            var_db->registerVariableAndContext(f_exact_cc_var,
                                               ctx,
                                               hier::IntVector<spacedim>(1));
          auto f_error_cc_index =
            var_db->registerVariableAndContext(f_error_cc_var,
                                               ctx,
                                               hier::IntVector<spacedim>(1));

          for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
            {
              tbox::Pointer<hier::PatchLevel<spacedim>> level =
                patch_hierarchy->getPatchLevel(ln);
              level->allocatePatchData(f_exact_cc_index, 0.0);
              level->allocatePatchData(f_error_cc_index, 0.0);
            }

          tbox::Pointer<pdat::SideVariable<spacedim, double>> f_var;
          // This is why you shouldn't invent your own type system...
          tbox::Pointer<hier::Variable<spacedim>> other_f_var;
          var_db->mapIndexToVariable(f_index, other_f_var);
          f_var = other_f_var;

          IBTK::HierarchyMathOps hier_math_ops("hier_math_ops",
                                               patch_hierarchy);

          tbox::Pointer<pdat::SideVariable<spacedim, double>> f_exact_var_2 =
            f_exact_var;
          tbox::Pointer<pdat::SideVariable<spacedim, double>> f_error_var_2 =
            f_error_var;
          hier_math_ops.interp(f_exact_cc_index,
                               f_exact_cc_var,
                               f_exact_index,
                               f_exact_var_2,
                               NULL,
                               0.0,
                               true);
          hier_math_ops.interp(f_error_cc_index,
                               f_error_cc_var,
                               f_error_index,
                               f_error_var_2,
                               NULL,
                               0.0,
                               true);

          for (int d = 0; d < n_components; ++d)
            {
              visit_data_writer->registerPlotQuantity(
                f_exact_cc_var->getName() + std::to_string(d),
                "SCALAR",
                f_exact_cc_index,
                d);
              visit_data_writer->registerPlotQuantity(
                f_error_cc_var->getName() + std::to_string(d),
                "SCALAR",
                f_error_cc_index,
                d);
            }
        }
    }

    {
      const auto & part = ifed_method.get_part(0);
      DataOut<dim> data_out;
      data_out.attach_dof_handler(part.get_dof_handler());
      data_out.add_data_vector(ifed_method.get_force(), "F");
      Assert(ifed_method.get_force().has_ghost_elements(),
             ExcMessage("Should have ghosts"));

      MappingFEField<dim, spacedim, LinearAlgebra::distributed::Vector<double>>
        position_mapping(part.get_dof_handler(), part.get_position());
      Assert(part.get_position().has_ghost_elements(),
             ExcMessage("Should have ghosts"));
      data_out.build_patches(position_mapping);
      data_out.write_vtu_with_pvtu_record("./", "solution", 0, mpi_comm, 8);
    }

    visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
  }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit                      ibtk_init(argc, argv, MPI_COMM_WORLD);
  tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "ifed_tag.log");

  test<NDIM>(app_initializer);
}
