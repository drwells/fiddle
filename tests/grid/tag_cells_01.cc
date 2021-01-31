#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/interaction/interaction.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

#include <deal.II/base/bounding_box_data_out.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/rtree.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>

#include "../tests.h"

// Test cell tagging

using namespace SAMRAI;
using namespace dealii;

template <int spacedim>
class TestTag : public mesh::StandardTagAndInitStrategy<spacedim>
{
public:
    TestTag(const std::vector<BoundingBox<spacedim, float>> &bboxes)
        : cell_bboxes(bboxes)
    {
    }

    virtual
    void initializeLevelData(const tbox::Pointer<hier::BasePatchHierarchy<spacedim> > /*hierarchy*/,
                         const int /*level_number*/,
                         const double /*init_data_time*/,
                         const bool /*can_be_refined*/,
                         const bool /*initial_time*/,
                         const tbox::Pointer<hier::BasePatchLevel<spacedim> > /*old_level*/ = nullptr,
                         const bool /*allocate_data*/ = true) override
    {
    }

    virtual
    void resetHierarchyConfiguration(const tbox::Pointer<hier::BasePatchHierarchy<spacedim> > /*hierarchy*/,
                                     const int /*coarsest_level*/,
                                     const int /*finest_level*/) override
    {
    }

    virtual
    void applyGradientDetector(const tbox::Pointer<hier::BasePatchHierarchy<spacedim> > hierarchy,
                               const int level_number,
                               const double /*error_data_time*/,
                               const int tag_index,
                               const bool /*initial_time*/,
                               const bool /*uses_richardson_extrapolation_too*/) override
    {
        tbox::Pointer<hier::PatchLevel<spacedim>> patch_level = hierarchy->getPatchLevel(level_number);
        fdl::tag_cells(cell_bboxes, tag_index, patch_level);
    }

protected:
        std::vector<BoundingBox<spacedim, float>> cell_bboxes;
};

template <int dim, int spacedim = dim>
void test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
    std::ofstream output("output");

    // setup deal.II stuff:
    const auto                       mpi_comm = MPI_COMM_WORLD;
    const auto rank = Utilities::MPI::this_mpi_process(mpi_comm);
    parallel::shared::Triangulation<dim, spacedim> native_tria
        (mpi_comm,
         {},
         false,
         parallel::shared::Triangulation<dim, spacedim>::Settings::partition_zorder);
    GridGenerator::concentric_hyper_shells(native_tria,
                                           Point<spacedim>(),
                                           0.125,
                                           0.25,
                                           2,
                                           0.0);
    native_tria.refine_global(4);
    if (rank == 0)
    {
        GridOut go;
        std::ofstream grid_out("grid.vtk");
        go.write_vtk(native_tria, grid_out);
    }

    std::vector<BoundingBox<spacedim, float>> cell_bboxes;
    for (const auto &cell : native_tria.active_cell_iterators())
    {
        const BoundingBox<spacedim> box = cell->bounding_box();
        BoundingBox<spacedim, float> fbox;
        fbox.get_boundary_points().first = box.get_boundary_points().first;
        fbox.get_boundary_points().second = box.get_boundary_points().second;
        cell_bboxes.push_back(fbox);
    }

    // test:
    TestTag<spacedim> test_tag(cell_bboxes);

    // Set up basic SAMRAI stuff:
    tbox::Pointer<geom::CartesianGridGeometry<NDIM>> grid_geometry =
      new geom::CartesianGridGeometry<NDIM>(
        "CartesianGeometry",
        app_initializer->getComponentDatabase("CartesianGeometry"));
    tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy =
      new hier::PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
    tbox::Pointer<mesh::StandardTagAndInitialize<NDIM>> error_detector =
      new mesh::StandardTagAndInitialize<NDIM>(
        "StandardTagAndInitialize",
        &test_tag,
        app_initializer->getComponentDatabase("StandardTagAndInitialize"));

    tbox::Pointer<mesh::BergerRigoutsos<NDIM>> box_generator =
      new mesh::BergerRigoutsos<NDIM>();
    tbox::Pointer<mesh::LoadBalancer<NDIM>> load_balancer =
      new mesh::LoadBalancer<NDIM>(
        "LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
    tbox::Pointer<mesh::GriddingAlgorithm<NDIM>> gridding_algorithm =
      new mesh::GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase(
                                          "GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

    // set up the SAMRAI grid:
    gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
    int       level_number = 0;
    while (gridding_algorithm->levelCanBeRefined(level_number))
      {
        gridding_algorithm->makeFinerLevel(patch_hierarchy,
                                           0.0,
                                           0.0,
                                           1);
        ++level_number;
      }

    // Set up a variable so that we can actually output the grid:
    auto *var_db = hier::VariableDatabase<NDIM>::getDatabase();
    tbox::Pointer<hier::VariableContext> ctx = var_db->getContext("context");
    tbox::Pointer<pdat::CellVariable<spacedim, double>> u_cc_var =
      new pdat::CellVariable<spacedim, double>("u_cc");
    const int u_cc_idx =
      var_db->registerVariableAndContext(u_cc_var,
                                         ctx,
                                         hier::IntVector<spacedim>(1));

    const int finest_level = patch_hierarchy->getFinestLevelNumber();
    for (int ln = 0; ln <= finest_level; ++ln)
      {
        tbox::Pointer<hier::PatchLevel<spacedim>> level =
          patch_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(u_cc_idx, 0.0);

        // obviously this won't generalize well
        auto patches = fdl::extract_patches(level);
        for (auto &patch : patches)
        {
            tbox::Pointer<pdat::CellData<spacedim, double>> data
                = patch->getPatchData(u_cc_idx);
            Assert(data, ExcMessage("pointer should not be null"));
            data->fillAll(0.0);
        }

        fdl::tag_cells(cell_bboxes, u_cc_idx, level);
      }

    // setup visualization:
    auto visit_data_writer = app_initializer->getVisItDataWriter();
    TBOX_ASSERT(visit_data_writer);
    visit_data_writer->registerPlotQuantity(u_cc_var->getName(),
                                            "SCALAR",
                                            u_cc_idx);
    visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);

    // save test output:
    print_partitioning_on_0(patch_hierarchy, 0, finest_level, output);
}

int main(int argc, char **argv)
{
    IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer
        = new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

    test<2>(app_initializer);
}
