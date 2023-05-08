#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/ifed_method_base.h>
#include <fiddle/interaction/interaction_utilities.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <ibamr/ibamr_utilities.h>

#include <ibtk/IBTK_MPI.h>

#include <tbox/RestartManager.h>

#include <limits>

namespace
{
  using namespace SAMRAI;
  static tbox::Timer *t_apply_gradient_detector;
  static tbox::Timer *t_max_point_displacement;
  static tbox::Timer *t_begin_data_redistribution;
  static tbox::Timer *t_end_data_redistribution;
  static tbox::Timer *t_preprocess_integrate_data;
  static tbox::Timer *t_postprocess_integrate_data;
} // namespace

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  //
  // Initialization
  //

  template <int dim, int spacedim>
  IFEDMethodBase<dim, spacedim>::IFEDMethodBase(
    const std::string                     &object_name,
    std::vector<Part<dim - 1, spacedim>> &&input_surface_parts,
    std::vector<Part<dim, spacedim>>     &&input_parts,
    const bool                             register_for_restart)
    : object_name(object_name)
    , register_for_restart(register_for_restart)
    , started_time_integration(false)
    , current_time(std::numeric_limits<double>::signaling_NaN())
    , half_time(std::numeric_limits<double>::signaling_NaN())
    , new_time(std::numeric_limits<double>::signaling_NaN())
    , parts(std::move(input_parts))
    , surface_parts(std::move(input_surface_parts))
    , part_vectors(this->parts)
    , surface_part_vectors(this->surface_parts)
  {
    // IBAMR does not support using threads so unconditionally disable them
    // here.
    MultithreadInfo::set_thread_limit(1);

    auto set_timer = [&](const char *name)
    { return tbox::TimerManager::getManager()->getTimer(name); };
    t_apply_gradient_detector =
      set_timer("fdl::IFEDMethodBase::applyGradientDetector()");
    t_max_point_displacement =
      set_timer("fdl::IFEDMethodBase::getMaxPointDisplacement()");
    t_begin_data_redistribution =
      set_timer("fdl::IFEDMethodBase::beginDataRedistribution()");
    t_end_data_redistribution =
      set_timer("fdl::IFEDMethodBase::endDataRedistribution()");
    t_preprocess_integrate_data =
      set_timer("fdl::IFEDMethodBase::preprocessIntegrateData()");
    t_postprocess_integrate_data =
      set_timer("fdl::IFEDMethodBase::postprocessIntegrateData()");

    auto init_regrid_positions = [](auto &vectors, const auto &collection)
    {
      for (const auto &part : collection)
        vectors.push_back(part.get_position());
    };

    if (register_for_restart)
      {
        auto *restart_manager = tbox::RestartManager::getManager();
        restart_manager->registerRestartItem(object_name, this);
        if (restart_manager->isFromRestart())
          {
            auto restart_db = restart_manager->getRootDatabase();
            if (restart_db->isDatabase(object_name))
              {
                auto db      = restart_db->getDatabase(object_name);
                auto do_load = [&](auto &collection, const std::string &prefix)
                {
                  for (unsigned int i = 0; i < collection.size(); ++i)
                    {
                      const std::string key = prefix + std::to_string(i);
                      AssertThrow(db->keyExists(key),
                                  ExcMessage("Couldn't find key " + key +
                                             " in the restart database"));
                      const std::string  serialization = load_binary(key, db);
                      std::istringstream in_str(serialization);
                      boost::archive::binary_iarchive iarchive(in_str);
                      collection[i].load(iarchive, 0);
                    }
                };
                do_load(this->parts, "part_");
                do_load(this->surface_parts, "surface_part_");
              }
            else
              {
                AssertThrow(false,
                            ExcMessage(
                              "The restart database does not contain key " +
                              object_name));
              }
          }
      }

    init_regrid_positions(positions_at_last_regrid, parts);
    init_regrid_positions(surface_positions_at_last_regrid, surface_parts);
  }

  template <int dim, int spacedim>
  IFEDMethodBase<dim, spacedim>::~IFEDMethodBase()
  {
    if (register_for_restart)
      {
        tbox::RestartManager::getManager()->unregisterRestartItem(object_name);
      }
  }


  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::initializePatchHierarchy(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> hierarchy,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/,
    int /*u_data_index*/,
    const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
      & /*u_synch_scheds*/,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
      & /*u_ghost_fill_scheds*/,
    int /*integrator_step*/,
    double /*init_data_time*/,
    bool /*initial_time*/)
  {
    patch_hierarchy = hierarchy;

    eulerian_data_cache = std::make_shared<IBTK::SAMRAIDataCache>();
    eulerian_data_cache->setPatchHierarchy(hierarchy);
    eulerian_data_cache->resetLevels(0, hierarchy->getFinestLevelNumber());
  }

  //
  // Data redistribution
  //

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::beginDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    IBAMR_TIMER_START(t_begin_data_redistribution);
    IBAMR_TIMER_STOP(t_begin_data_redistribution);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::endDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    IBAMR_TIMER_START(t_end_data_redistribution);
    auto do_reset = [](auto &positions_regrid, const auto &collection)
    {
      positions_regrid.clear();
      for (unsigned int i = 0; i < collection.size(); ++i)
        positions_regrid.push_back(collection[i].get_position());
    };
    do_reset(this->positions_at_last_regrid, this->parts);
    do_reset(this->surface_positions_at_last_regrid, this->surface_parts);
    IBAMR_TIMER_STOP(t_end_data_redistribution);
  }

  //
  // FSI
  //

  template <int dim, int spacedim>
  double
  IFEDMethodBase<dim, spacedim>::getMaxPointDisplacement() const
  {
    IBAMR_TIMER_START(t_max_point_displacement);
    double max_displacement = 0;

    auto max_op = [&](const auto &collection, const auto &regrid_positions)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          AssertDimension(collection.size(), regrid_positions.size());
          const auto &ref_position = regrid_positions[i];
          const auto &position     = collection[i].get_position();
          const auto  local_size   = position.locally_owned_size();
          for (unsigned int j = 0; j < local_size; ++j)
            max_displacement = std::max(max_displacement,
                                        std::abs(ref_position.local_element(j) -
                                                 position.local_element(j)));
        }
    };
    max_op(this->parts, positions_at_last_regrid);
    max_op(this->surface_parts, surface_positions_at_last_regrid);
    max_displacement =
      Utilities::MPI::max(max_displacement, IBTK::IBTK_MPI::getCommunicator());

    return max_displacement /
           IBTK::get_min_patch_dx(
             dynamic_cast<const hier::PatchLevel<spacedim> &>(
               *patch_hierarchy->getPatchLevel(
                 patch_hierarchy->getFinestLevelNumber())));
    IBAMR_TIMER_STOP(t_max_point_displacement);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::applyGradientDetector(
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> hierarchy,
    int                                               level_number,
    double /*error_data_time*/,
    int tag_index,
    bool /*initial_time*/,
    bool /*uses_richardson_extrapolation_too*/)
  {
    IBAMR_TIMER_START(t_apply_gradient_detector);
    // TODO: we should find a way to save the bboxes so they do not need to be
    // computed for each level that needs tagging - conceivably this could
    // happen in beginDataRedistribution() and the array can be cleared in
    // endDataRedistribution()
    auto do_tag = [&](const auto &collection)
    {
      for (const auto &part : collection)
        {
          constexpr int structdim =
            std::remove_reference_t<decltype(collection[0])>::dimension;
          MappingFEField<structdim,
                         spacedim,
                         LinearAlgebra::distributed::Vector<double>>
                     mapping(part.get_dof_handler(), part.get_position());
          const auto local_bboxes =
            compute_cell_bboxes<structdim, spacedim, float>(
              part.get_dof_handler(), mapping);
          // Like most other things this only works with p::s::T now
          const auto &tria = dynamic_cast<
            const parallel::shared::Triangulation<structdim, spacedim> &>(
            part.get_triangulation());
          const auto global_bboxes =
            collect_all_active_cell_bboxes(tria, local_bboxes);
          tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
            hierarchy->getPatchLevel(level_number);
          Assert(patch_level, ExcNotImplemented());
          tag_cells(global_bboxes, tag_index, patch_level);
        }
    };

    do_tag(parts);
    do_tag(surface_parts);
    IBAMR_TIMER_STOP(t_apply_gradient_detector);
  }

  //
  // Time stepping
  //

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::preprocessIntegrateData(double current_time,
                                                         double new_time,
                                                         int /*num_cycles*/)
  {
    IBAMR_TIMER_START(t_preprocess_integrate_data);
    started_time_integration = true;
    part_vectors.begin_time_step(current_time, new_time);
    surface_part_vectors.begin_time_step(current_time, new_time);
    current_time = current_time;
    new_time     = new_time;
    half_time    = current_time + 0.5 * (new_time - current_time);
    IBAMR_TIMER_STOP(t_preprocess_integrate_data);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::postprocessIntegrateData(
    double /*current_time*/,
    double /*new_time*/,
    int /*num_cycles*/)
  {
    IBAMR_TIMER_START(t_postprocess_integrate_data);
    current_time = std::numeric_limits<double>::quiet_NaN();
    new_time     = std::numeric_limits<double>::quiet_NaN();
    half_time    = std::numeric_limits<double>::quiet_NaN();

    // update positions and velocities:
    unsigned int channel = 0;
    auto do_set = [&](auto &collection, auto &positions, auto &velocities)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          part.set_position(std::move(positions[i]));
          part.get_position().update_ghost_values_start(channel++);
          part.set_velocity(std::move(velocities[i]));
          part.get_velocity().update_ghost_values_start(channel++);
        }
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          part.get_position().update_ghost_values_finish();
          part.get_velocity().update_ghost_values_finish();
        }
    };
    auto new_positions          = part_vectors.get_all_new_positions();
    auto new_velocities         = part_vectors.get_all_new_velocities();
    auto surface_new_positions  = surface_part_vectors.get_all_new_positions();
    auto surface_new_velocities = surface_part_vectors.get_all_new_velocities();
    do_set(parts, new_positions, new_velocities);
    do_set(surface_parts, surface_new_positions, surface_new_velocities);

    part_vectors.end_time_step();
    surface_part_vectors.end_time_step();
    IBAMR_TIMER_STOP(t_postprocess_integrate_data);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::forwardEulerStep(double current_time,
                                                  double new_time)
  {
    const double dt = new_time - current_time;
    Assert(this->current_time == current_time, ExcFDLNotImplemented());
    auto do_step = [&](auto &collection, auto &vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          // Set the position at the end time:
          LinearAlgebra::distributed::Vector<double> new_position(
            part.get_position());
          new_position.set_ghost_state(false);
          new_position.add(dt, part.get_velocity());
          vectors.set_position(i, new_time, std::move(new_position));

          // Set the position at the half time:
          LinearAlgebra::distributed::Vector<double> half_position(
            part.get_partitioner());
          half_position.set_ghost_state(false);
          half_position.add(0.5,
                            vectors.get_position(i, current_time),
                            0.5,
                            vectors.get_position(i, new_time));
          vectors.set_position(i, half_time, std::move(half_position));
        }
    };
    do_step(parts, part_vectors);
    do_step(surface_parts, surface_part_vectors);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::backwardEulerStep(double current_time,
                                                   double new_time)
  {
    (void)current_time;
    (void)new_time;
    Assert(false, ExcFDLNotImplemented());
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::midpointStep(double current_time,
                                              double new_time)
  {
    const double dt = new_time - current_time;
    Assert(this->current_time == current_time, ExcFDLNotImplemented());
    auto do_step = [&](auto &collection, auto &vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          // Set the position at the end time:
          LinearAlgebra::distributed::Vector<double> new_position(
            part.get_position());
          new_position.set_ghost_state(false);
          new_position.add(dt, vectors.get_velocity(i, half_time));
          vectors.set_position(i, new_time, std::move(new_position));

          // Set the position at the half time:
          LinearAlgebra::distributed::Vector<double> half_position(
            part.get_partitioner());
          half_position.set_ghost_state(false);
          half_position.add(0.5,
                            vectors.get_position(i, current_time),
                            0.5,
                            vectors.get_position(i, new_time));
          vectors.set_position(i, half_time, std::move(half_position));
        }
    };
    do_step(parts, part_vectors);
    do_step(surface_parts, surface_part_vectors);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::trapezoidalStep(double current_time,
                                                 double new_time)
  {
    (void)current_time;
    (void)new_time;
    Assert(false, ExcFDLNotImplemented());
  }

  //
  // Book-keeping
  //

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::putToDatabase(tbox::Pointer<tbox::Database> db)
  {
    auto do_put = [&](auto &collection, const std::string &prefix)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          std::ostringstream              out_str;
          boost::archive::binary_oarchive oarchive(out_str);
          collection[i].save(oarchive, 0);
          // TODO - with C++20 we can use view() instead of str() and skip
          // this copy
          const std::string out = out_str.str();
          save_binary(prefix + std::to_string(i),
                      out.c_str(),
                      out.c_str() + out.size(),
                      db);
        }
    };
    do_put(parts, "part_");
    do_put(surface_parts, "surface_part_");
  }

  template class IFEDMethodBase<NDIM, NDIM>;
} // namespace fdl
