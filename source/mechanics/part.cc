#include <fiddle/mechanics/part.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/reference_cell.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <boost/serialization/array_wrapper.hpp>

namespace fdl
{
  namespace internal
  {
    // matrix_free doesn't work with codim != 0 so we need a helper function
    template <int dim>
    void
    reinit_matrix_free(const Mapping<dim>              &mapping,
                       const DoFHandler<dim>           &dof_handler,
                       const AffineConstraints<double> &constraints,
                       const Quadrature<dim>           &quadrature,
                       MatrixFree<dim, double>         &matrix_free)
    {
      matrix_free.reinit(mapping, dof_handler, constraints, quadrature);
    }

    template <int dim>
    void
    reinit_matrix_free(const Mapping<dim - 1, dim> &,
                       const DoFHandler<dim - 1, dim> &,
                       const AffineConstraints<double> &,
                       const Quadrature<dim - 1> &,
                       MatrixFree<dim - 1, double> &)
    {
      // We shouldn't get here
      AssertThrow(false, ExcFDLInternalError());
    }
  } // namespace internal

  template <int dim, int spacedim>
  Part<dim, spacedim>::Part(
    std::shared_ptr<DoFHandler<dim, spacedim>> dh,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                              force_contributions,
    const Function<spacedim> &initial_position,
    const Function<spacedim> &initial_velocity)
    : Part(dh,
           std::move(force_contributions),
           {},
           initial_position,
           initial_velocity)
  {}

  template <int dim, int spacedim>
  Part<dim, spacedim>::Part(
    std::shared_ptr<DoFHandler<dim, spacedim>> dh,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
      force_contributions,
    std::vector<std::unique_ptr<ActiveStrain<dim, spacedim>>> active_strains,
    const Function<spacedim>                                 &initial_position,
    const Function<spacedim>                                 &initial_velocity)
    : tria(&dh->get_triangulation())
    , fe(dh->get_fe().clone())
    , dof_handler(dh)
    , force_contributions(std::move(force_contributions))
    , active_strains(std::move(active_strains))
  {
    for (const auto &f : this->force_contributions)
      AssertThrow(f != nullptr,
                  ExcMessage("The force contributions must not be nullptr"));

    if (this->active_strains.size() > 0)
      {
        // Verify that we don't have duplicate active strains:
        std::map<types::material_id, unsigned int> counts;
        for (const auto &as : this->active_strains)
          {
            Assert(as != nullptr,
                   ExcMessage("The active strains must not be nullptr"));
            for (const types::material_id mid : as->get_material_ids())
              counts[mid] += 1;
          }

        for (const auto &pair : counts)
          AssertThrow(
            pair.second == 1,
            ExcMessage(
              "More than one active strain is defined for material id " +
              std::to_string(pair.first) +
              ": this is not supported. See the documentation of Part for "
              "more information."));

        // Don't permit body forces to depend on FF:
        for (const auto &f : this->force_contributions)
          if (!f->is_stress())
            AssertThrow(
              !(resolve_flag_dependencies(f->get_mechanics_update_flags()) &
                update_FF),
              ExcMessage("Using quantities dependent on the deformation "
                         "gradient in a body force is not supported with "
                         "active strains."));
      }

    const auto &reference_cells = this->tria->get_reference_cells();
    AssertThrow(reference_cells.size() == 1, ExcFDLNotImplemented());
    mapping = reference_cells.front()
                .template get_default_linear_mapping<dim, spacedim>()
                .clone();
    quadrature =
      reference_cells.front().template get_gauss_type_quadrature<dim>(
        fe->tensor_degree() + 1);

    AssertThrow(fe->n_components() == spacedim,
                ExcMessage("The finite element should have spacedim components "
                           "since it will represent the position, velocity and "
                           "force of the part."));
    // Set up DoFs and finite element fields:
    dof_handler->distribute_dofs(*fe);
    constraints.close();

    // A MatrixFree object sets up the partitioning on its own - use that to
    // avoid issues with p::s::T where there may not be artificial cells
    //
    // TODO - understand this issue well enough to file a bug report
    matrix_free = std::make_shared<MatrixFree<dim, double>>();
    if (dim == spacedim)
      {
        // matrix-free is only implemented in codim 0
        internal::reinit_matrix_free(
          *mapping, *dof_handler, constraints, quadrature, *matrix_free);
        partitioner = matrix_free->get_vector_partitioner();
      }
    else
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                                locally_relevant_dofs);
        partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          dof_handler->locally_owned_dofs(),
          locally_relevant_dofs,
          tria->get_communicator());
      }

    position.reinit(partitioner);
    velocity.reinit(partitioner);

    // Set up matrix free components:
    if (dim == spacedim)
      {
        if (reference_cells.front() == ReferenceCells::get_hypercube<dim>())
          {
            using namespace MatrixFreeOperators;
            switch (fe->tensor_degree())
              {
                case 1:
                  mass_operator.reset(new MassOperator<dim, 1, 1 + 1, dim>());
                  break;
                case 2:
                  mass_operator.reset(new MassOperator<dim, 2, 2 + 1, dim>());
                  break;
                case 3:
                  mass_operator.reset(new MassOperator<dim, 3, 3 + 1, dim>());
                  break;
                case 4:
                  mass_operator.reset(new MassOperator<dim, 4, 4 + 1, dim>());
                  break;
                case 5:
                  mass_operator.reset(new MassOperator<dim, 5, 5 + 1, dim>());
                  break;
                default:
                  AssertThrow(false, ExcFDLNotImplemented());
              }
          }
        else
          {
            using namespace MatrixFreeOperators;
            switch (fe->tensor_degree())
              {
                case 1:
                  mass_operator.reset(new MassOperator<dim, -1, 0, dim>());
                  break;
                case 2:
                  mass_operator.reset(new MassOperator<dim, -1, 0, dim>());
                  break;
                case 3:
                  mass_operator.reset(new MassOperator<dim, -1, 0, dim>());
                  break;
                default:
                  AssertThrow(false, ExcFDLNotImplemented());
              }
          }
        mass_operator->initialize(matrix_free);
        mass_operator->compute_diagonal();
        mass_preconditioner.initialize(*mass_operator, 1.0);
      }

    // finally, FE fields:
    VectorTools::interpolate(*dof_handler, initial_position, position);
    // The initial velocity is probably zero:
    if (dynamic_cast<const Functions::ZeroFunction<dim> *>(&initial_velocity))
      velocity = 0.0;
    else
      VectorTools::interpolate(*dof_handler, initial_velocity, velocity);

    position.update_ghost_values();
    velocity.update_ghost_values();
  }

  namespace
  {
    template <int dim, int spacedim>
    std::shared_ptr<DoFHandler<dim, spacedim>>
    setup_dof_handler(const Triangulation<dim, spacedim> &tria,
                      const FiniteElement<dim, spacedim> &fe)
    {
      auto dof_handler = std::make_shared<DoFHandler<dim, spacedim>>(tria);
      dof_handler->distribute_dofs(fe);
      return dof_handler;
    }
  } // namespace

  template <int dim, int spacedim>
  Part<dim, spacedim>::Part(
    const Triangulation<dim, spacedim> &tria,
    const FiniteElement<dim, spacedim> &fe,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                              force_contributions,
    const Function<spacedim> &initial_position,
    const Function<spacedim> &initial_velocity)
    : Part(setup_dof_handler(tria, fe),
           std::move(force_contributions),
           {},
           initial_position,
           initial_velocity)
  {}

  template <int dim, int spacedim>
  Part<dim, spacedim>::Part(
    const Triangulation<dim, spacedim> &tria,
    const FiniteElement<dim, spacedim> &fe,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
      force_contributions,
    std::vector<std::unique_ptr<ActiveStrain<dim, spacedim>>> active_strains,
    const Function<spacedim>                                 &initial_position,
    const Function<spacedim>                                 &initial_velocity)
    : Part(setup_dof_handler(tria, fe),
           std::move(force_contributions),
           std::move(active_strains),
           initial_position,
           initial_velocity)
  {}

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::save(boost::archive::binary_oarchive &archive,
                            const unsigned int               version) const
  {
    const_cast<Part<dim, spacedim> *>(this)->serialize(archive, version);
  }


  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::load(boost::archive::binary_iarchive &archive,
                            const unsigned int               version)
  {
    serialize(archive, version);

    position.update_ghost_values();
    velocity.update_ghost_values();
  }

  template <int dim, int spacedim>
  template <class Archive>
  void
  Part<dim, spacedim>::serialize(Archive &ar, const unsigned int /*version*/)
  {
    boost::serialization::array_wrapper<double> position_wrapper(
      position.begin(), position.locally_owned_size());
    boost::serialization::array_wrapper<double> velocity_wrapper(
      velocity.begin(), velocity.locally_owned_size());

    ar &position_wrapper;
    ar &velocity_wrapper;
  }

  template <int dim, int spacedim>
  std::vector<ForceContribution<dim, spacedim> *>
  Part<dim, spacedim>::get_force_contributions() const
  {
    std::vector<ForceContribution<dim, spacedim> *> forces;

    for (auto &force_contribution : force_contributions)
      forces.push_back(force_contribution.get());

    return forces;
  }

  template <int dim, int spacedim>
  std::vector<ActiveStrain<dim, spacedim> *>
  Part<dim, spacedim>::get_active_strains() const
  {
    std::vector<ActiveStrain<dim, spacedim> *> strains;

    for (auto &as : active_strains)
      strains.push_back(as.get());

    return strains;
  }

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::add_force_contribution(
    std::unique_ptr<ForceContribution<dim, spacedim>> force)
  {
    force_contributions.push_back(std::move(force));
  }

  template class Part<NDIM - 1, NDIM>;
  template class Part<NDIM, NDIM>;
} // namespace fdl
