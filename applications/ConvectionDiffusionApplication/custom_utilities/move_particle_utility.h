// KRATOS ___ ___  _  ___   __   ___ ___ ___ ___
//       / __/ _ \| \| \ \ / /__|   \_ _| __| __|
//      | (_| (_) | .` |\ V /___| |) | || _|| _|
//       \___\___/|_|\_| \_/    |___/___|_| |_|  APPLICATION
//
//  License: BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:  Pablo Becker
//

#if !defined(KRATOS_MOVE_PARTICLE_UTILITY_FLUID_PFEM2_TRANSPORT_INCLUDED)
#define  KRATOS_MOVE_PARTICLE_UTILITY_FLUID_PFEM2_TRANSPORT_INCLUDED

// System includes
#include <string>
#include <iostream>
#include <algorithm>

// External includes

// Project includes
#include "includes/define.h"
#include "includes/node.h"

///
#include "includes/dof.h"
#include "includes/variables.h"
#include "containers/array_1d.h"
#include "containers/data_value_container.h"
#include "includes/mesh.h"
#include "utilities/math_utils.h"
///

#include "utilities/geometry_utilities.h"

#include "includes/model_part.h"


#include "spatial_containers/spatial_containers.h"
#include "spatial_containers/cell.h"
#include "spatial_containers/bins_dynamic_objects.h"

#include "utilities/spatial_containers_configure.h"

#include "geometries/line_2d_2.h"
#include "geometries/triangle_2d_3.h"
#include "geometries/triangle_3d_3.h"
#include "geometries/point.h"

#include "convection_diffusion_application.h"
#include "convection_particle.h"

#include "utilities/openmp_utils.h"
#include "utilities/parallel_utilities.h"
#include "time.h"

//#include "processes/process.h"

namespace Kratos
{
	//this class is to be modified by the user to customize the interpolation process
	template< unsigned int TDim>
	class MoveParticleUtilityScalarTransport
	{
	public:

	    typedef SpatialContainersConfigure<TDim>     Configure;
	    typedef typename Configure::PointType                      PointType;
	    //typedef PointType::CoordinatesArrayType           CoordinatesArrayType;
        typedef typename Configure::ContainerType                  ContainerType;
        //typedef Configure::PointerType                    PointerType;
        typedef typename Configure::IteratorType                   IteratorType;
        typedef typename Configure::ResultContainerType            ResultContainerType;
	    //typedef Configure::ResultPointerType              ResultPointerType;
        typedef typename Configure::ResultIteratorType             ResultIteratorType;
        typedef PointerVector< Convection_Particle, Convection_Particle*, std::vector<Convection_Particle*> > ParticlePointerVector;
        //typedef Configure::ContactPairType                ContactPairType;
        //typedef Configure::ContainerContactType           ContainerContactType;
        //typedef Configure::IteratorContactType            IteratorContactType;
        //typedef Configure::PointerContactType             PointerContactType;
        //typedef Configure::PointerTypeIterator            PointerTypeIterator;

		KRATOS_CLASS_POINTER_DEFINITION(MoveParticleUtilityScalarTransport);

		//template<unsigned int TDim>
		MoveParticleUtilityScalarTransport(ModelPart& model_part, int maximum_number_of_particles)
			: mr_model_part(model_part) , mmaximum_number_of_particles(maximum_number_of_particles) ,
			 mUnknownVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetUnknownVariable()) ,
			 mProjectionVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetProjectionVariable()) ,
			 mVelocityVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetVelocityVariable()) ,
			 mMeshVelocityVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetMeshVelocityVariable())
		{
			std::cout << "initializing moveparticle utility for scalar transport" << std::endl;

			Check();
            //storing water and air density and their inverses, just in case it is needed for the streamline integration
			//loop in elements to change their ID to their position in the array. Easier to get information later.
			//DO NOT PARALLELIZE THIS! IT MUST BE SERIAL!!!!!!!!!!!!!!!!!!!!!!
			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
			for(unsigned int  ii=0; ii<mr_model_part.Elements().size(); ii++)
			{
				ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
				ielem->SetId(ii+1);
			}
			mlast_elem_id= (mr_model_part.ElementsEnd()-1)->Id();
            int node_id=0;
            // we look for the smallest edge. could be used as a weighting function when going lagrangian->eulerian instead of traditional shape functions(method currently used)
			ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
			vector<unsigned int> node_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
				{
					ModelPart::NodesContainerType::iterator pnode = inodebegin+ii;
					array_1d<double,3> position_node;
					double distance=0.0;
					position_node = pnode->Coordinates();
					GlobalPointersVector< Node >& rneigh = pnode->GetValue(NEIGHBOUR_NODES);
					//we loop all the nodes to check all the edges
					const double number_of_neighbours = double(rneigh.size());
					for( GlobalPointersVector<Node >::iterator inode = rneigh.begin(); inode!=rneigh.end(); inode++)
					{
						array_1d<double,3> position_difference;
						position_difference = inode->Coordinates() - position_node;
						double current_distance= sqrt(pow(position_difference[0],2)+pow(position_difference[1],2)+pow(position_difference[2],2));
						//if (current_distance>distance)
						//	distance=current_distance;
						distance += current_distance / number_of_neighbours;
					}
					//and we save the largest edge.
					pnode->FastGetSolutionStepValue(MEAN_SIZE)=distance;

					node_id=pnode->GetId();
				}
			}
			mlast_node_id=node_id;

			//we also calculate the element mean size in the same way, for the courant number
			//also we set the right size to the LHS column for the pressure enrichments, in order to recover correctly the enrichment pressure
			vector<unsigned int> element_partition;
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

			//before doing anything we must reset the vector of nodes contained by each element (particles that are inside each element.
			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
				{
					ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;

					double mElemSize;
					array_1d<double,3> Edge(3,0.0);
					Edge = ielem->GetGeometry()[1].Coordinates() - ielem->GetGeometry()[0].Coordinates();
					mElemSize = Edge[0]*Edge[0];
					for (unsigned int d = 1; d < TDim; d++)
						mElemSize += Edge[d]*Edge[d];

					for (unsigned int i = 2; i < (TDim+1); i++)
						for(unsigned int j = 0; j < i; j++)
						{
							Edge = ielem->GetGeometry()[i].Coordinates() - ielem->GetGeometry()[j].Coordinates();
							double Length = Edge[0]*Edge[0];
							for (unsigned int d = 1; d < TDim; d++)
								Length += Edge[d]*Edge[d];
							if (Length < mElemSize) mElemSize = Length;
						}
					mElemSize = sqrt(mElemSize);
					ielem->GetValue(MEAN_SIZE) = mElemSize;
				}
			}


            //matrix containing the position of the 4/15/45 particles that we will seed at the beginning
            BoundedMatrix<double, 5*(1+TDim), 3 > pos;
            BoundedMatrix<double, 5*(1+TDim), (1+TDim) > N;

            int particle_id=0;
			mnelems = mr_model_part.Elements().size();

			std::cout << "about to resize vectors" << std::endl;


			//setting the right size to the vector containing the particles assigned to each element
			//particles vector. this vector contains ALL the particles in the simulation.
			mparticles_vector.resize(mnelems*mmaximum_number_of_particles);

			//and this vector contains the current number of particles that are in each element (currently zero)
			mnumber_of_particles_in_elems.resize(mnelems);
			mnumber_of_particles_in_elems=ZeroVector(mnelems);

			//when moving the particles, an auxiliary vector is necessary (to store the previous number)
			mnumber_of_particles_in_elems_aux.resize(mnelems);

			//each element will have a list of pointers to all the particles that are inside.
			//this vector contains the pointers to the vector of (particle) pointers of each element.
			mvector_of_particle_pointers_vectors.resize(mnelems);
            //int artz;
            //std::cin >> artz;
			int i_int=0; //careful! it's not the id, but the position inside the array!
			std::cout << "about to create particles" << std::endl;
			//now we seed: LOOP IN ELEMENTS
			//using loop index, DO NOT parallelize this! change lines : mparticles_in_elems_pointers((ii*mmaximum_number_of_particles)+mparticles_in_elems_integers(ii)) = pparticle; and the next one

			moffset=0;
			//Convection_Particle& firstparticle =mparticles_vector[0];
			for(unsigned int ii=0; ii<mr_model_part.Elements().size(); ii++)
			{
				ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
				//(ielem->GetValue(BED_PARTICLE_POINTERS)) = ParticlePointerVector( mmaximum_number_of_particles*2, &firstparticle );
				//ParticlePointerVector&  particle_pointers =  (ielem->GetValue(BED_PARTICLE_POINTERS));
				//now we link the mpointers_to_particle_pointers_vectors to the corresponding element
				//mpointers_to_particle_pointers_vectors(ii) = &particle_pointers;
				//now we resize the vector of particle pointers. it is double sized because we move the particles from an initial position (first half) to a final position (second half).
				//for(int j=0; j<(mmaximum_number_of_particles*2); j++)
				//        particle_pointers.push_back(&firstparticle);
				mvector_of_particle_pointers_vectors[ii] = ParticlePointerVector( mmaximum_number_of_particles*2 );
				ParticlePointerVector& particle_pointers = mvector_of_particle_pointers_vectors[ii];
				//int & number_of_particles = ielem->GetValue(NUMBER_OF_BED_PARTICLES);
				int & number_of_particles = mnumber_of_particles_in_elems[ii];
				number_of_particles=0;

				Geometry< Node >& geom = ielem->GetGeometry();
				//unsigned int elem_id = ielem->Id();
				//mareas_vector[i_int]=CalculateArea(geom); UNUSED SO COMMENTED
				ComputeGaussPointPositions_initial(geom, pos, N); //we also have the standard (4), and 45
				//now we seed the particles in the current element
				for (unsigned int j = 0; j < pos.size1(); j++)
				{
					++particle_id;

					Convection_Particle& pparticle = mparticles_vector[particle_id-1];
					pparticle.X()=pos(j,0);
					pparticle.Y()=pos(j,1);
					pparticle.Z()=pos(j,2);

					pparticle.GetEraseFlag()=false;

					float & scalar1= pparticle.GetScalar1();
					scalar1=0.0;

                    for (unsigned int k = 0; k < (TDim+1); k++)
                    {
						scalar1 +=  N(j, k) * geom[k].FastGetSolutionStepValue(mUnknownVar);
					}


					particle_pointers(j) = &pparticle;
					 number_of_particles++ ;

				}
				++i_int;
			}

			m_nparticles=particle_id; //we save the last particle created as the total number of particles we have. For the moment this is true.
			KRATOS_WATCH(m_nparticles);
			//KRATOS_WATCH(mlast_elem_id);
			mparticle_printing_tool_initialized=false;
			//std::cin >> artz;
		}


		virtual ~MoveParticleUtilityScalarTransport()
		{}

		void MountBin()
		{
			KRATOS_TRY

			//copy the elements to a new container, as the list will
			//be shuffled duringthe construction of the tree
			ContainerType& rElements           =  mr_model_part.ElementsArray();
	        IteratorType it_begin              =  rElements.begin();
	        IteratorType it_end                =  rElements.end();
	        //const int number_of_elem 		   =   rElements.size();

			typename BinsObjectDynamic<Configure>::Pointer paux = typename BinsObjectDynamic<Configure>::Pointer(new BinsObjectDynamic<Configure>(it_begin, it_end  ) );
			paux.swap(mpBinsObjectDynamic);
			//BinsObjectDynamic<Configure>  mpBinsObjectDynamic(it_begin, it_end );

			std::cout << "finished mounting Bins" << std::endl;

			KRATOS_CATCH("")
		}

		void MountBin(const double CellSize)
		{
			KRATOS_TRY

			//copy the elements to a new container, as the list will
			//be shuffled duringthe construction of the tree
			ContainerType& rElements           =  mr_model_part.ElementsArray();
	        IteratorType it_begin              =  rElements.begin();
	        IteratorType it_end                =  rElements.end();
			typename BinsObjectDynamic<Configure>::Pointer paux = typename BinsObjectDynamic<Configure>::Pointer(new BinsObjectDynamic<Configure>(it_begin, it_end, CellSize ) );
			paux.swap(mpBinsObjectDynamic);

			KRATOS_INFO("MoveParticleUtilityScalarTransport") << "Finished mounting Bins with cell size: " << CellSize << std::endl;

			KRATOS_CATCH("")
		}

		void CalculateVelOverElemSize()
		{
			KRATOS_TRY

			//ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();

			const double nodal_weight = 1.0/ (1.0 + double (TDim) );

			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
			vector<unsigned int> element_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
				{
					ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
					Geometry<Node >& geom = ielem->GetGeometry();

					array_1d<double, 3 >vector_mean_velocity=ZeroVector(3);

					for (unsigned int i=0; i != (TDim+1) ; i++)
						vector_mean_velocity += geom[i].FastGetSolutionStepValue(mVelocityVar);
					vector_mean_velocity *= nodal_weight;

					const double mean_velocity = sqrt ( pow(vector_mean_velocity[0],2) + pow(vector_mean_velocity[1],2) + pow(vector_mean_velocity[2],2) );
					ielem->GetValue(MEAN_VEL_OVER_ELEM_SIZE) = mean_velocity / (ielem->GetValue(MEAN_SIZE));
				}
			}
			KRATOS_CATCH("")
		}



		//name self explained
		void ResetBoundaryConditions()
		{
				KRATOS_TRY

				ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
				vector<unsigned int> node_partition;
				#ifdef _OPENMP
					int number_of_threads = omp_get_max_threads();
				#else
					int number_of_threads = 1;
				#endif
				OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

				#pragma omp parallel for
				for(int kkk=0; kkk<number_of_threads; kkk++)
				{
					for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
					{
							ModelPart::NodesContainerType::iterator inode = inodebegin+ii;

							if (inode->IsFixed(mUnknownVar))
							{
								inode->FastGetSolutionStepValue(mUnknownVar)=inode->GetSolutionStepValue(mUnknownVar,1);
							}
					}
				}

				KRATOS_CATCH("")
		}

		void CalculateDeltaVariables()
		{
			KRATOS_TRY
			ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
			vector<unsigned int> node_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
				{
						ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
						inode->FastGetSolutionStepValue(DELTA_SCALAR1) = inode->FastGetSolutionStepValue(mUnknownVar) - inode->FastGetSolutionStepValue(mProjectionVar) ;
				}
			}

			KRATOS_CATCH("")
		}


        void CopyScalarVarToPreviousTimeStep(const Variable<double>& OriginVariable,
                       ModelPart::NodesContainerType& rNodes)
        {
			KRATOS_TRY
			ModelPart::NodesContainerType::iterator inodebegin = rNodes.begin();
			vector<unsigned int> node_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, rNodes.size(), node_partition);

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
				{
					ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
				    inode->GetSolutionStepValue(OriginVariable,1) = inode->FastGetSolutionStepValue(OriginVariable);
				}
			}
			KRATOS_CATCH("")
        }


		//to move all the particles across the streamlines. heavy task!
		void MoveParticles()
		{

			KRATOS_TRY

			ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();

			const int offset = moffset; //the array of pointers for each element has twice the required size so that we use a part in odd timesteps and the other in even ones.
																				//moveparticlesdiff reads from the pointers of one part (ie odd) and saves into the other part (ie even part)
																				//since it is the only function in the whole procedure that does this, it must use alternatively one part and the other.
			//KRATOS_WATCH(offset)

			bool even_timestep;
			if (offset!=0) even_timestep=false;
			else even_timestep=true;

			const int post_offset = mmaximum_number_of_particles*int(even_timestep);	//and we also save the offset to know the location in which we will save the pointers after we've moved the particles
			//KRATOS_WATCH(post_offset)


			double delta_t = CurrentProcessInfo[DELTA_TIME];

			array_1d<double,TDim+1> N;
			const unsigned int max_results = 10000;

			//double integration_distance= 2.0;

			max_nsubsteps = 10;
			max_substep_dt=delta_t/double(max_nsubsteps);

			vector<unsigned int> element_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();



			//before doing anything we must reset the vector of nodes contained by each element (particles that are inside each element.
			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
				{
					//ModelPart::ElementsContainerType::iterator old_element = ielembegin+ii;

					int & number_of_particles = mnumber_of_particles_in_elems[ii]; //old_element->GetValue(NUMBER_OF_BED_PARTICLES);

					mnumber_of_particles_in_elems_aux[ii]=number_of_particles;
					mnumber_of_particles_in_elems[ii]=0;
					//we reset the local vectors for a faster access;
				}
			}
			std::cout << "convecting particles" << std::endl;
            //We move the particles across the fixed mesh and saving change data into them (using the function MoveParticle)

			#pragma omp barrier

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{

			  ResultContainerType results(max_results);

			  GlobalPointersVector< Element > elements_in_trajectory;
			  elements_in_trajectory.resize(20);

			  for(unsigned int ielem=element_partition[kkk]; ielem<element_partition[kkk+1]; ielem++)
			  {
			//for(unsigned int ielem=0; ielem<mr_model_part.Elements().size(); ielem++)
			//{

				ModelPart::ElementsContainerType::iterator old_element = ielembegin+ielem;
				const int old_element_id = old_element->Id();

				ParticlePointerVector& old_element_particle_pointers = mvector_of_particle_pointers_vectors(old_element_id-1);

				if ( (results.size()) !=max_results)
						results.resize(max_results);

				unsigned int number_of_elements_in_trajectory=0; //excluding the origin one (current one, ielem)

				for(int ii=0; ii<(mnumber_of_particles_in_elems_aux(ielem)); ii++)
				{

					Convection_Particle & pparticle = old_element_particle_pointers[offset+ii];

					Element::Pointer pcurrent_element( *old_element.base() );
					ResultIteratorType result_begin = results.begin();
					bool & erase_flag=pparticle.GetEraseFlag();
					if (erase_flag==false){
						MoveParticle(pparticle,pcurrent_element,elements_in_trajectory,number_of_elements_in_trajectory,result_begin,max_results); //saqué N de los argumentos, no lo necesito ya q empieza SIEMPRE en un nodo y no me importa donde termina

						const int current_element_id = pcurrent_element->Id();

						int & number_of_particles_in_current_elem = mnumber_of_particles_in_elems(current_element_id-1);
						//int & number_of_water_particles_in_current_elem = mnumber_of_water_particles_in_elems(current_element_id-1);

						if (number_of_particles_in_current_elem<mmaximum_number_of_particles && erase_flag==false)
						{
							{

								ParticlePointerVector& current_element_particle_pointers = mvector_of_particle_pointers_vectors(current_element_id-1);

								#pragma omp critical
								{
									if (number_of_particles_in_current_elem<mmaximum_number_of_particles) // we cant go over this node, there's no room. otherwise we would be in the position of the first particle of the next element!!
									{

										current_element_particle_pointers(post_offset+number_of_particles_in_current_elem) = &pparticle;

										number_of_particles_in_current_elem++ ;
										if (number_of_particles_in_current_elem>mmaximum_number_of_particles)
											KRATOS_WATCH("MAL");

									}
									else
										pparticle.GetEraseFlag()=true; //so we just delete it!
								}
							}
						}
						else
							pparticle.GetEraseFlag()=true; //so we just delete it!



					}
				}
			  }
			}
			/*
			//now we pass info from the local vector to the elements:
			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
				{
					ModelPart::ElementsContainerType::iterator old_element = ielembegin+ii;

					old_element->GetValue(NUMBER_OF_BED_PARTICLES) = mnumber_of_particles_in_elems(ii);
					//old_element->GetValue(NUMBER_OF_WATER_PARTICLES) = mnumber_of_water_particles_in_elems(ii);
				}

			}
			*/

			//after having changed everything we change the status of the modd_timestep flag:
			moffset = post_offset;; //

			KRATOS_CATCH("")
		}

		void TransferLagrangianToEulerian() //explicit
		{
			KRATOS_TRY

			//ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();
			//const double delta_t =CurrentProcessInfo[DELTA_TIME];
			const double threshold= 0.0/(double(TDim)+1.0);


			std::cout << "projecting info to mesh" << std::endl;


			const int offset = moffset; //the array of pointers for each element has twice the required size so that we use a part in odd timesteps and the other in even ones.
			//KRATOS_WATCH(offset)																	//(flag managed only by MoveParticles

			//we must project data from the particles (lagrangian)  into the eulerian mesh
			//ValuesVectorType eulerian_nodes_old_temperature;
			//int nnodes = mr_model_part.Nodes().size();
			//array_1d<double,(n_nodes)> eulerian_nodes_sumweights;

			//we save data from previous time step of the eulerian mesh in case we must reuse it later cos no particle was found around the nodes
			//though we could've use a bigger buffer, to be changed later!
			//after having saved data, we reset them to zero, this way it's easier to add the contribution of the surrounding particles.
			ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
			vector<unsigned int> node_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
				{
					ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
					inode->FastGetSolutionStepValue(mProjectionVar)=0.0;
					inode->FastGetSolutionStepValue(YP)=0.0;
				}

			}

			//adding contribution, loop on elements, since each element has stored the particles found inside of it
			vector<unsigned int> element_partition;
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
				{
					ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;

					array_1d<double,3*(TDim+1)> nodes_positions;
					array_1d<double,(TDim+1)> nodes_added_scalar1 = ZeroVector((TDim+1));
					array_1d<double,(TDim+1)> nodes_addedweights = ZeroVector((TDim+1));
					//array_1d<double,(TDim+1)> weighting_inverse_divisor;

					Geometry<Node >& geom = ielem->GetGeometry();

					for (int i=0 ; i!=(TDim+1) ; ++i)
					{
						nodes_positions[i*3+0]=geom[i].X();
						nodes_positions[i*3+1]=geom[i].Y();
						nodes_positions[i*3+2]=geom[i].Z();
						//weighting_inverse_divisor[i]=1.0/((geom[i].FastGetSolutionStepValue(MEAN_SIZE))*1.01);
					}
					///KRATOS_WATCH(ielem->Id())
					///KRATOS_WATCH(ielem->GetValue(NEIGHBOUR_NODES).size());

					//int & number_of_particles_in_elem= ielem->GetValue(NUMBER_OF_BED_PARTICLES);
					//ParticlePointerVector&  element_particle_pointers =  (ielem->GetValue(BED_PARTICLE_POINTERS));
					int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
					ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];


					for (int iii=0; iii<number_of_particles_in_elem ; iii++ )
					{
						if (iii==mmaximum_number_of_particles) //it means we are out of our portion of the array, abort loop!
							break;

						Convection_Particle & pparticle = element_particle_pointers[offset+iii];

						if (pparticle.GetEraseFlag()==false)
						{

							array_1d<double,3> & position = pparticle.Coordinates();

							const float& particle_scalar1 = pparticle.GetScalar1();  // -1 if water, +1 if air

							array_1d<double,TDim+1> N;
							bool is_found = CalculatePosition(nodes_positions,position[0],position[1],position[2],N);
							if (is_found==false) //something went wrong. if it was close enough to the edge we simply send it inside the element.
							{
								KRATOS_WATCH(N);
								for (int j=0 ; j!=(TDim+1); j++)
									if (N[j]<0.0 && N[j]> -1e-5)
										N[j]=1e-10;

							}

							for (int j=0 ; j!=(TDim+1); j++) //going through the 3/4 nodes of the element
							{
								//double sq_dist = 0;
								//these lines for a weighting function based on the distance (or square distance) from the node insteadof the shape functions
								//for (int k=0 ; k!=(TDim); k++) sq_dist += ((position[k] - nodes_positions[j*3+k])*(position[k] - nodes_positions[j*3+k]));
								//double weight = (1.0 - (sqrt(sq_dist)*weighting_inverse_divisor[j] ) );

								double weight=N(j)*N(j);
								//weight=N(j)*N(j)*N(j);
								if (weight<threshold) weight=1e-10;
								if (weight<0.0) {KRATOS_WATCH(weight)}//;weight=0.0;KRATOS_WATCH(velocity);KRATOS_WATCH(N);KRATOS_WATCH(number_of_particles_in_elem);}//{KRATOS_WATCH(weight); KRATOS_WATCH(geom[j].Id()); KRATOS_WATCH(position);}
								else
								{
									nodes_addedweights[j]+= weight;
									//nodes_addedtemp[j] += weight * particle_temp;

									nodes_added_scalar1[j] += weight*particle_scalar1;


								}//
							}
						}
					}

					for (int i=0 ; i!=(TDim+1) ; ++i) {
						geom[i].SetLock();
						geom[i].FastGetSolutionStepValue(mProjectionVar) +=nodes_added_scalar1[i];
						geom[i].FastGetSolutionStepValue(YP) +=nodes_addedweights[i];
						geom[i].UnSetLock();
					}
				}
			}

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
				{
					ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
					double sum_weights = inode->FastGetSolutionStepValue(YP);
					if (sum_weights>0.00001)
					{
						//inode->FastGetSolutionStepValue(TEMPERATURE_OLD_IT)=(inode->FastGetSolutionStepValue(TEMPERATURE_OLD_IT))/sum_weights; //resetting the temperature
						double & height = inode->FastGetSolutionStepValue(mProjectionVar);
						 height /=sum_weights; //resetting the density
					}

					else //this should never happen because other ways to recover the information have been executed before, but leaving it just in case..
					{
						inode->FastGetSolutionStepValue(mProjectionVar)=inode->FastGetSolutionStepValue(mUnknownVar,1); //resetting the temperature
					}
				}
			}


			KRATOS_CATCH("")
		}



		void TransferLagrangianToEulerianImp() //semi implicit
		{
			KRATOS_TRY

			  //	ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();

			std::cout << "projecting info to mesh (semi implicit)" << std::endl;


			const int offset = moffset; //the array of pointers for each element has twice the required size so that we use a part in odd timesteps and the other in even ones.
			//KRATOS_WATCH(offset)																	//(flag managed only by MoveParticles

			//we must project data from the particles (lagrangian)  into the eulerian mesh
			//ValuesVectorType eulerian_nodes_old_temperature;
			//int nnodes = mr_model_part.Nodes().size();
			//array_1d<double,(n_nodes)> eulerian_nodes_sumweights;

			//we save data from previous time step of the eulerian mesh in case we must reuse it later cos no particle was found around the nodes
			//though we could've use a bigger buffer, to be changed later!
			//after having saved data, we reset them to zero, this way it's easier to add the contribution of the surrounding particles.
			ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
			vector<unsigned int> node_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
				{
					ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
					inode->FastGetSolutionStepValue(mProjectionVar)=0.0;
					inode->FastGetSolutionStepValue(YP)=0.0;
				}

			}

			//adding contribution, loop on elements, since each element has stored the particles found inside of it
			vector<unsigned int> element_partition;
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{

				//creating a matrix for each of the problems.
				BoundedMatrix<double, TDim+1 , TDim+1  > mass_matrix; // WE ONLY NEED ONE! they are the same for all the variables!  //_x,mass_matrix_y,mass_matrix_z,mass_matrix_d; //mass matrices for the projected vel (x,y,z) and the distance
				array_1d<double,(TDim+1)> rhs_scalar1;

				array_1d<double,3*(TDim+1)> nodes_positions;
				array_1d<double,(TDim+1)> nodes_added_scalar1 = ZeroVector((TDim+1));
				array_1d<double,(TDim+1)> nodes_addedweights = ZeroVector((TDim+1));

				for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
				{
					ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;

					nodes_added_scalar1 = ZeroVector((TDim+1));   //resetting vectors
					nodes_addedweights = ZeroVector((TDim+1));     //resetting vectors
					mass_matrix = ZeroMatrix(TDim+1 , TDim+1 );  //resetting matrices. WE ONLY NEED ONE! they are the same for all the variable. only the rhs changes.
					//mass_matrix_y = ZeroMatrix(TDim+1 , TDim+1 );  //resetting matrices
					//mass_matrix_z = ZeroMatrix(TDim+1 , TDim+1 );  //resetting matrices
					//mass_matrix_d = ZeroMatrix(TDim+1 , TDim+1 );  //resetting matrices
					rhs_scalar1 = ZeroVector((TDim+1));         //resetting vectors

					Geometry<Node >& geom = ielem->GetGeometry();
					const double elem_volume = geom.Area();

					for (int i=0 ; i!=(TDim+1) ; ++i)  //saving the nodal positions for faster access
					{
						nodes_positions[i*3+0]=geom[i].X();
						nodes_positions[i*3+1]=geom[i].Y();
						nodes_positions[i*3+2]=geom[i].Z();
					}
					///KRATOS_WATCH(ielem->Id())
					///KRATOS_WATCH(ielem->GetValue(NEIGHBOUR_NODES).size());

					//int & number_of_particles_in_elem= ielem->GetValue(NUMBER_OF_BED_PARTICLES);
					//ParticlePointerVector&  element_particle_pointers =  (ielem->GetValue(BED_PARTICLE_POINTERS));
					int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
					ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];

					for (int iii=0; iii<number_of_particles_in_elem ; iii++ )
					{
						if (iii==mmaximum_number_of_particles) //it means we are out of our portion of the array, abort loop!
							break;

						Convection_Particle & pparticle = element_particle_pointers[offset+iii];

						if (pparticle.GetEraseFlag()==false)
						{

							array_1d<double,3> & position = pparticle.Coordinates();

							const float& particle_scalar1 = pparticle.GetScalar1();  // -1 if water, +1 if air

							array_1d<double,TDim+1> N;
							bool is_found = CalculatePosition(nodes_positions,position[0],position[1],position[2],N);
							if (is_found==false) //something went wrong. if it was close enough to the edge we simply send it inside the element.
							{
								KRATOS_WATCH(N);
								for (int j=0 ; j!=(TDim+1); j++)
									if (N[j]<0.0 && N[j]> -1e-5)
										N[j]=1e-10;

							}

							for (int j=0 ; j!=(TDim+1); j++) //going through the 3/4 nodes of the element
							{
								double weight=N(j);
								for (int k=0 ; k!=(TDim+1); k++) //building the mass matrix
									mass_matrix(j,k) += weight*N(k);

								rhs_scalar1[j] += weight * double(particle_scalar1);

								//adding also a part with the lumped mass matrix to reduce overshoots and undershoots
								if(true)
								{
									double this_particle_weight = weight*elem_volume/(double(number_of_particles_in_elem))*0.1; //can be increased or reduced to change the lumped mass contrubtion
									nodes_addedweights[j]+= this_particle_weight;
									nodes_added_scalar1[j] += this_particle_weight*particle_scalar1;
								}
							}
						}
					}

					//now we invert the matrix
					BoundedMatrix<double, TDim+1 , TDim+1  > inverse_mass_matrix=ZeroMatrix(TDim+1 , TDim+1);
					if constexpr (TDim==3)
						InvertMatrix( mass_matrix,  inverse_mass_matrix);
					else
						InvertMatrix3x3( mass_matrix,  inverse_mass_matrix);
					//and now compute the elemental contribution to the global system:

					if(number_of_particles_in_elem > static_cast<int>(TDim)*3) //otherwise it's impossible to define a correctly the gradients, therefore the results inside the element are useless.
					{
						for (int i=0 ; i!=(TDim+1); i++)
						{
							for (int j=0 ; j!=(TDim+1); j++)
							{
								nodes_added_scalar1[i] += inverse_mass_matrix(i,j)*rhs_scalar1[j]*elem_volume*(1.0/(double(1+TDim)));

							}
						}
						//and also to the mass matrix. LUMPED (but for the contribution of the grandient at elemental level.
						for (int i=0 ; i!=(TDim+1); i++)
							nodes_addedweights[i] += elem_volume*(1.0/(double(1+TDim)));
					}


					for (int i=0 ; i!=(TDim+1) ; ++i) {
						geom[i].SetLock();
						geom[i].FastGetSolutionStepValue(mProjectionVar) +=nodes_added_scalar1[i];
						geom[i].FastGetSolutionStepValue(YP) +=nodes_addedweights[i];
						geom[i].UnSetLock();
					}
				}
			}

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
				{
					ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
					double sum_weights = inode->FastGetSolutionStepValue(YP);
					if (sum_weights>0.00001)
					{
						 double & scalar1 = inode->FastGetSolutionStepValue(mProjectionVar);
						 scalar1 /=sum_weights; //resetting the density
					}

					else //this should never happen because other ways to recover the information have been executed before, but leaving it just in case..
					{
						inode->FastGetSolutionStepValue(mProjectionVar)=inode->FastGetSolutionStepValue(mUnknownVar,1);
					}
				}
			}


			KRATOS_CATCH("")
		}

		void CorrectParticlesWithoutMovingUsingDeltaVariables()
		{
			KRATOS_TRY
			//std::cout << "updating particles" << std::endl;
			//ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();

			const int offset = moffset; //the array of pointers for each element has twice the required size so that we use a part in odd timesteps and the other in even ones.
																				//(flag managed only by MoveParticles
			//KRATOS_WATCH(offset)
			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();


			vector<unsigned int> element_partition;
			#ifdef _OPENMP
				int number_of_threads = omp_get_max_threads();
			#else
				int number_of_threads = 1;
			#endif
			OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

			#pragma omp parallel for
			for(int kkk=0; kkk<number_of_threads; kkk++)
			{
				for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
				{
					//const int & elem_id = ielem->Id();
					ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
					Element::Pointer pelement(*ielem.base());
					Geometry<Node >& geom = ielem->GetGeometry();

					//ParticlePointerVector&  element_particle_pointers =  (ielem->GetValue(BED_PARTICLE_POINTERS));
					//int & number_of_particles_in_elem=ielem->GetValue(NUMBER_OF_BED_PARTICLES);
					int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
					ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];
					//std::cout << "elem " << ii << " with " << (unsigned int)number_of_particles_in_elem << " particles" << std::endl;

					for (int iii=0; iii<number_of_particles_in_elem ; iii++ )
					{
						//KRATOS_WATCH(iii)
						if (iii>mmaximum_number_of_particles) //it means we are out of our portion of the array, abort loop!
							break;

						Convection_Particle & pparticle = element_particle_pointers[offset+iii];


						bool erase_flag= pparticle.GetEraseFlag();
						if (erase_flag==false)
						{
							CorrectParticleUsingDeltaVariables(pparticle,pelement,geom); //'lite' version, we pass by reference the geometry, so much cheaper
						}


					}
				}
			}
			KRATOS_CATCH("")
		}

		//**************************************************************************************************************
		//**************************************************************************************************************


		template< class TDataType > void  AddUniqueWeakPointer
			(GlobalPointersVector< TDataType >& v, const typename TDataType::WeakPointer candidate)
		{
			typename GlobalPointersVector< TDataType >::iterator i = v.begin();
			typename GlobalPointersVector< TDataType >::iterator endit = v.end();
			while ( i != endit && (i)->Id() != (candidate)->Id())
			{
				i++;
			}
			if( i == endit )
			{
				v.push_back(candidate);
			}

		}

		//**************************************************************************************************************
		//**************************************************************************************************************

		void PreReseed(int minimum_number_of_particles)
		{
			KRATOS_TRY


			//ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();
			const int offset =moffset;
			const int max_results = 1000;

			//tools for the parallelization
			unsigned int number_of_threads = ParallelUtilities::GetNumThreads();
			vector<unsigned int> elem_partition;
			int number_of_rows=mr_model_part.Elements().size();
			elem_partition.resize(number_of_threads + 1);
			int elem_partition_size = number_of_rows / number_of_threads;
			elem_partition[0] = 0;
			elem_partition[number_of_threads] = number_of_rows;
			//KRATOS_WATCH(elem_partition_size);
			for (unsigned int i = 1; i < number_of_threads; i++)
			elem_partition[i] = elem_partition[i - 1] + elem_partition_size;
			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();

			#pragma omp parallel firstprivate(elem_partition)
			{
				ResultContainerType results(max_results);
				int k = OpenMPUtils::ThisThread();
				//ModelPart::ElementsContainerType::iterator it_begin = mr_model_part.ElementsBegin() +  elem_partition[k];
				//ModelPart::ElementsContainerType::iterator it_end = mr_model_part.ElementsBegin() + elem_partition[k+1] ;
				//ModelPart::NodesContainerType local_list=aux[k];
				//PointerVectorSet<Convection_Particle, IndexedObject> & list=aux[k];
				//KRATOS_WATCH(k);
			    BoundedMatrix<double, (TDim+1), 3 > pos;
				BoundedMatrix<double, (TDim+1) , (TDim+1) > N;
				unsigned int freeparticle=0; //we start with the first position in the particles array

				//int local_id=1;
				for(unsigned int ii=elem_partition[k]; ii<elem_partition[k+1]; ii++)
				{
					//const int & elem_id = ielem->Id();
					ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
					results.resize(max_results);
					//const int & elem_id = ielem->Id();
					//ParticlePointerVector&  element_particle_pointers =  (ielem->GetValue(BED_PARTICLE_POINTERS));
					//int & number_of_particles_in_elem=ielem->GetValue(NUMBER_OF_BED_PARTICLES);
					int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
					ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];
					if (number_of_particles_in_elem<(minimum_number_of_particles))// && (ielem->GetGeometry())[0].Y()<0.10 )
				    	{
						//KRATOS_WATCH("elem with little particles")
						Geometry< Node >& geom = ielem->GetGeometry();
						ComputeGaussPointPositionsForPreReseed(geom, pos, N);
						//double conductivity = ielem->GetProperties()[CONDUCTIVITY];
						//KRATOS_WATCH(conductivity);
						for (unsigned int j = 0; j < (pos.size1()); j++) //i am dropping the last one, the one in the middle of the element
						{
							bool keep_looking = true;
							while(keep_looking)
							{
								if (mparticles_vector[freeparticle].GetEraseFlag()==true)
								{
									#pragma omp critical
									{
										if (mparticles_vector[freeparticle].GetEraseFlag()==true)
										{
											mparticles_vector[freeparticle].GetEraseFlag()=false;
											keep_looking=false;
										}
									}
									if (keep_looking==false)
										break;
									else
										freeparticle++;
								}
								else
								{
										freeparticle++;
								}
							}

							Convection_Particle pparticle(pos(j,0),pos(j,1),pos(j,2));

							array_1d<double,TDim+1>aux2_N;
							bool is_found = CalculatePosition(geom,pos(j,0),pos(j,1),pos(j,2),aux2_N);
							if (is_found==false)
							{
								KRATOS_WATCH(aux2_N);
							}


							pparticle.GetEraseFlag()=false;

						    ResultIteratorType result_begin = results.begin();
							Element::Pointer pelement( *ielem.base() );
							MoveParticle_inverse_way(pparticle, pelement, result_begin, max_results);

							 //and we copy it to the array:
							 mparticles_vector[freeparticle] =  pparticle;

							 element_particle_pointers(offset+number_of_particles_in_elem) = &mparticles_vector[freeparticle];
							 pparticle.GetEraseFlag()=false;

							number_of_particles_in_elem++;


						  }
					  }
				  }
			}




			KRATOS_CATCH("")
		}


		//**************************************************************************************************************
		//**************************************************************************************************************


		void PostReseed(int minimum_number_of_particles) //pooyan's way
		{
			KRATOS_TRY

			//ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();
			const int offset = moffset;

			//TOOLS FOR THE PARALLELIZATION
			//int last_id= (mr_linea_model_part.NodesEnd()-1)->Id();
			unsigned int number_of_threads = ParallelUtilities::GetNumThreads();
			//KRATOS_WATCH(number_of_threads);
			vector<unsigned int> elem_partition;
			int number_of_rows=mr_model_part.Elements().size();
			//KRATOS_WATCH(number_of_threads);
			//KRATOS_THROW_ERROR(std::logic_error, "Add  ----NODAL_H---- variable!!!!!! ERROR", "");
			elem_partition.resize(number_of_threads + 1);
			int elem_partition_size = number_of_rows / number_of_threads;
			elem_partition[0] = 0;
			elem_partition[number_of_threads] = number_of_rows;
			//KRATOS_WATCH(elem_partition_size);
			for (unsigned int i = 1; i < number_of_threads; i++)
			elem_partition[i] = elem_partition[i - 1] + elem_partition_size;
			//typedef Node PointType;
			//std::vector<ModelPart::NodesContainerType> aux;// aux;
			//aux.resize(number_of_threads);

			//ModelPart::NodesContainerType::iterator it_begin_particle_model_part = mr_linea_model_part.NodesBegin();
			//ModelPart::NodesContainerType::iterator it_end_particle_model_part = mr_linea_model_part.NodesEnd();
			ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();

			#pragma omp parallel firstprivate(elem_partition) // firstprivate(results)//we will add the nodes in different parts of aux and later assemble everything together, remaming particles ids to get consecutive ids
			{
				unsigned int reused_particles=0;

				unsigned int freeparticle = 0; //we start by the first position;

				int k = OpenMPUtils::ThisThread();
				//ModelPart::ElementsContainerType::iterator it_begin = mr_model_part.ElementsBegin() +  elem_partition[k];
				//ModelPart::ElementsContainerType::iterator it_end = mr_model_part.ElementsBegin() + elem_partition[k+1] ;

				BoundedMatrix<double, (3+2*TDim), 3 > pos; //7 particles (2D) or 9 particles (3D)
				BoundedMatrix<double, (3+2*TDim), (TDim+1) > N;

				double mesh_scalar1;

				array_1d<int, (3+2*TDim) > positions;

				unsigned int number_of_reseeded_particles;
				//unsigned int number_of_water_reseeded_particles;

				//array_1d<double, 3 > nodes_distances;

				for(unsigned int ii=elem_partition[k]; ii<elem_partition[k+1]; ii++)
				{
					//const int & elem_id = ielem->Id();
					ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;

					//int & number_of_particles_in_elem= ielem->GetValue(NUMBER_OF_BED_PARTICLES);
					//ParticlePointerVector&  element_particle_pointers =  (ielem->GetValue(BED_PARTICLE_POINTERS));
					int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
					ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];

					Geometry< Node >& geom = ielem->GetGeometry();
					if ( (number_of_particles_in_elem<(minimum_number_of_particles)))// && (geom[0].Y()<0.10) ) || (number_of_water_particles_in_elem>2 && number_of_particles_in_elem<(minimum_number_of_particles) ) )
				    {

						//bool reseed_more=false;
						number_of_reseeded_particles=0;


						//reseed_more=true;
						number_of_reseeded_particles= 3+2*TDim;
						ComputeGaussPointPositionsForPostReseed(geom, pos, N);

						for (unsigned int j = 0; j < number_of_reseeded_particles; j++)
						{
							//now we have to find an empty space ( a particle that was about to be deleted) in the particles model part. once found. there will be our renewed particle:
							bool keep_looking = true;
							while(keep_looking)
							{
								if (mparticles_vector[freeparticle].GetEraseFlag()==true)
								{
									#pragma omp critical
									{
										if (mparticles_vector[freeparticle].GetEraseFlag()==true)
										{
											mparticles_vector[freeparticle].GetEraseFlag()=false;
											keep_looking=false;
										}
									}
									if (keep_looking==false)
										break;

									else
										freeparticle++;
								}
								else
								{
										freeparticle++;
								}
							}

							Convection_Particle pparticle(pos(j,0),pos(j,1),pos(j,2));

							array_1d<double,TDim+1>aux_N;
							bool is_found = CalculatePosition(geom,pos(j,0),pos(j,1),pos(j,2),aux_N);
							if (is_found==false)
							{
								KRATOS_WATCH(aux_N);
								KRATOS_WATCH(j)
								KRATOS_WATCH(ielem->Id())
							}

							mesh_scalar1 = 0.0;

							for (unsigned int l = 0; l < (TDim+1); l++)
							{
								mesh_scalar1 +=  N(j,l) * geom[l].FastGetSolutionStepValue(mUnknownVar);
							}
							pparticle.GetScalar1()=mesh_scalar1;
							pparticle.GetEraseFlag()=false;


							mparticles_vector[freeparticle]=pparticle;
							element_particle_pointers(offset+number_of_particles_in_elem) = &mparticles_vector[freeparticle];
							number_of_particles_in_elem++;


							if (keep_looking)
							{
								KRATOS_THROW_ERROR(std::logic_error, "FINISHED THE LIST AND COULDN'T FIND A FREE CELL FOR THE NEW PARTICLE!", "");
							}
						    else
						    {
								reused_particles++;
							}

						  }
					  }
				  }
			}

			KRATOS_CATCH("")
		}

		void ExecuteParticlesPritingTool( ModelPart& lagrangian_model_part, int input_filter_factor )
		{
			KRATOS_TRY
			//mfilter_factor; //we will only print one out of every "filter_factor" particles of the total particle list

			if(mparticle_printing_tool_initialized==false)
			{
				mfilter_factor=input_filter_factor;

				if(lagrangian_model_part.NodesBegin()-lagrangian_model_part.NodesEnd()>0)
					KRATOS_THROW_ERROR(std::logic_error, "AN EMPTY MODEL PART IS REQUIRED FOR THE PRINTING OF PARTICLES", "");

				lagrangian_model_part.AddNodalSolutionStepVariable(DISPLACEMENT);
				lagrangian_model_part.AddNodalSolutionStepVariable(mUnknownVar);

				for (unsigned int i=0; i!=((mmaximum_number_of_particles*mnelems)/mfilter_factor)+mfilter_factor; i++)
				{
					Node ::Pointer pnode = lagrangian_model_part.CreateNewNode( i+mlast_node_id+1 , 0.0, 0.0, 0.0);  //recordar que es el nueevo model part!!
					//pnode->SetBufferSize(mr_model_part.NodesBegin()->GetBufferSize());
					pnode->SetBufferSize(1);
				}
				mparticle_printing_tool_initialized=true;
			}

			//resetting data of the unused particles
			const double inactive_particle_position= -10.0;
			array_1d<double,3>inactive_particle_position_vector;
			inactive_particle_position_vector(0)=inactive_particle_position;
			inactive_particle_position_vector(1)=inactive_particle_position;
			inactive_particle_position_vector(2)=inactive_particle_position;
			ModelPart::NodesContainerType::iterator inodebegin = lagrangian_model_part.NodesBegin();
			for(unsigned int ii=0; ii<lagrangian_model_part.Nodes().size(); ii++)
			{
				ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
				inode->FastGetSolutionStepValue(mUnknownVar) = 0.0;
				inode->FastGetSolutionStepValue(DISPLACEMENT) = inactive_particle_position_vector;
			}


			int counter=0;
			//ModelPart::NodesContainerType::iterator it_begin = lagrangian_model_part.NodesBegin();
			for (int i=0; i!=mmaximum_number_of_particles*mnelems; i++)
			{
				Convection_Particle& pparticle =mparticles_vector[i];
				if(pparticle.GetEraseFlag()==false && i%mfilter_factor==0)
				{
					ModelPart::NodesContainerType::iterator inode = inodebegin+counter; //copying info from the particle to the (printing) node.
					inode->FastGetSolutionStepValue(mUnknownVar) = pparticle.GetScalar1();
					inode->FastGetSolutionStepValue(DISPLACEMENT) = pparticle.Coordinates();
					counter++;
				}
			}

			KRATOS_CATCH("")

		}

	protected:


	private:


	///this function moves a particle according to the "velocity" given
	///by "rVariable". The movement is performed in nsubsteps, during a total time
	///of Dt
	void MoveParticle(  Convection_Particle & pparticle,
						 Element::Pointer & pelement,
						 GlobalPointersVector< Element >& elements_in_trajectory,
						 unsigned int & number_of_elements_in_trajectory,
						 ResultIteratorType result_begin,
						 const unsigned int MaxNumberOfResults)
	{

		ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();
		double delta_t = CurrentProcessInfo[DELTA_TIME];
		unsigned int nsubsteps;
		double substep_dt;


	    bool KEEP_INTEGRATING=false;
		bool is_found;
		//bool have_air_node;
		//bool have_water_node;

		array_1d<double,3> vel;
		array_1d<double,3> vel_without_other_phase_nodes=ZeroVector(3);
		array_1d<double,3> position;
		array_1d<double,3> mid_position;
		array_1d<double,TDim+1> N;

		//we start with the first position, then it will enter the loop.
		position = pparticle.Coordinates(); //initial coordinates


		is_found = FindNodeOnMesh(position, N ,pelement,result_begin,MaxNumberOfResults); //good, now we know where this point is:
		if(is_found == true)
		{
			KEEP_INTEGRATING=true;
			Geometry< Node >& geom = pelement->GetGeometry();//the element we're in
			vel=ZeroVector(3);

			for(unsigned int j=0; j<(TDim+1); j++)
			{
				noalias(vel) += geom[j].FastGetSolutionStepValue(mVelocityVar)*N[j];
			}

			//calculating substep to get +- courant(substep) = 0.1
			nsubsteps = 10.0 * (delta_t * pelement->GetValue(MEAN_VEL_OVER_ELEM_SIZE));
			if (nsubsteps<1)
				nsubsteps=1;
			substep_dt = delta_t / double(nsubsteps);

			position += vel*substep_dt;//weight;

			//DONE THE FIRST LOCATION OF THE PARTICLE, NOW WE PROCEED TO STREAMLINE INTEGRATION USING THE MESH SEDIMENT_VELOCITY
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			unsigned int check_from_element_number=0;


			for(unsigned int i=0; i<(nsubsteps-1); i++)// this is for the substeps n+1. in the first one we already knew the position of the particle.
			{
			  if (KEEP_INTEGRATING==true)
			  {
				  is_found = FindNodeOnMesh(position, N ,pelement,elements_in_trajectory,number_of_elements_in_trajectory,check_from_element_number,result_begin,MaxNumberOfResults); //good, now we know where this point is:
				  if(is_found == true)
				  {
					Geometry< Node >& geom = pelement->GetGeometry();//the element we're in

					vel = ZeroVector(3);
					for(unsigned int j=0; j<(TDim+1); j++)
					{
						noalias(vel) += geom[j].FastGetSolutionStepValue(mVelocityVar)*N[j];
					}


					position+=vel*substep_dt;//weight;

				  }
				  else
				  {
					  KEEP_INTEGRATING=false;
					  break;
				  }
				}
				else
					break;


			}

		}

		if (KEEP_INTEGRATING==false) (pparticle.GetEraseFlag()=true);
		else is_found = FindNodeOnMesh(position, N ,pelement,result_begin,MaxNumberOfResults); //we must save the pointer of the last element that we're in (inside the pointervector pelement)

		if (is_found==false) ( pparticle.GetEraseFlag()=true);

		 pparticle.Coordinates() = position;
	}


	void CorrectParticleUsingDeltaVariables(
						 Convection_Particle & pparticle,
						 Element::Pointer & pelement,
						 Geometry< Node >& geom)
	{
		array_1d<double,TDim+1> N;

		//we start with the first position, then it will enter the loop.
		array_1d<double,3> coords = pparticle.Coordinates();
		float & particle_scalar1 = pparticle.GetScalar1();
		//double distance=0.0;
		double delta_scalar1 = 0.0;

		bool is_found = CalculatePosition(geom,coords[0],coords[1],coords[2],N);
		if(is_found == false)
		{
			KRATOS_WATCH(N)
			for (int j=0 ; j!=(TDim+1); j++)
								if (N[j]<0.0 )
									N[j]=1e-10;
		}


		for(unsigned int j=0; j<(TDim+1); j++)
		{
			delta_scalar1 += geom[j].FastGetSolutionStepValue(DELTA_SCALAR1)*N[j];
		}
		particle_scalar1 = particle_scalar1 + delta_scalar1;
	}

	void MoveParticle_inverse_way(
						 Convection_Particle & pparticle,
						 Element::Pointer & pelement, //NOT A REFERENCE!! WE SHALL NOT OVERWRITE THE ELEMENT IT BELONGS TO!
						 ResultIteratorType result_begin,
						 const unsigned int MaxNumberOfResults)
	{

		ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();
		double delta_t = CurrentProcessInfo[DELTA_TIME];
		unsigned int nsubsteps;
		double substep_dt;


	    bool KEEP_INTEGRATING=false;
		bool is_found;

		array_1d<double,3> vel;
		array_1d<double,3> position;
		array_1d<double,3> mid_position;
		array_1d<double,TDim+1> N;
		double scalar1 = 0.0;


		//we start with the first position, then it will enter the loop.
		position = pparticle.Coordinates(); // + (pparticle)->FastGetSolutionStepValue(DISPLACEMENT); //initial coordinates


		is_found = FindNodeOnMesh(position, N ,pelement,result_begin,MaxNumberOfResults); //good, now we know where this point is:
		if(is_found == true)
		{
			KEEP_INTEGRATING=true;
			Geometry< Node >& geom = pelement->GetGeometry();//the element we're in
			vel=ZeroVector(3);
			scalar1=0.0;

			for(unsigned int j=0; j<(TDim+1); j++)
			{
				scalar1 +=  geom[j].FastGetSolutionStepValue(mUnknownVar)*N(j);
				noalias(vel) += geom[j].FastGetSolutionStepValue(mVelocityVar)*N[j];
			}
			//calculating substep to get +- courant(substep) = 1/4
			nsubsteps = 10.0 * (delta_t * pelement->GetValue(MEAN_VEL_OVER_ELEM_SIZE));
			if (nsubsteps<1)
				nsubsteps=1;
			substep_dt = delta_t / double(nsubsteps);

			position -= vel*substep_dt;//weight;

			for(unsigned int i=0; i<(nsubsteps-1); i++)// this is for the substeps n+1. in the first one we already knew the position of the particle.
			{ if (KEEP_INTEGRATING==true) {
				is_found = FindNodeOnMesh(position, N ,pelement,result_begin,MaxNumberOfResults); //good, now we know where this point is:
				if(is_found == true)
				{
					Geometry< Node >& geom = pelement->GetGeometry();//the element we're in

					vel=ZeroVector(3);
					scalar1=0.0;


					for(unsigned int j=0; j<(TDim+1); j++)
					{
						noalias(vel) += geom[j].FastGetSolutionStepValue(mVelocityVar)*N[j] ;
						scalar1 +=  geom[j].FastGetSolutionStepValue(mUnknownVar)*N(j);
					}


					position-=vel*substep_dt;//weight;
				  }
				  else KEEP_INTEGRATING=false;
				}


			}

			pparticle.GetScalar1()=scalar1;
		}
		//else {KRATOS_WATCH(position); }

	}



	///this function should find the element into which a given node is located
	///and return a pointer to the element and the vector containing the
	///shape functions that define the position within the element
	///if "false" is devolved the element is not found
	bool FindNodeOnMesh( array_1d<double,3>& position,
						 array_1d<double,TDim+1>& N,
						 Element::Pointer & pelement,
						 ResultIteratorType result_begin,
						 const unsigned int MaxNumberOfResults)
	{
		typedef std::size_t SizeType;

		const array_1d<double,3>& coords = position;
		 array_1d<double,TDim+1> aux_N;
	    //before using the bin to search for possible elements we check first the last element in which the particle was.
		Geometry<Node >& geom_default = pelement->GetGeometry(); //(*(i))->GetGeometry();
		bool is_found_1 = CalculatePosition(geom_default,coords[0],coords[1],coords[2],N);
		if(is_found_1 == true) //that was easy!
		{
			return true;
		}

		//to begin with we check the neighbour elements; it is a bit more expensive
		GlobalPointersVector< Element >& neighb_elems = pelement->GetValue(NEIGHBOUR_ELEMENTS);
		//the first we check is the one that has negative shape function, because it means it went outside in this direction:
		//commented, it is not faster than simply checking all the neighbours (branching)
		/*
		unsigned int checked_element=0;
		for (unsigned int i=0;i!=(TDim+1);i++)
		{
			if (N[i]<0.0)
			{
				checked_element=i;
				Geometry<Node >& geom = neighb_elems[i].GetGeometry();
				bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],aux_N);
				if (is_found_2)
				{
					pelement=Element::Pointer(((neighb_elems(i))));
					N=aux_N;
					return true;
				}
				break;
			}
		}
		*/
		//we check all the neighbour elements
		for (unsigned int i=0;i!=(neighb_elems.size());i++)
		{
			if(neighb_elems(i).get()!=nullptr)
			{
				Geometry<Node >& geom = neighb_elems[i].GetGeometry();
				bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],N);
				if (is_found_2)
				{
					pelement=neighb_elems(i)->shared_from_this();
					return true;
				}
			}
		}

	    //if checking all the neighbour elements did not work, we have to use the bins
		//ask to the container for the list of candidate elements
		SizeType results_found = mpBinsObjectDynamic->SearchObjectsInCell(Point{coords}, result_begin, MaxNumberOfResults );

		if(results_found>0){
		//loop over the candidate elements and check if the particle falls within
		for(SizeType i = 0; i< results_found; i++)
		{
			Geometry<Node >& geom = (*(result_begin+i))->GetGeometry();

			//find local position
			bool is_found = CalculatePosition(geom,coords[0],coords[1],coords[2],N);

			if(is_found == true)
			{
				pelement=Element::Pointer((*(result_begin+i)));
				return true;
			}
		}
	}
		//if nothing worked, then:
		//not found case
		return false;
	}


	// VERSION INCLUDING PREDEFINED ELEMENTS FOLLOWING A TRAJECTORY
		bool FindNodeOnMesh( array_1d<double,3>& position,
						 array_1d<double,TDim+1>& N,
						 Element::Pointer & pelement,
						 GlobalPointersVector< Element >& elements_in_trajectory,
						 unsigned int & number_of_elements_in_trajectory,
						 unsigned int & check_from_element_number,
						 ResultIteratorType result_begin,
						 const unsigned int MaxNumberOfResults)
	{
		typedef std::size_t SizeType;

		const array_1d<double,3>& coords = position;
		 array_1d<double,TDim+1> aux_N;
	    //before using the bin to search for possible elements we check first the last element in which the particle was.
		Geometry<Node >& geom_default = pelement->GetGeometry(); //(*(i))->GetGeometry();
		bool is_found_1 = CalculatePosition(geom_default,coords[0],coords[1],coords[2],N);
		if(is_found_1 == true)
		{
			return true; //that was easy!
		}

		//if it was not found in the first element, we can proceed to check in the following elements (in the trajectory defined by previous particles that started from the same element.
		for (unsigned int i=(check_from_element_number);i!=number_of_elements_in_trajectory;i++)
		{
			Geometry<Node >& geom = elements_in_trajectory[i].GetGeometry();
			bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],aux_N);
			if (is_found_2)
			{
				pelement=elements_in_trajectory(i)->shared_from_this();
				N=aux_N;
				check_from_element_number = i+1 ; //now i element matches pelement, so to avoid checking twice the same element we send the counter to the following element.
				return true;
			}

		}

		//now we check the neighbour elements:
		auto& neighb_elems = pelement->GetValue(NEIGHBOUR_ELEMENTS);
		//the first we check is the one that has negative shape function, because it means it went outside in this direction:
		//commented, it is not faster than simply checking all the neighbours (branching)
		/*
		unsigned int checked_element=0;
		for (unsigned int i=0;i!=(TDim+1);i++)
		{
			if (N[i]<0.0)
			{
				checked_element=i;
				Geometry<Node >& geom = neighb_elems[i].GetGeometry();
				bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],aux_N);
				if (is_found_2)
				{
					pelement=Element::Pointer(((neighb_elems(i))));
					N=aux_N;
					return true;
				}
				break;
			}
		}
		*/
		//we check all the neighbour elements
		for (unsigned int i=0;i!=(neighb_elems.size());i++)
		{
			if(neighb_elems(i).get()!=nullptr)
			{
				Geometry<Node >& geom = neighb_elems[i].GetGeometry();
				bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],N);
				if (is_found_2)
				{
					pelement=neighb_elems(i)->shared_from_this();
					if (number_of_elements_in_trajectory<20)
					{
						elements_in_trajectory(number_of_elements_in_trajectory)=pelement;
						number_of_elements_in_trajectory++;
						check_from_element_number = number_of_elements_in_trajectory;  //we do it after doing the ++ to the counter, so we woudlnt enter the loop that searches in the elements_in_trajectory list. we are the particle that is adding elements to the list
					}
					return true;
				}
			}
		}


		//if checking all the neighbour elements did not work, we have to use the bins
		//ask to the container for the list of candidate elements
		SizeType results_found = mpBinsObjectDynamic->SearchObjectsInCell(Point{coords}, result_begin, MaxNumberOfResults );

		if(results_found>0)
		{
			//loop over the candidate elements and check if the particle falls within
			for(SizeType i = 0; i< results_found; i++)
			{
				Geometry<Node >& geom = (*(result_begin+i))->GetGeometry();

				//find local position
				bool is_found = CalculatePosition(geom,coords[0],coords[1],coords[2],N);

				if(is_found == true)
				{
					pelement=Element::Pointer((*(result_begin+i)));
					if (number_of_elements_in_trajectory<20)
					{
					elements_in_trajectory(number_of_elements_in_trajectory)=pelement;
					number_of_elements_in_trajectory++;
					check_from_element_number = number_of_elements_in_trajectory;  //we do it after doing the ++ to the counter, so we woudlnt enter the loop that searches in the elements_in_trajectory list. we are the particle that is adding elements to the list
					}
					return true;
				}
			}
		}

		//not found case
		return false;
	}



	//***************************************
        //***************************************

        inline bool CalculatePosition(Geometry<Node >&geom,
                const double xc, const double yc, const double zc,
                array_1d<double, 3 > & N
                )
        {
            double x0 = geom[0].X();
            double y0 = geom[0].Y();
            double x1 = geom[1].X();
            double y1 = geom[1].Y();
            double x2 = geom[2].X();
            double y2 = geom[2].Y();

            double area = CalculateVol(x0, y0, x1, y1, x2, y2);
            double inv_area = 0.0;
            if (area == 0.0)
            {
                KRATOS_THROW_ERROR(std::logic_error, "element with zero area found", "");
            } else
            {
                inv_area = 1.0 / area;
            }


            N[0] = CalculateVol(x1, y1, x2, y2, xc, yc) * inv_area;
            N[1] = CalculateVol(x2, y2, x0, y0, xc, yc) * inv_area;
            N[2] = CalculateVol(x0, y0, x1, y1, xc, yc) * inv_area;
			//KRATOS_WATCH(N);

            if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0) //if the xc yc is inside the triangle return true
                return true;

            return false;
        }
        ////////////
        //using the pre loaded nodal coordinates
        inline bool CalculatePosition(const array_1d<double,3*(TDim+1)>& nodes_positions,
                const double xc, const double yc, const double zc,
                array_1d<double, 3 > & N
                )
        {
            const double& x0 = nodes_positions[0];
            const double& y0 = nodes_positions[1];
            const double& x1 = nodes_positions[3];
            const double& y1 = nodes_positions[4];
            const double& x2 = nodes_positions[6];
            const double& y2 = nodes_positions[7];

            double area = CalculateVol(x0, y0, x1, y1, x2, y2);
            double inv_area = 0.0;
            if (area == 0.0)
            {
                KRATOS_THROW_ERROR(std::logic_error, "element with zero area found", "");
            } else
            {
                inv_area = 1.0 / area;
            }


            N[0] = CalculateVol(x1, y1, x2, y2, xc, yc) * inv_area;
            N[1] = CalculateVol(x2, y2, x0, y0, xc, yc) * inv_area;
            N[2] = CalculateVol(x0, y0, x1, y1, xc, yc) * inv_area;
			//KRATOS_WATCH(N);

            if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0) //if the xc yc is inside the triangle return true
                return true;

            return false;
        }


	    //***************************************
        //***************************************

        inline bool CalculatePosition(Geometry<Node >&geom,
                const double xc, const double yc, const double zc,
                array_1d<double, 4 > & N
                )
        {

            double x0 = geom[0].X();
            double y0 = geom[0].Y();
            double z0 = geom[0].Z();
            double x1 = geom[1].X();
            double y1 = geom[1].Y();
            double z1 = geom[1].Z();
            double x2 = geom[2].X();
            double y2 = geom[2].Y();
            double z2 = geom[2].Z();
            double x3 = geom[3].X();
            double y3 = geom[3].Y();
            double z3 = geom[3].Z();

            double vol = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3);

            double inv_vol = 0.0;
            if (vol < 0.000000000000000000000000000001)
            {
                KRATOS_THROW_ERROR(std::logic_error, "element with zero vol found", "");
            } else
            {
                inv_vol = 1.0 / vol;
            }

            N[0] = CalculateVol(x1, y1, z1, x3, y3, z3, x2, y2, z2, xc, yc, zc) * inv_vol;
            N[1] = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, xc, yc, zc) * inv_vol;
            N[2] = CalculateVol(x3, y3, z3, x1, y1, z1, x0, y0, z0, xc, yc, zc) * inv_vol;
            N[3] = CalculateVol(x3, y3, z3, x0, y0, z0, x2, y2, z2, xc, yc, zc) * inv_vol;


            if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[3] >= 0.0 &&
                    N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0 && N[3] <= 1.0)
                //if the xc yc zc is inside the tetrahedron return true
                return true;

            return false;
        }
        ///////////////////
        //using the pre loaded nodal coordinates
		 inline bool CalculatePosition(const array_1d<double,3*(TDim+1)>& nodes_positions,
                const double xc, const double yc, const double zc,
                array_1d<double, 4 > & N
                )
        {

            const double& x0 = nodes_positions[0];
            const double& y0 = nodes_positions[1];
            const double& z0 = nodes_positions[2];
            const double& x1 = nodes_positions[3];
            const double& y1 = nodes_positions[4];
            const double& z1 = nodes_positions[5];
            const double& x2 = nodes_positions[6];
            const double& y2 = nodes_positions[7];
            const double& z2 = nodes_positions[8];
            const double& x3 = nodes_positions[9];
            const double& y3 = nodes_positions[10];
            const double& z3 = nodes_positions[11];

            double vol = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3);

            double inv_vol = 0.0;
            if (vol < 0.000000000000000000000000000001)
            {
                KRATOS_THROW_ERROR(std::logic_error, "element with zero vol found", "");
            } else
            {
                inv_vol = 1.0 / vol;
            }

            N[0] = CalculateVol(x1, y1, z1, x3, y3, z3, x2, y2, z2, xc, yc, zc) * inv_vol;
            N[1] = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, xc, yc, zc) * inv_vol;
            N[2] = CalculateVol(x3, y3, z3, x1, y1, z1, x0, y0, z0, xc, yc, zc) * inv_vol;
            N[3] = CalculateVol(x3, y3, z3, x0, y0, z0, x2, y2, z2, xc, yc, zc) * inv_vol;


            if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[3] >= 0.0 &&
                    N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0 && N[3] <= 1.0)
                //if the xc yc zc is inside the tetrahedron return true
                return true;

            return false;
        }

        inline double CalculateVol(const double x0, const double y0,
                const double x1, const double y1,
                const double x2, const double y2
                )
        {
            return 0.5 * ((x1 - x0)*(y2 - y0)- (y1 - y0)*(x2 - x0));
        }
        //***************************************
        //***************************************

        inline double CalculateVol(const double x0, const double y0, const double z0,
                const double x1, const double y1, const double z1,
                const double x2, const double y2, const double z2,
                const double x3, const double y3, const double z3
                )
        {
            double x10 = x1 - x0;
            double y10 = y1 - y0;
            double z10 = z1 - z0;

            double x20 = x2 - x0;
            double y20 = y2 - y0;
            double z20 = z2 - z0;

            double x30 = x3 - x0;
            double y30 = y3 - y0;
            double z30 = z3 - z0;

            double detJ = x10 * y20 * z30 - x10 * y30 * z20 + y10 * z20 * x30 - y10 * x20 * z30 + z10 * x20 * y30 - z10 * y20 * x30;
            return detJ * 0.1666666666666666666667;
        }



		void ComputeGaussPointPositions_4(Geometry< Node >& geom, BoundedMatrix<double, 7, 3 > & pos,BoundedMatrix<double, 7, 3 > & N)
        {
            double one_third = 1.0 / 3.0;
            double one_sixt = 0.15; //1.0 / 6.0;
            double two_third = 0.7; //2.0 * one_third;

            N(0, 0) = one_sixt;
            N(0, 1) = one_sixt;
            N(0, 2) = two_third;
            N(1, 0) = two_third;
            N(1, 1) = one_sixt;
            N(1, 2) = one_sixt;
            N(2, 0) = one_sixt;
            N(2, 1) = two_third;
            N(2, 2) = one_sixt;
            N(3, 0) = one_third;
			N(3, 1) = one_third;
			N(3, 2) = one_third;

            //first
            pos(0, 0) = one_sixt * geom[0].X() + one_sixt * geom[1].X() + two_third * geom[2].X();
            pos(0, 1) = one_sixt * geom[0].Y() + one_sixt * geom[1].Y() + two_third * geom[2].Y();
            pos(0, 2) = one_sixt * geom[0].Z() + one_sixt * geom[1].Z() + two_third * geom[2].Z();

            //second
            pos(1, 0) = two_third * geom[0].X() + one_sixt * geom[1].X() + one_sixt * geom[2].X();
            pos(1, 1) = two_third * geom[0].Y() + one_sixt * geom[1].Y() + one_sixt * geom[2].Y();
            pos(1, 2) = two_third * geom[0].Z() + one_sixt * geom[1].Z() + one_sixt * geom[2].Z();

            //third
            pos(2, 0) = one_sixt * geom[0].X() + two_third * geom[1].X() + one_sixt * geom[2].X();
            pos(2, 1) = one_sixt * geom[0].Y() + two_third * geom[1].Y() + one_sixt * geom[2].Y();
            pos(2, 2) = one_sixt * geom[0].Z() + two_third * geom[1].Z() + one_sixt * geom[2].Z();

            //fourth
			pos(3, 0) = one_third * geom[0].X() + one_third * geom[1].X() + one_third * geom[2].X();
            pos(3, 1) = one_third * geom[0].Y() + one_third * geom[1].Y() + one_third * geom[2].Y();
            pos(3, 2) = one_third * geom[0].Z() + one_third * geom[1].Z() + one_third * geom[2].Z();

        }


        void ComputeGaussPointPositionsForPostReseed(Geometry< Node >& geom, BoundedMatrix<double, 7, 3 > & pos,BoundedMatrix<double, 7, 3 > & N) //2d
        {
            double one_third = 1.0 / 3.0;
            double one_eight = 0.12; //1.0 / 6.0;
            double three_quarters = 0.76; //2.0 * one_third;

            N(0, 0) = one_eight;
            N(0, 1) = one_eight;
            N(0, 2) = three_quarters;

            N(1, 0) = three_quarters;
            N(1, 1) = one_eight;
            N(1, 2) = one_eight;

            N(2, 0) = one_eight;
            N(2, 1) = three_quarters;
            N(2, 2) = one_eight;

            N(3, 0) = one_third;
			N(3, 1) = one_third;
			N(3, 2) = one_third;

			N(4, 0) = one_eight;
            N(4, 1) = 0.44;
            N(4, 2) = 0.44;

            N(5, 0) = 0.44;
            N(5, 1) = one_eight;
            N(5, 2) = 0.44;

            N(6, 0) = 0.44;
            N(6, 1) = 0.44;
            N(6, 2) = one_eight;


            //first
            pos(0, 0) = one_eight * geom[0].X() + one_eight * geom[1].X() + three_quarters * geom[2].X();
            pos(0, 1) = one_eight * geom[0].Y() + one_eight * geom[1].Y() + three_quarters * geom[2].Y();
            pos(0, 2) = one_eight * geom[0].Z() + one_eight * geom[1].Z() + three_quarters * geom[2].Z();

            //second
            pos(1, 0) = three_quarters * geom[0].X() + one_eight * geom[1].X() + one_eight * geom[2].X();
            pos(1, 1) = three_quarters * geom[0].Y() + one_eight * geom[1].Y() + one_eight * geom[2].Y();
            pos(1, 2) = three_quarters * geom[0].Z() + one_eight * geom[1].Z() + one_eight * geom[2].Z();

            //third
            pos(2, 0) = one_eight * geom[0].X() + three_quarters * geom[1].X() + one_eight * geom[2].X();
            pos(2, 1) = one_eight * geom[0].Y() + three_quarters * geom[1].Y() + one_eight * geom[2].Y();
            pos(2, 2) = one_eight * geom[0].Z() + three_quarters * geom[1].Z() + one_eight * geom[2].Z();

            //fourth
			pos(3, 0) = one_third * geom[0].X() + one_third * geom[1].X() + one_third * geom[2].X();
            pos(3, 1) = one_third * geom[0].Y() + one_third * geom[1].Y() + one_third * geom[2].Y();
            pos(3, 2) = one_third * geom[0].Z() + one_third * geom[1].Z() + one_third * geom[2].Z();

            //fifth
            pos(4, 0) = one_eight * geom[0].X() + 0.44 * geom[1].X() + 0.44 * geom[2].X();
            pos(4, 1) = one_eight * geom[0].Y() + 0.44 * geom[1].Y() + 0.44 * geom[2].Y();
            pos(4, 2) = one_eight * geom[0].Z() + 0.44 * geom[1].Z() + 0.44 * geom[2].Z();

            //sixth
            pos(5, 0) = 0.44 * geom[0].X() + one_eight * geom[1].X() + 0.44 * geom[2].X();
            pos(5, 1) = 0.44 * geom[0].Y() + one_eight * geom[1].Y() + 0.44 * geom[2].Y();
            pos(5, 2) = 0.44 * geom[0].Z() + one_eight * geom[1].Z() + 0.44 * geom[2].Z();

            //seventh
            pos(6, 0) = 0.44 * geom[0].X() + 0.44 * geom[1].X() + one_eight * geom[2].X();
            pos(6, 1) = 0.44 * geom[0].Y() + 0.44 * geom[1].Y() + one_eight * geom[2].Y();
            pos(6, 2) = 0.44 * geom[0].Z() + 0.44 * geom[1].Z() + one_eight * geom[2].Z();




        }

        void ComputeGaussPointPositionsForPostReseed(Geometry< Node >& geom, BoundedMatrix<double, 9, 3 > & pos,BoundedMatrix<double, 9, 4 > & N) //3D
        {
            double one_quarter = 0.25;
            double small_fraction = 0.1; //1.0 / 6.0;
            double big_fraction = 0.7; //2.0 * one_third;
            double mid_fraction = 0.3; //2.0 * one_third;

            N(0, 0) = big_fraction;
            N(0, 1) = small_fraction;
            N(0, 2) = small_fraction;
            N(0, 3) = small_fraction;

            N(1, 0) = small_fraction;
            N(1, 1) = big_fraction;
            N(1, 2) = small_fraction;
            N(1, 3) = small_fraction;

            N(2, 0) = small_fraction;
            N(2, 1) = small_fraction;
            N(2, 2) = big_fraction;
            N(2, 3) = small_fraction;

            N(3, 0) = small_fraction;
            N(3, 1) = small_fraction;
            N(3, 2) = small_fraction;
            N(3, 3) = big_fraction;

			N(4, 0) = one_quarter;
			N(4, 1) = one_quarter;
			N(4, 2) = one_quarter;
			N(4, 3) = one_quarter;

			N(5, 0) = small_fraction;
            N(5, 1) = mid_fraction;
            N(5, 2) = mid_fraction;
            N(5, 3) = mid_fraction;

			N(6, 0) = mid_fraction;
            N(6, 1) = small_fraction;
            N(6, 2) = mid_fraction;
            N(6, 3) = mid_fraction;

			N(7, 0) = mid_fraction;
            N(7, 1) = mid_fraction;
            N(7, 2) = small_fraction;
            N(7, 3) = mid_fraction;

            N(8, 0) = mid_fraction;
            N(8, 1) = mid_fraction;
            N(8, 2) = mid_fraction;
            N(8, 3) = small_fraction;

			pos=ZeroMatrix(9,3);
            for (unsigned int i=0; i!=4; i++) //going through the 4 nodes
            {
				array_1d<double, 3 > & coordinates = geom[i].Coordinates();
				for (unsigned int j=0; j!=9; j++) //going through the 9 particles
				{
					for (unsigned int k=0; k!=3; k++) //x,y,z
						pos(j,k) += N(j,i) * coordinates[k];
				}
			}


        }



        void ComputeGaussPointPositionsForPreReseed(Geometry< Node >& geom, BoundedMatrix<double, 3, 3 > & pos,BoundedMatrix<double, 3, 3 > & N) //2D
        {

            N(0, 0) = 0.5;
            N(0, 1) = 0.25;
            N(0, 2) = 0.25;

            N(1, 0) = 0.25;
            N(1, 1) = 0.5;
            N(1, 2) = 0.25;

            N(2, 0) = 0.25;
            N(2, 1) = 0.25;
            N(2, 2) = 0.5;

            //first
            pos(0, 0) = 0.5 * geom[0].X() + 0.25 * geom[1].X() + 0.25 * geom[2].X();
            pos(0, 1) = 0.5 * geom[0].Y() + 0.25 * geom[1].Y() + 0.25 * geom[2].Y();
            pos(0, 2) = 0.5 * geom[0].Z() + 0.25 * geom[1].Z() + 0.25 * geom[2].Z();

            //second
            pos(1, 0) = 0.25 * geom[0].X() + 0.5 * geom[1].X() + 0.25 * geom[2].X();
            pos(1, 1) = 0.25 * geom[0].Y() + 0.5 * geom[1].Y() + 0.25 * geom[2].Y();
            pos(1, 2) = 0.25 * geom[0].Z() + 0.5 * geom[1].Z() + 0.25 * geom[2].Z();

            //third
            pos(2, 0) = 0.25 * geom[0].X() + 0.25 * geom[1].X() + 0.5 * geom[2].X();
            pos(2, 1) = 0.25 * geom[0].Y() + 0.25 * geom[1].Y() + 0.5 * geom[2].Y();
            pos(2, 2) = 0.25 * geom[0].Z() + 0.25 * geom[1].Z() + 0.5 * geom[2].Z();

        }

        void ComputeGaussPointPositionsForPreReseed(Geometry< Node >& geom, BoundedMatrix<double, 4, 3 > & pos,BoundedMatrix<double, 4, 4 > & N) //3D
        {
			//creating 4 particles, each will be closer to a node and equidistant to the other nodes


            N(0, 0) = 0.4;
            N(0, 1) = 0.2;
            N(0, 2) = 0.2;
            N(0, 3) = 0.2;

            N(1, 0) = 0.2;
            N(1, 1) = 0.4;
            N(1, 2) = 0.2;
            N(1, 3) = 0.2;

            N(2, 0) = 0.2;
            N(2, 1) = 0.2;
            N(2, 2) = 0.4;
            N(2, 3) = 0.2;

            N(3, 0) = 0.2;
            N(3, 1) = 0.2;
            N(3, 2) = 0.2;
            N(3, 3) = 0.4;

            pos=ZeroMatrix(4,3);
            for (unsigned int i=0; i!=4; i++) //going through the 4 nodes
            {
				array_1d<double, 3 > & coordinates = geom[i].Coordinates();
				for (unsigned int j=0; j!=4; j++) //going through the 4 particles
				{
					for (unsigned int k=0; k!=3; k++) //x,y,z
						pos(j,k) += N(j,i) * coordinates[k];
				}
			}

        }



		void ComputeGaussPointPositions_45(Geometry< Node >& geom, BoundedMatrix<double, 45, 3 > & pos,BoundedMatrix<double, 45, 3 > & N)
        {
			//std::cout << "NEW ELEMENT" << std::endl;
			unsigned int counter=0;
			for (unsigned int i=0; i!=9;i++)
			{
				for (unsigned int j=0; j!=(9-i);j++)
				{
					N(counter,0)=0.05+double(i)*0.1;
					N(counter,1)=0.05+double(j)*0.1;
					N(counter,2)=1.0 - ( N(counter,1)+ N(counter,0) ) ;
					pos(counter, 0) = N(counter,0) * geom[0].X() + N(counter,1) * geom[1].X() + N(counter,2) * geom[2].X();
					pos(counter, 1) = N(counter,0) * geom[0].Y() + N(counter,1) * geom[1].Y() + N(counter,2) * geom[2].Y();
					pos(counter, 2) = N(counter,0) * geom[0].Z() + N(counter,1) * geom[1].Z() + N(counter,2) * geom[2].Z();
					//std::cout << N(counter,0) << " " << N(counter,1) << " " << N(counter,2) << " " << std::endl;
					counter++;

				}
			}

        }

        void ComputeGaussPointPositions_initial(Geometry< Node >& geom, BoundedMatrix<double, 15, 3 > & pos,BoundedMatrix<double, 15, 3 > & N) //2D
        {
			//std::cout << "NEW ELEMENT" << std::endl;
			unsigned int counter=0;
			for (unsigned int i=0; i!=5;i++)
			{
				for (unsigned int j=0; j!=(5-i);j++)
				{
					N(counter,0)=0.05+double(i)*0.2;
					N(counter,1)=0.05+double(j)*0.2;
					N(counter,2)=1.0 - ( N(counter,1)+ N(counter,0) ) ;
					pos(counter, 0) = N(counter,0) * geom[0].X() + N(counter,1) * geom[1].X() + N(counter,2) * geom[2].X();
					pos(counter, 1) = N(counter,0) * geom[0].Y() + N(counter,1) * geom[1].Y() + N(counter,2) * geom[2].Y();
					pos(counter, 2) = N(counter,0) * geom[0].Z() + N(counter,1) * geom[1].Z() + N(counter,2) * geom[2].Z();
					//std::cout << N(counter,0) << " " << N(counter,1) << " " << N(counter,2) << " " << std::endl;
					counter++;

				}
			}

        }

        void ComputeGaussPointPositions_initial(Geometry< Node >& geom, BoundedMatrix<double, 20, 3 > & pos,BoundedMatrix<double, 20, 4 > & N) //3D
        {
			//std::cout << "NEW ELEMENT" << std::endl;
			//double total;
			double fraction_increment;
			unsigned int counter=0;
			for (unsigned int i=0; i!=4;i++) //going to build a particle "pyramid"(tetrahedra) by layers. the first layer will be made by a triangle of 4 base X 4 height. since it is a triangle, it means it will have 10 particles
			{
				//std::cout << "inside i" <<  i << std::endl;
				for (unsigned int j=0; j!=(4-i);j++)
				{
					//std::cout << "inside j" << j << std::endl;
					for (unsigned int k=0; k!=(4-i-j);k++)
					{
						//std::cout << "inside k" << k << std::endl;
						N(counter,0)= 0.27 * ( 0.175 + double(i) ) ; //this is our "surface" in which we will build each layer, so we must construct a triangle using what's left of the shape functions total (a total of 1)

						//total = 1.0 - N(counter,0);
						fraction_increment = 0.27; //

						N(counter,1)=fraction_increment * (0.175 + double(j));
						N(counter,2)=fraction_increment * (0.175 + double(k));
						N(counter,3)=1.0 - ( N(counter,0)+ N(counter,1) + N(counter,2) ) ;
						pos(counter, 0) = N(counter,0) * geom[0].X() + N(counter,1) * geom[1].X() + N(counter,2) * geom[2].X() + N(counter,3) * geom[3].X();
						pos(counter, 1) = N(counter,0) * geom[0].Y() + N(counter,1) * geom[1].Y() + N(counter,2) * geom[2].Y() + N(counter,3) * geom[3].Y();
						pos(counter, 2) = N(counter,0) * geom[0].Z() + N(counter,1) * geom[1].Z() + N(counter,2) * geom[2].Z() + N(counter,3) * geom[3].Z();
						//std::cout << N(counter,0) << " " << N(counter,1) << " " << N(counter,2) << " " << std::endl;
						counter++;
					}

				}
			}

        }

 		template<class T>
		bool InvertMatrix(const T& input, T& inverse)
		{
			typedef permutation_matrix<std::size_t> pmatrix;

			// create a working copy of the input
			T A(input);

			// create a permutation matrix for the LU-factorization
			pmatrix pm(A.size1());

			// perform LU-factorization
			int res = lu_factorize(A, pm);
			if (res != 0)
				return false;

			// create identity matrix of "inverse"
			inverse.assign(identity_matrix<double> (A.size1()));

			// backsubstitute to get the inverse
			lu_substitute(A, pm, inverse);

			return true;
		}

		bool InvertMatrix3x3(const BoundedMatrix<double, TDim+1 , TDim+1  >& A, BoundedMatrix<double, TDim+1 , TDim+1  >& result)
		{
			double determinant =    +A(0,0)*(A(1,1)*A(2,2)-A(2,1)*A(1,2))
                        -A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
                        +A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
			double invdet = 1/determinant;
			result(0,0) =  (A(1,1)*A(2,2)-A(2,1)*A(1,2))*invdet;
			result(1,0) = -(A(0,1)*A(2,2)-A(0,2)*A(2,1))*invdet;
			result(2,0) =  (A(0,1)*A(1,2)-A(0,2)*A(1,1))*invdet;
			result(0,1) = -(A(1,0)*A(2,2)-A(1,2)*A(2,0))*invdet;
			result(1,1) =  (A(0,0)*A(2,2)-A(0,2)*A(2,0))*invdet;
			result(2,1) = -(A(0,0)*A(1,2)-A(1,0)*A(0,2))*invdet;
			result(0,2) =  (A(1,0)*A(2,1)-A(2,0)*A(1,1))*invdet;
			result(1,2) = -(A(0,0)*A(2,1)-A(2,0)*A(0,1))*invdet;
			result(2,2) =  (A(0,0)*A(1,1)-A(1,0)*A(0,1))*invdet;

			return true;
		}

	virtual int Check()
    {
        KRATOS_TRY
        ProcessInfo& rCurrentProcessInfo = mr_model_part.GetProcessInfo();
        if (rCurrentProcessInfo.Has(CONVECTION_DIFFUSION_SETTINGS)==false)
			KRATOS_THROW_ERROR(std::logic_error, "no CONVECTION_DIFFUSION_SETTINGS in model_part", "");
        //std::cout << "ConvDiff::Check(). If crashes, check CONVECTION_DIFFUSION_SETTINGS is defined" << std::endl;

        ConvectionDiffusionSettings::Pointer my_settings = rCurrentProcessInfo.GetValue(CONVECTION_DIFFUSION_SETTINGS);

		//UNKNOWN VARIABLE
		if(my_settings->IsDefinedUnknownVariable()==true)
		{
			if (mr_model_part.NodesBegin()->SolutionStepsDataHas(my_settings->GetUnknownVariable()) == false)
				KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Unknown Variable defined but not contained in the model part", "");
		}
		else
			KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Unknown Variable not defined!", "");


		//PROJECTION VARIABLE
		//used as intermediate variable, is the variable at time n+1 but only accounting for the convective term.
		if(my_settings->IsDefinedProjectionVariable()==true)
		{
			if (mr_model_part.NodesBegin()->SolutionStepsDataHas(my_settings->GetProjectionVariable()) == false)
				KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Projection Variable defined but not contained in the model part", "");
		}
		else
			KRATOS_THROW_ERROR(std::logic_error, "No Projection variable assigned for ConvDiff!", "");


		//CONVECTION VELOCITY VARIABLE
		//CURRENTLY WE ARE USING (VELOCITY -MESH_VELOCITY) TO CONVECT, so the ConvectionVariable must not be used:
		//if(my_settings->IsDefinedConvectionVariable()==true)
		//{
		//	if (BaseType::GetModelPart().NodesBegin()->SolutionStepsDataHas(my_settings->GetConvectionVariable()) == false)
		//		KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Convection Variable defined but not contained in the model part", "");
		//}
		//else
		//	std::cout << "No Projection variable assigned for ConvDiff. Assuming Convection=0" << std::endl;
		if(my_settings->IsDefinedConvectionVariable()==true)
			KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: ConvectionVariable not used. Use VelocityVariable instead", "");

		//VELOCITY VARIABLE
		if(my_settings->IsDefinedVelocityVariable()==true)
		{
			if (mr_model_part.NodesBegin()->SolutionStepsDataHas(my_settings->GetVelocityVariable()) == false)
				KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Velocity Variable defined but not contained in the model part", "");
		}
		else
			KRATOS_THROW_ERROR(std::logic_error, "No Velocity variable assigned for ConvDiff!", "");

		if (mr_model_part.NodesBegin()->SolutionStepsDataHas(MEAN_SIZE) == false)
				KRATOS_THROW_ERROR(std::logic_error, "Add MEAN_SIZE variable to model part!", "");

		if (mr_model_part.NodesBegin()->SolutionStepsDataHas(DELTA_SCALAR1) == false)
				KRATOS_THROW_ERROR(std::logic_error, "Add DELTA_SCALAR1 variable to model part!", "");

        return 0;

        KRATOS_CATCH("")

    }




	ModelPart& mr_model_part;
	int m_nparticles;
	int mnelems;
	int moffset;
	//vector<double> mareas_vector; UNUSED SO COMMENTED
	int max_nsubsteps;
	double max_substep_dt;
	int mmaximum_number_of_particles;
	std::vector< Convection_Particle > mparticles_vector; //Point<3>
	int mlast_elem_id;
	bool modd_timestep;
	bool mparticle_printing_tool_initialized;
	unsigned int mfilter_factor;
	unsigned int mlast_node_id;
	//ModelPart& mr_particle_model_part;

	vector<int> mnumber_of_particles_in_elems;
	vector<int> mnumber_of_particles_in_elems_aux;
	//vector<ParticlePointerVector*>  mpointers_to_particle_pointers_vectors; //pointing to the GetValue of each element
    vector<ParticlePointerVector>  mvector_of_particle_pointers_vectors;

	typename BinsObjectDynamic<Configure>::Pointer  mpBinsObjectDynamic;

	const Variable<double>& mUnknownVar;
	const Variable<double>& mProjectionVar;
	const Variable<array_1d<double,3> >& mVelocityVar;
	const Variable<array_1d<double,3> >& mMeshVelocityVar;

	};

}  // namespace Kratos.

#endif // KRATOS_MOVE_PARTICLE_UTILITY_FLUID_PFEM2_TRANSPORT_INCLUDED  defined
