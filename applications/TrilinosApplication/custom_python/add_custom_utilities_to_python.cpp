//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Riccardo Rossi
//

// System includes

// External includes

// Project includes
#include "linear_solvers/linear_solver.h"
#include "spaces/ublas_space.h"

// Application includes
#include "trilinos_space.h"
#include "trilinos_space_experimental.h"
#include "custom_python/add_custom_utilities_to_python.h"
#include "custom_python/add_trilinos_convergence_accelerators_to_python.h"
#include "custom_python/trilinos_pointer_wrapper.h"
#include "custom_utilities/trilinos_assembling_utilities.h"
#include "custom_utilities/trilinos_cutting_app.h"
#include "custom_utilities/trilinos_cutting_iso_app.h"
#include "custom_utilities/trilinos_refine_mesh.h"
#include "custom_utilities/trilinos_partitioned_fsi_utilities.h"

namespace Kratos::Python
{
namespace py = pybind11;

typedef UblasSpace<double, Matrix, Vector> TrilinosLocalSpaceType;
typedef TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector> TrilinosSparseSpaceType;
#ifdef HAVE_TPETRA
typedef TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::Vector<>> TrilinosExperimentalSparseSpaceType;
#endif

template <class TSparseSpace, class TValueType, unsigned int TDim>
void AuxiliarUpdateInterfaceValues(
    TrilinosPartitionedFSIUtilities<TSparseSpace, TValueType, TDim> &dummy,
    ModelPart &rModelPart,
    const Variable<TValueType> &rSolutionVariable,
    AuxiliaryVectorWrapper &rCorrectedGuess)
{
    dummy.UpdateInterfaceValues(
        rModelPart,
        rSolutionVariable,
        rCorrectedGuess.GetReference());
}

template <class TSparseSpace, class TValueType, unsigned int TDim>
void AuxiliarComputeInterfaceResidualVector(
    TrilinosPartitionedFSIUtilities<TSparseSpace, TValueType, TDim> &dummy,
    ModelPart &rInterfaceModelPart,
    const Variable<TValueType> &rOriginalVariable,
    const Variable<TValueType> &rModifiedVariable,
    const Variable<TValueType> &rResidualVariable,
    AuxiliaryVectorWrapper &rInterfaceResidual,
    const std::string ResidualType = "nodal",
    const Variable<double> &rResidualNormVariable = FSI_INTERFACE_RESIDUAL_NORM)
{
    dummy.ComputeInterfaceResidualVector(
        rInterfaceModelPart,
        rOriginalVariable,
        rModifiedVariable,
        rResidualVariable,
        rInterfaceResidual.GetReference(),
        ResidualType,
        rResidualNormVariable);
}

template<class TSparseSpace>
void AddCustomUtilitiesToPythonTemplate(pybind11::module& m, std::string NameSuffix)
{
    using CuttingAppType = TrilinosCuttingApplication<TSparseSpace>;
    py::class_<CuttingAppType>(m,("TrilinosCuttingApplication" + NameSuffix).c_str())
        .def(py::init<typename TSparseSpace::CommunicatorType& >() )
        .def("FindSmallestEdge", &CuttingAppType::FindSmallestEdge )
        .def("GenerateCut", &CuttingAppType::GenerateCut )
        .def("AddSkinConditions", &CuttingAppType::AddSkinConditions )
        .def("AddVariablesToCutModelPart", &CuttingAppType::AddVariablesToCutModelPart )
        .def("UpdateCutData", &CuttingAppType::UpdateCutData )
        ;

    using CuttingIsoAppType = TrilinosCuttingIsosurfaceApplication<TSparseSpace>;
    py::class_<CuttingIsoAppType>(m,("TrilinosCuttingIsosurfaceApplication" + NameSuffix).c_str())
        .def(py::init<typename TSparseSpace::CommunicatorType& >() )
        .def("GenerateScalarVarCut", &CuttingIsoAppType::template GenerateVariableCut<double>)
        .def("AddSkinConditions", &CuttingIsoAppType::AddSkinConditions)
        .def("UpdateCutData", &CuttingIsoAppType::UpdateCutData)
        .def("DeleteCutData", &CuttingIsoAppType::DeleteCutData)
        ;

    using RefineMeshType = TrilinosRefineMesh<TSparseSpace>;
    py::class_<RefineMeshType>(m,("TrilinosRefineMesh" + NameSuffix).c_str())
        .def(py::init<ModelPart& , typename TSparseSpace::CommunicatorType& >() )
        .def("Local_Refine_Mesh", &RefineMeshType::Local_Refine_Mesh )
        .def("PrintDebugInfo", &RefineMeshType::PrintDebugInfo )
        ;

    typedef PartitionedFSIUtilities<TSparseSpace, double, 2> BasePartitionedFSIUtilitiesDouble2DType;
    typedef PartitionedFSIUtilities<TSparseSpace, double, 3> BasePartitionedFSIUtilitiesDouble3DType;
    typedef PartitionedFSIUtilities<TSparseSpace, array_1d<double,3>, 2> BasePartitionedFSIUtilitiesArray2DType;
    typedef PartitionedFSIUtilities<TSparseSpace, array_1d<double,3>, 3> BasePartitionedFSIUtilitiesArray3DType;

    py::class_<BasePartitionedFSIUtilitiesDouble2DType, typename BasePartitionedFSIUtilitiesDouble2DType::Pointer>(m, ("PartitionedFSIUtilitiesDouble2D" + NameSuffix).c_str())
        .def("CreateCouplingSkin", &BasePartitionedFSIUtilitiesDouble2DType::CreateCouplingSkin)
        .def("InitializeInterfaceVector", [](BasePartitionedFSIUtilitiesDouble2DType& rSelf, const ModelPart& rInterfaceModelPart, const Variable<double> &rOriginVariable, AuxiliaryVectorWrapper &rInterfaceVector){
            rSelf.InitializeInterfaceVector(rInterfaceModelPart, rOriginVariable, rInterfaceVector.GetReference());})
        ;
    py::class_<BasePartitionedFSIUtilitiesDouble3DType, typename BasePartitionedFSIUtilitiesDouble3DType::Pointer>(m, ("PartitionedFSIUtilitiesDouble3D" + NameSuffix).c_str())
        .def("CreateCouplingSkin", &BasePartitionedFSIUtilitiesDouble3DType::CreateCouplingSkin)
        .def("InitializeInterfaceVector", [](BasePartitionedFSIUtilitiesDouble3DType& rSelf, const ModelPart& rInterfaceModelPart, const Variable<double> &rOriginVariable, AuxiliaryVectorWrapper &rInterfaceVector){
            rSelf.InitializeInterfaceVector(rInterfaceModelPart, rOriginVariable, rInterfaceVector.GetReference());})
        ;
    py::class_<BasePartitionedFSIUtilitiesArray2DType, typename BasePartitionedFSIUtilitiesArray2DType::Pointer>(m, ("PartitionedFSIUtilitiesArray2D" + NameSuffix).c_str())
        .def("CreateCouplingSkin", &BasePartitionedFSIUtilitiesArray2DType::CreateCouplingSkin)
        .def("InitializeInterfaceVector", [](BasePartitionedFSIUtilitiesArray2DType& rSelf, const ModelPart& rInterfaceModelPart, const Variable<array_1d<double,3>> &rOriginVariable, AuxiliaryVectorWrapper &rInterfaceVector){
            rSelf.InitializeInterfaceVector(rInterfaceModelPart, rOriginVariable, rInterfaceVector.GetReference());})
        ;
    py::class_<BasePartitionedFSIUtilitiesArray3DType, typename BasePartitionedFSIUtilitiesArray3DType::Pointer>(m, ("PartitionedFSIUtilitiesArray3D" + NameSuffix).c_str())
        .def("CreateCouplingSkin", &BasePartitionedFSIUtilitiesArray3DType::CreateCouplingSkin)
        .def("InitializeInterfaceVector", [](BasePartitionedFSIUtilitiesArray3DType& rSelf, const ModelPart& rInterfaceModelPart, const Variable<array_1d<double,3>> &rOriginVariable, AuxiliaryVectorWrapper &rInterfaceVector){
            rSelf.InitializeInterfaceVector(rInterfaceModelPart, rOriginVariable, rInterfaceVector.GetReference());})
        ;

    typedef TrilinosPartitionedFSIUtilities<TSparseSpace, double, 2> TrilinosPartitionedFSIUtilitiesDouble2DType;
    typedef TrilinosPartitionedFSIUtilities<TSparseSpace, double, 3> TrilinosPartitionedFSIUtilitiesDouble3DType;
    typedef TrilinosPartitionedFSIUtilities<TSparseSpace, array_1d<double,3>, 2> TrilinosPartitionedFSIUtilitiesArray2DType;
    typedef TrilinosPartitionedFSIUtilities<TSparseSpace, array_1d<double,3>, 3> TrilinosPartitionedFSIUtilitiesArray3DType;

    py::class_<TrilinosPartitionedFSIUtilitiesDouble2DType, typename TrilinosPartitionedFSIUtilitiesDouble2DType::Pointer, BasePartitionedFSIUtilitiesDouble2DType>(m, ("TrilinosPartitionedFSIUtilitiesDouble2D" + NameSuffix).c_str())
        .def(py::init<const typename TSparseSpace::CommunicatorType &>())
        .def("GetInterfaceArea", &TrilinosPartitionedFSIUtilitiesDouble2DType::GetInterfaceArea)
        .def("GetInterfaceResidualSize", &TrilinosPartitionedFSIUtilitiesDouble2DType::GetInterfaceResidualSize)
        .def("SetUpInterfaceVector", [](TrilinosPartitionedFSIUtilitiesDouble2DType& self, ModelPart& rModelPart){
            return AuxiliaryVectorWrapper(self.SetUpInterfaceVector(rModelPart));})
        .def("UpdateInterfaceValues", &AuxiliarUpdateInterfaceValues<TSparseSpace, double,2>)
        .def("ComputeInterfaceResidualNorm", &TrilinosPartitionedFSIUtilitiesDouble2DType::ComputeInterfaceResidualNorm)
        .def("ComputeInterfaceResidualVector", &AuxiliarComputeInterfaceResidualVector<TSparseSpace, double,2>)
        .def("ComputeAndPrintFluidInterfaceNorms", &TrilinosPartitionedFSIUtilitiesDouble2DType::ComputeAndPrintFluidInterfaceNorms)
        .def("ComputeAndPrintStructureInterfaceNorms", &TrilinosPartitionedFSIUtilitiesDouble2DType::ComputeAndPrintStructureInterfaceNorms)
        .def("CheckCurrentCoordinatesFluid", &TrilinosPartitionedFSIUtilitiesDouble2DType::CheckCurrentCoordinatesFluid)
        .def("CheckCurrentCoordinatesStructure", &TrilinosPartitionedFSIUtilitiesDouble2DType::CheckCurrentCoordinatesStructure);

    py::class_<TrilinosPartitionedFSIUtilitiesArray2DType, typename TrilinosPartitionedFSIUtilitiesArray2DType::Pointer, BasePartitionedFSIUtilitiesArray2DType>(m, ("TrilinosPartitionedFSIUtilitiesArray2D" + NameSuffix).c_str())
        .def(py::init<const typename TSparseSpace::CommunicatorType &>())
        .def("GetInterfaceArea", &TrilinosPartitionedFSIUtilitiesArray2DType::GetInterfaceArea)
        .def("GetInterfaceResidualSize", &TrilinosPartitionedFSIUtilitiesArray2DType::GetInterfaceResidualSize)
        .def("SetUpInterfaceVector", [](TrilinosPartitionedFSIUtilitiesArray2DType& self, ModelPart& rModelPart){
            return AuxiliaryVectorWrapper(self.SetUpInterfaceVector(rModelPart));})
        .def("UpdateInterfaceValues", &AuxiliarUpdateInterfaceValues<TSparseSpace, array_1d<double,3>,2>)
        .def("ComputeInterfaceResidualNorm", &TrilinosPartitionedFSIUtilitiesArray2DType::ComputeInterfaceResidualNorm)
        .def("ComputeInterfaceResidualVector", &AuxiliarComputeInterfaceResidualVector<TSparseSpace, array_1d<double,3>,2>)
        .def("ComputeAndPrintFluidInterfaceNorms", &TrilinosPartitionedFSIUtilitiesArray2DType::ComputeAndPrintFluidInterfaceNorms)
        .def("ComputeAndPrintStructureInterfaceNorms", &TrilinosPartitionedFSIUtilitiesArray2DType::ComputeAndPrintStructureInterfaceNorms)
        .def("CheckCurrentCoordinatesFluid", &TrilinosPartitionedFSIUtilitiesArray2DType::CheckCurrentCoordinatesFluid)
        .def("CheckCurrentCoordinatesStructure", &TrilinosPartitionedFSIUtilitiesArray2DType::CheckCurrentCoordinatesStructure);

    py::class_<TrilinosPartitionedFSIUtilitiesDouble3DType, typename TrilinosPartitionedFSIUtilitiesDouble3DType::Pointer, BasePartitionedFSIUtilitiesDouble3DType>(m, ("TrilinosPartitionedFSIUtilitiesDouble3D" + NameSuffix).c_str())
        .def(py::init<const typename TSparseSpace::CommunicatorType &>())
        .def("GetInterfaceArea", &TrilinosPartitionedFSIUtilitiesDouble3DType::GetInterfaceArea)
        .def("GetInterfaceResidualSize", &TrilinosPartitionedFSIUtilitiesDouble3DType::GetInterfaceResidualSize)
        .def("SetUpInterfaceVector", [](TrilinosPartitionedFSIUtilitiesDouble3DType& self, ModelPart& rModelPart){
            return AuxiliaryVectorWrapper(self.SetUpInterfaceVector(rModelPart));})
        .def("UpdateInterfaceValues", &AuxiliarUpdateInterfaceValues<TSparseSpace, double,3>)
        .def("ComputeInterfaceResidualNorm", &TrilinosPartitionedFSIUtilitiesDouble3DType::ComputeInterfaceResidualNorm)
        .def("ComputeInterfaceResidualVector", &AuxiliarComputeInterfaceResidualVector<TSparseSpace, double,3>)
        .def("ComputeAndPrintFluidInterfaceNorms", &TrilinosPartitionedFSIUtilitiesDouble3DType::ComputeAndPrintFluidInterfaceNorms)
        .def("ComputeAndPrintStructureInterfaceNorms", &TrilinosPartitionedFSIUtilitiesDouble3DType::ComputeAndPrintStructureInterfaceNorms)
        .def("CheckCurrentCoordinatesFluid", &TrilinosPartitionedFSIUtilitiesDouble3DType::CheckCurrentCoordinatesFluid)
        .def("CheckCurrentCoordinatesStructure", &TrilinosPartitionedFSIUtilitiesDouble3DType::CheckCurrentCoordinatesStructure);

    py::class_<TrilinosPartitionedFSIUtilitiesArray3DType, typename TrilinosPartitionedFSIUtilitiesArray3DType::Pointer, BasePartitionedFSIUtilitiesArray3DType>(m, ("TrilinosPartitionedFSIUtilitiesArray3D" + NameSuffix).c_str())
        .def(py::init<const typename TSparseSpace::CommunicatorType &>())
        .def("GetInterfaceArea", &TrilinosPartitionedFSIUtilitiesArray3DType::GetInterfaceArea)
        .def("GetInterfaceResidualSize", &TrilinosPartitionedFSIUtilitiesArray3DType::GetInterfaceResidualSize)
        .def("SetUpInterfaceVector", [](TrilinosPartitionedFSIUtilitiesArray3DType& self, ModelPart& rModelPart){
            return AuxiliaryVectorWrapper(self.SetUpInterfaceVector(rModelPart));})
        .def("UpdateInterfaceValues", &AuxiliarUpdateInterfaceValues<TSparseSpace, array_1d<double,3>,3>)
        .def("ComputeInterfaceResidualNorm", &TrilinosPartitionedFSIUtilitiesArray3DType::ComputeInterfaceResidualNorm)
        .def("ComputeInterfaceResidualVector", &AuxiliarComputeInterfaceResidualVector<TSparseSpace, array_1d<double,3>,3>)
        .def("ComputeAndPrintFluidInterfaceNorms", &TrilinosPartitionedFSIUtilitiesArray3DType::ComputeAndPrintFluidInterfaceNorms)
        .def("ComputeAndPrintStructureInterfaceNorms", &TrilinosPartitionedFSIUtilitiesArray3DType::ComputeAndPrintStructureInterfaceNorms)
        .def("CheckCurrentCoordinatesFluid", &TrilinosPartitionedFSIUtilitiesArray3DType::CheckCurrentCoordinatesFluid)
        .def("CheckCurrentCoordinatesStructure", &TrilinosPartitionedFSIUtilitiesArray3DType::CheckCurrentCoordinatesStructure);
}

void  AddCustomUtilitiesToPython(pybind11::module& m)
{
    AddCustomUtilitiesToPythonTemplate<TrilinosSparseSpaceType>(m, "");
#ifdef HAVE_TPETRA
    AddCustomUtilitiesToPythonTemplate<TrilinosExperimentalSparseSpaceType>(m, "Experimental");
#endif
}

} // Namespace Kratos::Python.
