//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ \.
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Pooyan Dadvand
//                   Riccardo Rossi
//  Collaborator:    Vicente Mataix Ferrandiz
//

#pragma once

// System includes
#include <filesystem>
#include <fstream>

// External includes

// Project includes
#include "includes/io.h"

namespace Kratos
{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/// An IO class for reading and writing a modelpart
/** This class reads and writes all modelpart data including the meshes.
*/
class KRATOS_API(KRATOS_CORE) ModelPartIO 
    : public IO
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of ModelPartIO
    KRATOS_CLASS_POINTER_DEFINITION(ModelPartIO);

    /// Alias for the base IO type.
    using BaseType = IO;

    /// Alias for the node type.
    using NodeType = BaseType::NodeType;

    /// Alias for the mesh type.
    using MeshType = BaseType::MeshType;

    /// Alias for the nodes container type.
    using NodesContainerType = BaseType::NodesContainerType;

    /// Alias for the properties container type.
    using PropertiesContainerType = BaseType::PropertiesContainerType;

    /// Alias for the elements container type.
    using ElementsContainerType = BaseType::ElementsContainerType;

    /// Alias for the conditions container type.
    using ConditionsContainerType = BaseType::ConditionsContainerType;

    /// Alias for the connectivities container type.
    using ConnectivitiesContainerType = BaseType::ConnectivitiesContainerType;

    /// Alias for the output files container type.
    using OutputFilesContainerType = std::vector<std::ostream*>;

    /// Alias for the size type.
    using SizeType = std::size_t;

    /// Prevents this class from hiding IO::WriteProperties(Properties)
    using BaseType::WriteProperties;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Constructor with filename.
    ModelPartIO(
        std::filesystem::path const& Filename,
        const Flags Options = IO::READ | IO::IGNORE_VARIABLES_ERROR.AsFalse() | IO::SKIP_TIMER);

    /// Constructor with stream.
    ModelPartIO(
        Kratos::shared_ptr<std::iostream> Stream,
        const Flags Options = IO::IGNORE_VARIABLES_ERROR.AsFalse() | IO::SKIP_TIMER);

    /// Constructor with filenames.
    // ModelPartIO(std::string const& InputFilename, std::string const& OutputFilename)
    //     : mNumberOfLines(0), mInput(std::ifstream(InputFilename.c_str())), mOutput(std::ofstream(OutputFilename.c_str()))
    // {
    // }

    /// Destructor.
    ~ModelPartIO() override;

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief This method reads one node
     * @param rThisNode The node to be read
     */
    bool ReadNode(NodeType& rThisNode) override;

    /**
     * @brief This method reads the nodes from an array of nodes
     * @param rThisNodes The array of nodes to be read
     */
    bool ReadNodes(NodesContainerType& rThisNodes) override;

    /**
     * @brief This method reads the number of nodes
     * @return The number of nodes
     */
    std::size_t ReadNodesNumber() override;

    /**
     * @brief This method writes the nodes from an array of nodes
     * @param rThisNodes The array of nodes to be written
     */
    void WriteNodes(NodesContainerType const& rThisNodes) override;

    /**
     * @brief This method reads one Properties
     * @param rThisProperties The Properties to be read
     */
    void ReadProperties(Properties& rThisProperties) override;

    /**
     * @brief This method reads the Properties from an array of Properties
     * @param rThisProperties The array of Properties to be read
     */
    void ReadProperties(PropertiesContainerType& rThisProperties) override;

    /**
     * @brief This method writes one Properties
     * @param rThisProperties The Properties to be written
     */
    void WriteProperties(PropertiesContainerType const& rThisProperties) override;

    /**
     * @brief This method reads one geometry
     * @param rThisNodes The nodes constituying the geometry
     * @param pThisGeometries The pointer to the geometry
     */
    void ReadGeometry(
        NodesContainerType& rThisNodes,
        GeometryType::Pointer& pThisGeometry
        ) override;

    /**
     * @brief This method reads an array of geometries
     * @param rThisNodes The nodes constituying the geometry
     * @param rThisGeometry The array of geometries
     */
    void ReadGeometries(
        NodesContainerType& rThisNodes,
        GeometryContainerType& rThisGeometries
        ) override;

    /**
     * @brief This method reads the geometries connectivities
     * @param rGeometriesConnectivities The geometries connectivities
     * @return The number of geometries
     */
    std::size_t ReadGeometriesConnectivities(ConnectivitiesContainerType& rGeometriesConnectivities) override;

    /**
     * @brief This method writes an array of geometries
     * @param rThisGeometries The array of geometries to be written
     */
    void WriteGeometries(GeometryContainerType const& rThisGeometries) override;

    /**
     * @brief This method reads one element
     * @param rThisNodes The nodes constituying the element
     * @param rThisProperties The Properties of the element
     * @param pThisElements The pointer to the element
     */
    void ReadElement(
        NodesContainerType& rThisNodes,
        PropertiesContainerType& rThisProperties,
        Element::Pointer& pThisElement
        ) override;

    /**
     * @brief This method reads an array of elements
     * @param rThisNodes The nodes constituying the element
     * @param rThisProperties The Properties of the element
     * @param rThisElement The array of elements
     */
    void ReadElements(
        NodesContainerType& rThisNodes,
        PropertiesContainerType& rThisProperties,
        ElementsContainerType& rThisElements
        ) override;

    /**
     * @brief This method reads the elements connectivities
     * @param rElementsConnectivities The elements connectivities
     * @return The number of elements
     */
    std::size_t ReadElementsConnectivities(ConnectivitiesContainerType& rElementsConnectivities) override;

    /**
     * @brief This method writes an array of elements
     * @param rThisElements The array of elements to be written
     */
    void WriteElements(ElementsContainerType const& rThisElements) override;

    /**
     * @brief This method reads an array of conditions
     * @param rThisNodes The nodes constituying the condition
     * @param rThisProperties The Properties of the condition
     * @param rThisConditions The array of conditions
     */
    void ReadConditions(
        NodesContainerType& rThisNodes,
        PropertiesContainerType& rThisProperties,
        ConditionsContainerType& rThisConditions
        ) override;

    /**
     * @brief This method reads the conditions connectivities
     * @param rConditionsConnectivities The conditions connectivities
     * @return The number of conditions
     */
    std::size_t ReadConditionsConnectivities(ConnectivitiesContainerType& rConditionsConnectivities) override;

    /**
     * @brief This method writes an array of conditions
     * @param rThisConditions The array of conditions to be written
     */
    void WriteConditions(ConditionsContainerType const& rThisConditions) override;

    /**
     * @brief This method reads the initial values of the model part
     * @param rThisModelPart The model part with the initial values to be read
     */
    void ReadInitialValues(ModelPart& rThisModelPart) override;

    /**
     * @brief This method reads the mesh
     * @param rThisMesh The mesh to be read
     */
    void ReadMesh(MeshType & rThisMesh) override;

    /**
     * @brief This method writes the mesh
     * @param rThisMesh The mesh to be written
     */
    void WriteMesh(MeshType & rThisMesh) override;

    /**
     * @brief This method reads the model part
     * @param rThisModelPart The model part to be read
     */
    void ReadModelPart(ModelPart & rThisModelPart) override;

    /**
     * @brief This method writes the model part
     * @param rThisModelPart The model part to be written
     */
    void WriteModelPart(ModelPart & rThisModelPart) override;

    /**
     * @brief Read the input file and create the nodal connectivities graph, stored in CSR format.
     * @details This function produces input for Metis' nodal graph partitioning algorithms.
     * The nodal graph is stored as a (compressed) matrix where index (i,j) is non-zero if
     * there is an edge in the mesh joining nodes i and j (note that nodes are numbered from zero here,
     * to make integration with Metis simpler).
     * After call, will point to C array of size NumNodes+1 containing the first CSR array: entries related to node k are stored between positions (*NodeIndices)[k] and (*NodeIndices)[k+1] of *NodeConnectivities.
     * @param rAuxConnectivities After call, will point to a C array of size (*NodeIndices)[NumNodes].
     * entries between (*NodeIndices)[k] and (*NodeIndices)[k+1] are a list of all nodes connected
     * to node k (counting from 0).
     * @return Number of nodes.
     */
    std::size_t ReadNodalGraph(ConnectivitiesContainerType& rAuxConnectivities) override;

    /**
     * @brief This method divides a model part into partitions
     * @param NumberOfPartitions The number of partitions
     * @param rPartitioningInfo Information about partitioning of entities
     */
    void DivideInputToPartitions(SizeType NumberOfPartitions,
                                 const PartitioningInfo& rPartitioningInfo) override;

    /**
     * @brief This method divides a model part into partitions
     * @param pStreams The stream pointer
     * @param NumberOfPartitions The number of partitions
     * @param rPartitioningInfo Information about partitioning of entities
     */
    void DivideInputToPartitions(Kratos::shared_ptr<std::iostream> * pStreams,
                                SizeType NumberOfPartitions,
                                const PartitioningInfo& rPartitioningInfo) override;

    void SwapStreamSource(Kratos::shared_ptr<std::iostream> newStream);

    /**
     * @brief Virtual method to read element and condition IDs from a sub-model part.
     * @details This method is intended to be overridden by derived classes.
     * @param rModelPartName The name of the sub-model part to read from.
     * @param rElementsIds Set to store element IDs.
     * @param rConditionsIds Set to store condition IDs.
     */
    void ReadSubModelPartElementsAndConditionsIds(
        std::string const& rModelPartName,
        std::unordered_set<SizeType>& rElementsIds,
        std::unordered_set<SizeType>& rConditionsIds
        ) override;

    /**
     * @brief Virtual method to read nodal graph from entities list.
     * @details This method is intended to be overridden by derived classes.
     * @param rAuxConnectivities Container of connectivities.
     * @param rElementsIds Set of element IDs.
     * @param rConditionsIds Set of condition IDs.
     * @return The size of the nodal graph.
     */
    std::size_t ReadNodalGraphFromEntitiesList(
        ConnectivitiesContainerType& rAuxConnectivities,
        std::unordered_set<SizeType>& rElementsIds,
        std::unordered_set<SizeType>& rConditionsIds
        ) override;

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override
    {
        return "ModelPartIO";
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "ModelPartIO";
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const override
    {
    }

    ///@}
    ///@name Friends
    ///@{

    ///@}
protected:
    ///@name Protected static Member Variables
    ///@{

    ///@}
    ///@name Protected member Variables
    ///@{

    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    /**
     * @brief Returns the reordered node ID.
     * @param NodeId The original node ID.
     * @return The reordered node ID (same as input in this base class).
     */
    virtual ModelPartIO::SizeType ReorderedNodeId(ModelPartIO::SizeType NodeId);

    /**
     * @brief Returns the reordered geometry ID.
     * @param GeometryId The original geometry ID.
     * @return The reordered geometry ID (same as input in this base class).
     */
    virtual ModelPartIO::SizeType ReorderedGeometryId(ModelPartIO::SizeType GeometryId);

    /**
     * @brief Returns the reordered element ID.
     * @param ElementId The original element ID.
     * @return The reordered element ID (same as input in this base class).
     */
    virtual ModelPartIO::SizeType ReorderedElementId(ModelPartIO::SizeType ElementId);

    /**
     * @brief Returns the reordered condition ID.
     * @param ConditionId The original condition ID.
     * @return The reordered condition ID (same as input in this base class).
     */
    virtual ModelPartIO::SizeType ReorderedConditionId(ModelPartIO::SizeType ConditionId);

    ///@}
    ///@name Protected  Access
    ///@{

    ///@}
    ///@name Protected Inquiry
    ///@{

    ///@}
    ///@name Protected LifeCycle
    ///@{

    ///@}
private:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Member Variables
    ///@{

    SizeType mNumberOfLines;

    std::filesystem::path mBaseFilename;
    Flags mOptions;

    Kratos::shared_ptr<std::iostream> mpStream;

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

    std::string& ReadBlockName(std::string& rBlockName);

    void SkipBlock(std::string const& BlockName);

    bool CheckEndBlock(std::string const& BlockName, std::string& rWord);

    void ReadModelPartDataBlock(ModelPart& rModelPart, const bool IsSubmodelpart=false);

    void WriteModelPartDataBlock(ModelPart& rModelPart, const bool IsSubmodelpart=false);

    template<class TablesContainerType>
    void ReadTableBlock(TablesContainerType& rTables);

    void ReadTableBlock(ModelPart::TablesContainerType& rTables);

    template<class TablesContainerType>
    void WriteTableBlock(TablesContainerType& rTables);

    void WriteTableBlock(ModelPart::TablesContainerType& rTables);

    void ReadNodesBlock(NodesContainerType& rThisNodes);

    void ReadNodesBlock(ModelPart& rModelPart);

    std::size_t CountNodesInBlock();

    void ReadPropertiesBlock(PropertiesContainerType& rThisProperties);

    void ReadGeometriesBlock(ModelPart& rModelPart);

    void ReadGeometriesBlock(NodesContainerType& rThisNodes, GeometryContainerType& rThisGeometries);

    void ReadElementsBlock(ModelPart& rModelPart);

    void ReadElementsBlock(NodesContainerType& rThisNodes, PropertiesContainerType& rThisProperties, ElementsContainerType& rThisElements);


    void ReadConditionsBlock(ModelPart& rModelPart);

    void ReadConditionsBlock(NodesContainerType& rThisNodes, PropertiesContainerType& rThisProperties, ConditionsContainerType& rThisConditions);


    void ReadNodalDataBlock(ModelPart& rThisModelPart);

    void WriteNodalDataBlock(ModelPart& rThisModelPart);

    template<class TVariableType>
    void ReadNodalDofVariableData(NodesContainerType& rThisNodes, const TVariableType& rVariable);


    void ReadNodalFlags(NodesContainerType& rThisNodes, Flags const& rFlags);

    template<class TVariableType>
    void ReadNodalScalarVariableData(NodesContainerType& rThisNodes, const TVariableType& rVariable);



    template<class TVariableType, class TDataType>
    void ReadNodalVectorialVariableData(NodesContainerType& rThisNodes, const TVariableType& rVariable, TDataType Dummy);

    void ReadElementalDataBlock(ElementsContainerType& rThisElements);
    template<class TObjectsContainerType>
    void WriteDataBlock(const TObjectsContainerType& rThisObjectContainer, const std::string& rObjectName);
    template<class TVariableType, class TObjectsContainerType>
    void WriteDataBlock(const TObjectsContainerType& rThisObjectContainer,const VariableData* rVariable, const std::string& rObjectName);

    template<class TVariableType>
    void ReadElementalScalarVariableData(ElementsContainerType& rThisElements, const TVariableType& rVariable);


    template<class TVariableType, class TDataType>
    void ReadElementalVectorialVariableData(ElementsContainerType& rThisElements, const TVariableType& rVariable, TDataType Dummy);
    void ReadConditionalDataBlock(ConditionsContainerType& rThisConditions);

    template<class TVariableType>
    void ReadConditionalScalarVariableData(ConditionsContainerType& rThisConditions, const TVariableType& rVariable);


    template<class TVariableType, class TDataType>
    void ReadConditionalVectorialVariableData(ConditionsContainerType& rThisConditions, const TVariableType& rVariable, TDataType Dummy);

    SizeType ReadGeometriesConnectivitiesBlock(ConnectivitiesContainerType& rThisConnectivities);

    SizeType ReadElementsConnectivitiesBlock(ConnectivitiesContainerType& rThisConnectivities);

    SizeType ReadConditionsConnectivitiesBlock(ConnectivitiesContainerType& rThisConnectivities);

    void FillNodalConnectivitiesFromGeometryBlock(ConnectivitiesContainerType& rNodalConnectivities);

    void FillNodalConnectivitiesFromElementBlock(ConnectivitiesContainerType& rNodalConnectivities);

    void FillNodalConnectivitiesFromConditionBlock(ConnectivitiesContainerType& rNodalConnectivities);

    void FillNodalConnectivitiesFromGeometryBlockInList(
        ConnectivitiesContainerType& rNodalConnectivities,
        std::unordered_set<SizeType>& rGeometriesIds);

    void FillNodalConnectivitiesFromElementBlockInList(
        ConnectivitiesContainerType& rNodalConnectivities,
        std::unordered_set<SizeType>& rElementsIds);

    void FillNodalConnectivitiesFromConditionBlockInList(
        ConnectivitiesContainerType& rNodalConnectivities,
        std::unordered_set<SizeType>& rConditionsIds);

    void ReadCommunicatorDataBlock(Communicator& rThisCommunicator, NodesContainerType& rThisNodes);

    void ReadCommunicatorLocalNodesBlock(Communicator& rThisCommunicator, NodesContainerType& rThisNodes);


    void ReadCommunicatorGhostNodesBlock(Communicator& rThisCommunicator, NodesContainerType& rThisNodes);

    void ReadMeshBlock(ModelPart& rModelPart);

    void WriteMeshBlock(ModelPart& rModelPart);


    void ReadMeshDataBlock(MeshType& rMesh);


    void ReadMeshNodesBlock(ModelPart& rModelPart, MeshType& rMesh);

    void ReadMeshElementsBlock(ModelPart& rModelPart, MeshType& rMesh);

    void ReadMeshConditionsBlock(ModelPart& rModelPart, MeshType& rMesh);

    void ReadMeshPropertiesBlock(ModelPart& rModelPart, MeshType& rMesh);

    void ReadSubModelPartBlock(ModelPart& rMainModelPart, ModelPart& rParentModelPart);

    void WriteSubModelPartBlock(ModelPart& rMainModelPart, const std::string& InitialTabulation);

    void ReadSubModelPartDataBlock(ModelPart& rModelPart);

    void ReadSubModelPartTablesBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

    void ReadSubModelPartPropertiesBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

    /**
     * @brief Reads the nodes block of a submodelpart.
     * @param rMainModelPart Reference to the main model part.
     * @param rSubModelPart Reference to the submodelpart where the nodes will be added.
     */
    void ReadSubModelPartNodesBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

    /**
     * @brief Reads the elements block of a submodelpart.
     * @param rMainModelPart Reference to the main model part.
     * @param rSubModelPart Reference to the submodelpart where the elements will be added.
     */
    void ReadSubModelPartElementsBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

    /**
     * @brief Reads the conditions block of a submodelpart.
     * @param rMainModelPart Reference to the main model part.
     * @param rSubModelPart Reference to the submodelpart where the conditions will be added.
     */
    void ReadSubModelPartConditionsBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

    /**
     * @brief Reads the geometries block of a submodelpart.
     * @param rMainModelPart Reference to the main model part.
     * @param rSubModelPart Reference to the submodelpart where the geometries will be added.
     */
    void ReadSubModelPartGeometriesBlock(
        ModelPart &rMainModelPart,
        ModelPart &rSubModelPart);

    void DivideInputToPartitionsImpl(
        OutputFilesContainerType& rOutputFiles,
        SizeType NumberOfPartitions,
        const PartitioningInfo& rPartitioningInfo);

    void DivideModelPartDataBlock(OutputFilesContainerType& OutputFiles);

    void DivideTableBlock(OutputFilesContainerType& OutputFiles);

    void DividePropertiesBlock(OutputFilesContainerType& OutputFiles);

    void DivideNodesBlock(OutputFilesContainerType& OutputFiles,
                          PartitionIndicesContainerType const& NodesAllPartitions);

    void DivideGeometriesBlock(OutputFilesContainerType& OutputFiles,
                             PartitionIndicesContainerType const& GeometriesAllPartitions);

    void DivideElementsBlock(OutputFilesContainerType& OutputFiles,
                             PartitionIndicesContainerType const& ElementsAllPartitions);



    void DivideConditionsBlock(OutputFilesContainerType& OutputFiles,
                               PartitionIndicesContainerType const& ConditionsAllPartitions);


    void DivideNodalDataBlock(OutputFilesContainerType& OutputFiles,
                              PartitionIndicesContainerType const& NodesAllPartitions);

    void DivideFlagVariableData(OutputFilesContainerType& OutputFiles,
                               PartitionIndicesContainerType const& NodesAllPartitions);

    void DivideDofVariableData(OutputFilesContainerType& OutputFiles,
                               PartitionIndicesContainerType const& NodesAllPartitions);

    template<class TValueType>
    void DivideVectorialVariableData(OutputFilesContainerType& OutputFiles,
                                     PartitionIndicesContainerType const& EntitiesPartitions,
                                     std::string BlockName);


    void DivideElementalDataBlock(OutputFilesContainerType& OutputFiles,
                                  PartitionIndicesContainerType const& ElementsAllPartitions);

    void DivideScalarVariableData(OutputFilesContainerType& OutputFiles,
                                  PartitionIndicesContainerType const& EntitiesPartitions,
                                  std::string BlockName);


    void DivideConditionalDataBlock(OutputFilesContainerType& OutputFiles,
                                    PartitionIndicesContainerType const& ConditionsAllPartitions);


    void DivideMeshBlock(OutputFilesContainerType& OutputFiles,
                                         PartitionIndicesContainerType const& NodesAllPartitions,
                                         PartitionIndicesContainerType const& ElementsAllPartitions,
                                         PartitionIndicesContainerType const& ConditionsAllPartitions);

	void DivideSubModelPartBlock(OutputFilesContainerType& OutputFiles,
		PartitionIndicesContainerType const& NodesAllPartitions,
		PartitionIndicesContainerType const& ElementsAllPartitions,
		PartitionIndicesContainerType const& ConditionsAllPartitions);

    void DivideMeshDataBlock(OutputFilesContainerType& OutputFiles);


    void DivideMeshNodesBlock(OutputFilesContainerType& OutputFiles,
                                         PartitionIndicesContainerType const& NodesAllPartitions);


    void DivideMeshElementsBlock(OutputFilesContainerType& OutputFiles,
                                         PartitionIndicesContainerType const& ElementsAllPartitions);

    void DivideMeshConditionsBlock(OutputFilesContainerType& OutputFiles,
                                         PartitionIndicesContainerType const& ConditionsAllPartitions);


	void DivideSubModelPartDataBlock(OutputFilesContainerType& OutputFiles);

	void DivideSubModelPartTableBlock(OutputFilesContainerType& OutputFiles);


	void DivideSubModelPartNodesBlock(OutputFilesContainerType& OutputFiles,
		PartitionIndicesContainerType const& NodesAllPartitions);


	void DivideSubModelPartElementsBlock(OutputFilesContainerType& OutputFiles,
		PartitionIndicesContainerType const& ElementsAllPartitions);

	void DivideSubModelPartConditionsBlock(OutputFilesContainerType& OutputFiles,
		PartitionIndicesContainerType const& ConditionsAllPartitions);

	void WritePartitionIndices(OutputFilesContainerType& OutputFiles, PartitionIndicesType const&  NodesPartitions, PartitionIndicesContainerType const& NodesAllPartitions);


    void WriteCommunicatorData(OutputFilesContainerType& OutputFiles, SizeType NumberOfPartitions, GraphType const& DomainsColoredGraph,
                               PartitionIndicesType const& NodesPartitions,
                               PartitionIndicesType const& ElementsPartitions,
                               PartitionIndicesType const& ConditionsPartitions,
                               PartitionIndicesContainerType const& NodesAllPartitions,
                               PartitionIndicesContainerType const& ElementsAllPartitions,
                               PartitionIndicesContainerType const& ConditionsAllPartitions);

    void WriteCommunicatorLocalNodes(OutputFilesContainerType& OutputFiles, SizeType NumberOfPartitions, PartitionIndicesType const& NodesPartitions, PartitionIndicesContainerType const& NodesAllPartitions);

    void WriteInAllFiles(OutputFilesContainerType& OutputFiles, std::string const& ThisWord);


    template<class TContainerType, class TKeyType>
    typename TContainerType::iterator FindKey(TContainerType& ThisContainer , TKeyType ThisKey, std::string ComponentName);




    // Basically it starts to read the character sequence until reaching a
    // "(" and then goes until corresponding ")" which means the vector or
    // matrix value is completely read. It can be used to read any kind of
    // vector or matrix with operator >> defined and writtern in following
    // format for a vector: [size] ( value1, value2,...., valueN )
    // format for a matrix: [size1,size2] ( )( )...( ) //look props read
    template<class TValueType>
    TValueType& ReadVectorialValue(TValueType& rValue);

    template<class TValueType>
    TValueType& ExtractValue(std::string rWord, TValueType & rValue);

    bool& ExtractValue(std::string rWord, bool & rValue);

    void ReadConstitutiveLawValue(ConstitutiveLaw::Pointer& rValue);

    ModelPartIO& ReadWord(std::string& Word);

    ModelPartIO& ReadBlock(std::string& Block, std::string const& BlockName);

    char SkipWhiteSpaces();

    char GetCharacter();

    void CheckStatement(std::string const& rStatement, std::string const& rGivenWord) const;

    void ResetInput();

    inline void CreatePartition(unsigned int NumberOfThreads,const int NumberOfRows, DenseVector<unsigned int>& partitions);

    /// Iterate over a Node block, calling ReorderedNodeId on each node.
    /** This method allows derived implementations to initialize reordering
     *  without storing the nodes.
     */
    void ScanNodeBlock();

    ///@}
    ///@name Private  Access
    ///@{


    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{

    ///@}

}; // Class ModelPartIO

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


//   /// input stream function
//   inline std::istream& operator >> (std::istream& rIStream,
//                  ModelPartIO& rThis);

//   /// output stream function
//   inline std::ostream& operator << (std::ostream& rOStream,
//                  const ModelPartIO& rThis)
//     {
//       rThis.PrintInfo(rOStream);
//       rOStream << std::endl;
//       rThis.PrintData(rOStream);

//       return rOStream;
//     }
//   ///@}


}  // namespace Kratos.
