# Configuration Guide

This Document provides an overview of the different attributes, method and properties, that you can overwrite.
Function docstring can be found in the files or sphinx documentation.

## AbstractInput

### Attributes - Input

* `required_tags: List[int]` - The list of tags that must be present in a dataset to be accepted into the input. Consider checking SOP_mapping.py for collections of Tags.
* `required_values: Dict[int, Any]` - A Mapping of tags and associated values, doesn't work for values in sequences
* `image_grinder: Grinder = identity_grinder` Function for initial preprocessing, often to transform to data format, better suited for image processing.

### Methods - Input

* `validate (self) -> bool` - Method for checking that all data is available. Should return True when input is ready for processing, false otherwise.

## Abstract Pipeline

This class is the server running the dicom node. It contains configuration for the many subclasses to instantiated.

### Attributes - Pipeline

* `processing_directory` - Base directory that the processing will take place in. The specific directory that will run is: `processing_directory/patient_ID`

#### Maintenance Configuration

* `maintenance_thread: Type[MaintenanceThread] = MaintenanceThread` - Class of MaintenanceThread to be created when the server opens
* `study_expiration_days: int = 14` - The amount of days a study will hang in memory, before being clean up by the MaintenanceThread

#### Input configuration

* `input: Dict[str, Type[AbstractInput]] = {}` - Defines the AbstractInput leafs for each Patient Node.
* `input_config: Dict[str, Dict[str, Any]] = {}` - config parsed to input, outer dict should have same keys as `input`.
* `patient_identifier_tag: int = 0x00100020 # Patient ID` - Dicom tag to separate each study
* `data_directory: Optional[Path] = None` - Path to where the pipeline tree may store dicom objects "permanently"
* `lazy_storage: bool = False` - Indicates if the abstract inputs should use Lazy datasets.
* `pipeline_tree_type: Type[PipelineTree] = PipelineTree` - Class of PipelineTree that the node will create as main data storage
* `patient_container_type: Type[PatientNode] = PatientNode` - Class of PatientNode that the the PipelineTree should create as nodes.
* `input_container_type: Type[PatientContainer] = PatientContainer` - Class of PatientContainer that the PatientNode should create when processing a patient

#### DicomGeneration

* `factory: Optional[DicomFactory] = None` - Class for producing various Dicom objects and series
* `filling_strategy: FillingStrategy = FillingStrategy.DISCARD` - Filling strategy the dicom factory should follow in the case of unspecified tags in the blueprint.
* `header_blueprint: Optional[Blueprint] = None` - Blueprint for creating a series header
* `c_move_blueprint: Optional[Blueprint] = None` - Blueprint for create a C Move object

#### AE configuration tags

* `ae_title: str = "Your_AE_TITLE"` - AE title of  the dicomnode
* `ip: str = 'localhost'` - IP of node, Either 0.0.0.0 or localhost
* `port: int = 104` - Port of Node, int in range 1-65535 (Requires root access to open port <1024)
* `supported_contexts: List[PresentationContext] = AllStoragePresentationContexts` - Presentation contexts accepted by the node
* `require_called_aet: bool = True` - Require caller to specify AE title of node
* `require_calling_aet: List[str] = []` - If not empty require the node only to accept connection from AE titles in this attribute
* `known_endpoints: Dict[str, Address]` - Address book indexed by AE titles.
* `_associations_responds_addresses: Dict[int, Address] = {}` - Internal variable containing a mapping of association to endpoint address
* `association_container_factory: Type[AssociationContainerFactory] = AssociationContainerFactory` - Class of Factory, that extracts information from the association to the underlying processing function.
* `default_response_port: int = 104` - Default Port used for unspecified Dicomnodes

#### Logging Configuration

* `backup_weeks: int = 8` - Backup of log are made weekly, this specifies how many weeks of logs is saved
* `log_date_format: str` - String format for timestamps in logs.
* `log_level: int = logging.INFO` - Level of Logger
* `log_format: str = "%(asctime)s %(name)s %(levelname)s %(message)s"` - Log format as per pythons logging module.
* `disable_pynetdicom_logger: bool = True` - Disables pynetdicom logger
* `log_output Optional[Union[TextIO, Path, str]]` - Destination of log output:
  * `None` - Disables The logger
  * `TextIO` - output to that stream, This is stdout / stderr
  * `Path | str` - creates a rotating log at the path

#### Handler directories

* `_acceptation_handlers: Dict[AssociationTypes, Callable[[Self, AcceptedContainer], None]]` - Dictionary containing handler functions for different types of association called when a association is accepted
* `_release_handlers: Dict[AssociationTypes, Callable[[Self, ReleasedContainer], None]]` - Dictionary containing handler functions for different types of association called when a association is released

### Methods - Pipeline

These methods are intended to be overwritten by a user with minimal knowledge of the library. They should contain no side effects effecting the rest of the library.

* `Filter(self, dataset: Dataset) -> bool` - initial Function for filtering datasets received by c stores
* `process(self, input_container: InputContainer): -> PipelineOutput` - Post processing function
* `post_init(self, start)` - Function called after most init, but before the node opens, used to setup other inputs.
* `open(self,blocking=True)` - Opens the server. If overwritten super should be called at the end user function.
* `closed(self)` - Closes the server. If overwritten super should be called at the end user function.

#### Private Methods

These methods are not intended to be overwritten by a user as they require intimate knowledge of the library. They should contain side effects effecting the rest of the library.

* `_consume_association_accept_store_association(self, accepted_container: AcceptedContainer)` -> Handler function that performs required side effects when a store association is established.
* `_consume_c_store_container(self, c_store_container: CStoreContainer) -> int` -> Updates the pipeline with respect C Store request
* `_consume_association_release_store_association` -> Handler function that performs side effects related to the releasing of a store association
* `_dispatch(self, output: PipelineOutput): -> bool` - Dispatching output to their destination and handles errors of dispatching
* `_get_input_container(self, patient_ID: str, released_container: ReleasedContainer) -> InputContainer` - Extracts 
