# Configuration Guide

This Document provides an overview of the different attributes, method and properties, that you can overwrite.
Function docstring can be found in the files or sphinx documentation.

## AbstractInput

### Attributes - Input

* `required_tags`: `List[int]` - The list of tags that must be present in a dataset to be accepted into the input. Consider checking SOP_mapping.py for collections of Tags.
* `required_values` : `Dict[int, Any]` - A "List" of tags and associated values. If you need to check values in sequences, you should overwrite the add_image method, and do the check. It's highly recommend that you call the super method if you do this.
* `image_grinder : Callable[[Iterator[Dataset]],Any] = identity_grinder` Function for initial preprocessing, often to transform to data format, better suited for image processing.

### Methods - Input

* `validate (self) -> bool` - Method for checking that all data is available. Should return True when input is ready for processing, false otherwise.

## Abstract Pipeline

### Attributes - Pipeline

#### Input configuration

* `input: Dict[str, Type[AbstractInput]] = {}` - This defined the input for your process function.
* `input_config: Dict[str, Dict[str, Any]] = {}` - config parsed to input, outer dict should have same keys as `input`.
* `patient_identifier_tag: int = 0x00100020 # Patient ID` - Dicom tag to separate data
* `root_data_directory: Optional[Path] = None` - Path to where the pipeline tree may store dicom objects "permanently"
* `pipelineTreeType: Type[PipelineTree] = PipelineTree` - PipelineTree for creating input containers
* `inputContainerType: Type[PatientContainer] = PatientContainer` - Class that will be instantiated in Pipeline tree and passed to process
* `lazy: bool = False` - Determined the abstract inputs should use Lazy datasets.

#### DicomGeneration

* `factory: Optional[DicomFactory] = None` - Class for producing various Dicom objects and series
* `header_blueprint: Optional[Blueprint] = None` - Blueprint for creating a series header
* `c_move_blueprint: Optional[Blueprint] = None` - Blueprint for create a C Move object

#### AE configuration tags

* `ae_title: str = "Your_AE_TITLE"` - AE title of node
* `ip: str = 'localhost'` - IP of node
* `port: int = 104` - Port of Node
* `supported_contexts: List[PresentationContext] = AllStoragePresentationContexts` - Accepted Presentation contexts by the node-
* `require_called_aet: bool = True` - Require caller to specify AE title of node
* `require_calling_aet: List[str] = []` - Only accept connection from these AE titles

#### Logging Configuration

* `backup_weeks: int = 8` - Backup of log are made weekly, this specifies how many weeks of logs is saved
* `log_path: Optional[Union[str, Path]] = None` - Path to log file, if `None` then log is outputted to stdout
* `log_level: int = logging.INFO` - Level of Log file
* `log_format: str = "%(asctime)s %(name)s %(levelname)s %(message)s"` - Log format as per pythons logging module.
* `disable_pynetdicom_logger: bool = True` - Disables pynetdicom logger

### Methods - Pipeline

* `Filter(self, dataset: Dataset) -> bool` - initial Function for filtering datasets received by c stores
* `process(self, input_container: InputContainer): -> PipelineOutput` - Post processing function
* `post_init(self, start)` - Function called after most init, but before the node opens, used to setup other inputs.
