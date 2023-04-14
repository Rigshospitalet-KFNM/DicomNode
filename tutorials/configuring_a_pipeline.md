# Configuring the pipeline

So far we have only scratched the surface of configuration. To get the full overview, I suggest reading the `nodes.py` file, but some common configuration can be found below:

## Permanent file storage

By default all the dataset is kept in memory. That means if the program is stopped, all data is lost.
To make the pipeline save a copy to disk you need to overwrite an attribute:

```python
class MyPipeline(AbstractPipeline):
  ...
  root_data_directory: Union[str, Path] = Path('path/to/data/directory')
  ...
```

The pipeline will now save a copy of each dataset, and delete them when it's done with the dataset.
The file structure produced looks like this:

```text
  root_data_directory / {\$patient_identifier_tag} / {\$input_arg_name_1} / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                   / {\$input_arg_name_2} / Image_{\$image.modality}_{\$instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                               ... / ...
                      / {\$patient_identifier_tag} / {\$input_arg_name_1} / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                   / {\$input_arg_name_2} / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                               ... / ...
                  ... / ...
```

The `patient_identifier_tag` is another pipeline attribute, which the pipeline uses to separate images belonging to differing batches.
The tag defaults to the tag PatientID. The `input_arg_name` is a key in the input directory, the files are the images stored in the input instance.

## Logging

A pipeline creates a logger by default using python standard library, you can modify the following properties to make the logger behave like you want:

```python
class MyPipeline(AbstractPipeline):
  number_of_backups: int = 8
  "Number of backups before the os starts deleting old logs"

  log_date_format = "%Y/%m/%d %H:%M:%S"
  "String format for timestamps in logs."

  log_output: Optional[Union[TextIO, Path, str]] = stdout
  """Destination of log output:
  * `None` - Disables The logger
  * `TextIO` - output to that stream, This is stdout / stderr
  * `Path | str` - creates a rotating log at the path
  """

  log_when = "w0"
  "At what points in time the log should roll over, defaults to monday midnight"

  log_level: int = logging.INFO
  "Level of Logger"

  log_format: str = "%(asctime)s %(name)s %(levelname)s %(message)s"
  "Format of log messages using the '%' style."

  disable_pynetdicom_logger: bool = True
  "Disables pynetdicom logger"

  ...
```

The logger is injected into most sub-libraries.

## Customizing outputs

Sometimes you want to create a report supplementing an image series or you want to send data over some other form communication protocol. In that case you need to start customizing the output
