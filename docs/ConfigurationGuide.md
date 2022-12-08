# Configuration Guide

This Document provides an overview of the different attributes, method and properties, that you can overwrite.

## AbstractInput

### Attributes

* `required_tags`: `List[int]` - The list of tags that must be present in a dataset to be accepted into the input.
* `required_values` : `Dict[int, Any]` - A "List" of tags and associated values. If you need to check values in sequences, you should overwrite the add_image method, and do the check. It's highly recommend that you call the super method if you do this. 

### Methods

* `validate (self) -> bool` - Method for checking that all data is available. 