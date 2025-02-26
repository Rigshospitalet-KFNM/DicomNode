from typing import List, Optional


from dicomnode.lib import config_parser

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class ConfigTestCase(DicomnodeTestCase):
  def test_typecast_int(self):
    casted_config = config_parser._typecast_config(
      {"key" : "1"}, { "key" : int}
    )

    self.assertIn('key', casted_config)
    self.assertEqual(casted_config['key'], 1)

  def test_typecast_list_of_int(self):
    casted_config = config_parser._typecast_config(
      {"key" : "[1,3,3,4,4]"},
      {"key" : List[int] }
    )

    self.assertIn('key', casted_config)
    self.assertListEqual(casted_config['key'],
                         [1,3,3,4,4])

  def test_typecast_list_of_int_with_spaces(self):
    casted_config = config_parser._typecast_config(
      {"key" : "[1 , 3 ,3 ,4 , 4]"},
      {"key" : List[int] }
    )

    self.assertIn('key', casted_config)
    self.assertListEqual(casted_config['key'],
                         [1,3,3,4,4])

  def test_optional_typecast_to_none(self):
    casted_config = config_parser._typecast_config(
      {"key" : None},
      {"key" : Optional[str] }
    )

    self.assertIn('key', casted_config)
    self.assertIsNone(casted_config['key'])

  def test_optional_typecast_to_none_str(self):
    casted_config = config_parser._typecast_config(
      {"key" : "None"},
      {"key" : Optional[str] }
    )

    self.assertIn('key', casted_config)
    self.assertIsNone(casted_config['key'])