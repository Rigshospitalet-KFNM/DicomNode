from typing import Optional, Tuple
import re

class PrivateTagParserReadException(Exception):
  pass


def read_private_tag(line) -> Optional[Tuple[int, Tuple[str,str,str,str,str]]]:
  line.replace("\t", " ")
  line.replace("\r", " ")
  line.replace("\n", "")
  line = line.strip()

  tokens = list(filter(lambda token: token, line.split(" ")))
  # filter Tokens
  token_list = []
  for token in tokens:
    if token[0] == '#':
      break
    token_list.append(token)

  if len(token_list) == 0: # The line was a comment or empty
    return None

  print(token_list)



  tag = 0x0000
  return tag, ("","","","","")