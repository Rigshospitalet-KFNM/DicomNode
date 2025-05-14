from tests.helpers.dicomnode_test_case import DicomnodeTestCase

import os
import re
import importlib
import inspect
from collections import defaultdict


class CircularImportDetector:
  """A class that detects circular imports in a Python package."""

  def __init__(self, package_name):
    """
    Initialize the detector.

    Args:
        package_name: The name of the package to check.
    """
    self.package_name = package_name
    self.dependencies = defaultdict(set)
    self.visited = set()
    self.recursion_stack = set()

  def _is_package_module(self, module_name):
    """Check if a module belongs to the package we're analyzing."""
    return module_name == self.package_name or module_name.startswith(f"{self.package_name}.")

  def _extract_imports(self, module_name):
    """Extract all imports from a module."""
    try:
        # Import the module
        module = importlib.import_module(module_name)

        # Get the source code
        source_code = inspect.getsource(module)

        # Use regex to find all import statements
        # Match 'import x' and 'from x import y' patterns
        import_pattern = re.compile(r'(?:from\s+([\w.]+)\s+import)|(?:import\s+([\w.,\s]+))|import_module\(\'[\w]+\'\)')
        matches = import_pattern.findall(source_code)
        print(matches)

        imports = set()
        for from_import, direct_import in matches:
            if from_import:
              # Handle 'from x import y'
              imports.add(from_import)
            if direct_import:
              # Handle 'import x, y, z'
              for imp in direct_import.split(','):
                # Remove 'as' aliases and get the base module name
                base_import = imp.strip().split(' as ')[0].strip()
                imports.add(base_import)

        # Filter to only include imports from the package
        return {imp for imp in imports if self._is_package_module(imp)}

    except (ImportError, AttributeError, TypeError):
        # If there's an issue importing or analyzing the module, return empty set
        return set()

  def build_dependency_graph(self):
    """
    Build a dependency graph of the package modules.
    """
    # Find all modules in the package
    package = importlib.import_module(self.package_name)
    package_path = os.path.dirname(package.__file__)

    for root, _, files in os.walk(package_path):
      for file in files:
        if file.endswith('.py') and not file.startswith('_'):
          # Convert file path to module path
          rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(package_path))
          module_path = rel_path.replace(os.sep, '.').replace('.py', '')

          # Get the full module name
          if module_path == self.package_name:
            continue  # Skip the package's __init__.py

          module_name = f"{self.package_name}.{module_path}"

          # Extract imports
          imports = self._extract_imports(module_name)

          # Add to dependency graph
          self.dependencies[module_name] = imports

  def detect_circular_imports(self):
    """
    Detect circular imports in the dependency graph.

    Returns:
        A list of cycles found in the dependency graph.
    """
    cycles = []

    def dfs(node, path=None):
      if path is None:
        path = []

      if node in self.recursion_stack:
        # Found a cycle
        cycle_start_index = path.index(node)
        cycle = path[cycle_start_index:] + [node]
        cycles.append(cycle)
        return

      if node in self.visited:
        return

      self.visited.add(node)
      self.recursion_stack.add(node)
      path.append(node)

      for dependency in self.dependencies.get(node, set()):
        if dependency in self.dependencies:  # Only follow dependencies that are actually in our graph
          dfs(dependency, path.copy())

      self.recursion_stack.remove(node)

    # Run DFS for each node
    for module in self.dependencies:
      if module not in self.visited:
        dfs(module)

    return cycles


class CircularImportsTestCase(DicomnodeTestCase):
  """Test case for detecting circular imports in a package."""

  def test_no_circular_imports(self):
    """Test that there are no circular imports in the package."""
    package_name = "dicomnode"

    # Check if the package can be imported (this should pass based on your precondition)
    try:
      importlib.import_module(package_name)
    except ImportError:
      self.fail(f"Package {package_name} could not be imported")

    # Detect circular imports
    detector = CircularImportDetector(package_name)
    detector.build_dependency_graph()
    cycles = detector.detect_circular_imports()

    # If there are cycles, format them for clear error message
    if cycles:
      cycle_messages = []
      for cycle in cycles:
        cycle_str = " -> ".join(cycle)
        cycle_messages.append(f"  {cycle_str}")

      cycle_details = "\n".join(cycle_messages)
      self.fail(f"Circular imports detected:\n{cycle_details}")
