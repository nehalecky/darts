#!/usr/bin/env python3
"""
Test notebook 28 execution with Air Passengers dataset only.
Executes cells 0-18 (through Air Passengers Performance Summary).
"""
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def test_air_passengers_section():
    """Execute notebook 28 through Air Passengers section only."""
    notebook_path = "examples/28-Foundation-Models-Tutorial.ipynb"

    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    print(f"Total cells in notebook: {len(nb.cells)}")

    # Only execute cells 0-18 (Air Passengers section)
    air_cells = nb.cells[:19]  # 0-18 inclusive
    print(f"Testing Air Passengers section: {len(air_cells)} cells")

    # Create temporary notebook with only Air Passengers cells
    test_nb = nbformat.v4.new_notebook()
    test_nb.cells = air_cells
    test_nb.metadata = nb.metadata

    # Execute
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        print("\nExecuting Air Passengers section...")
        ep.preprocess(test_nb, {'metadata': {'path': 'examples/'}})
        print("\n✅ Air Passengers section executed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        # Print last few cells for debugging
        for i, cell in enumerate(test_nb.cells[-3:]):
            if 'outputs' in cell:
                print(f"\n--- Cell {len(test_nb.cells)-3+i} outputs ---")
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'error':
                        print(f"ERROR: {output.get('ename')}: {output.get('evalue')}")
        return False

if __name__ == '__main__':
    success = test_air_passengers_section()
    sys.exit(0 if success else 1)
