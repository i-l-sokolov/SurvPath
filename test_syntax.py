"""
Simple syntax checker for modified files - no dependencies required
"""

import sys
import py_compile

print("=" * 80)
print("Syntax Validation for Protein Integration")
print("=" * 80)

files_to_check = [
    'datasets/dataset_survival.py',
    'models/model_SurvPath.py',
    'models/layers/cross_attention.py',
    'utils/core_utils.py',
]

all_passed = True

for file_path in files_to_check:
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"✓ {file_path:50s} - Syntax OK")
    except py_compile.PyCompileError as e:
        print(f"✗ {file_path:50s} - Syntax ERROR")
        print(f"  {e}")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✓ All files have valid Python syntax!")
    print("\nModifications summary:")
    print("  1. datasets/dataset_survival.py - Added protein loading")
    print("  2. models/model_SurvPath.py - Added 3-modality fusion")
    print("  3. models/layers/cross_attention.py - Added 3-way attention")
    print("  4. utils/core_utils.py - Added protein data unpacking")
    print("\nThe code is syntactically correct and ready for testing with PyTorch!")
else:
    print("✗ Some files have syntax errors")
    sys.exit(1)
print("=" * 80)
