#!/usr/bin/env python3
"""Test runner script for TensorLens."""
import subprocess
import sys
import os

os.chdir('/var/lib/alan-portfolio/workspaces/ryan/tensor-lens-ryan-vo')

# Run backend tests
print("Running backend tests...")
result = subprocess.run([sys.executable, '-m', 'pytest', 'backend/tests', '-v'], 
                       capture_output=True, text=True, cwd='/var/lib/alan-portfolio/workspaces/ryan/tensor-lens-ryan-vo')
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)

</arg_key>
<arg_key=i</arg_key>
<arg_value>Writing test runner script
