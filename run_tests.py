#!/usr/bin/env python3
"""Test runner script for TensorLens."""
import subprocess
import sys
import os

os.chdir('/var/lib/alan-portfolio/workspaces/ryan/tensor-lens-ryan-vo')

# Install backend dependencies
print("Installing backend dependencies...")
result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[dev]'], 
                       capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Run backend tests
print("\n\nRunning backend tests...")
result = subprocess.run([sys.executable, '-m', 'pytest', 'backend/tests', '-v'], 
                       capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Install frontend dependencies
print("\n\nInstalling frontend dependencies...")
os.chdir('/var/lib/alan-portfolio/workspaces/ryan/tensor-lens-ryan-vo/frontend')
result = subprocess.run(['npm', 'install'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Run frontend tests
print("\n\nRunning frontend tests...")
result = subprocess.run(['npm', 'test', '--', '--run'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Build frontend
print("\n\nBuilding frontend...")
result = subprocess.run(['npm', 'run', 'build'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

