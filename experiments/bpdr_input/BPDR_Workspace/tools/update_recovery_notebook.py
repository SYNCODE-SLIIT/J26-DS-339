"""Compatibility entry point for the complete notebook source builder."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('build_complete_recovery_notebook.py')), run_name='__main__')
