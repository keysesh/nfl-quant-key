#!/usr/bin/env python3
"""
Comprehensive Audit: Find All Hardcoded Values

This script searches the entire codebase for hardcoded assumptions, defaults, and fallback values.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import ast

def find_hardcoded_values(file_path: Path) -> List[Dict]:
    """Find hardcoded values in a Python file."""
    findings = []
    
    try:
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception as e:
        return findings
    
    # Patterns to look for
    suspicious_patterns = [
        # Common hardcoded percentages/rates
        (r'\b0\.7[0-9]\b', 'Catch rate / percentage'),
        (r'\b0\.6[0-9]\b', 'Snap share / percentage'),
        (r'\b0\.5[0-9]\b', 'Default percentage'),
        (r'\b0\.[1-4][0-9]\b', 'Small percentage/rate'),
        
        # Common hardcoded defaults
        (r'\b28\.0\b', 'Default team total'),
        (r'\b24\.0\b', 'Default opponent total'),
        (r'\b35\.0\b', 'Default pass attempts'),
        (r'\b25\.0\b', 'Default rush attempts'),
        (r'\b2\.0\b', 'Default targets/receptions'),
        (r'\b10\.0\b', 'Default carries'),
        
        # Hardcoded multipliers
        (r'\* 0\.[0-9]+\b', 'Multiplier'),
        (r'\* 1\.[0-9]+\b', 'Multiplier'),
        (r'/ 0\.[0-9]+\b', 'Divisor'),
    ]
    
    for lineno, line in enumerate(content.split('\n'), 1):
        # Skip comments and docstrings
        if line.strip().startswith('#') or '"""' in line or "'''" in line:
            continue
        
        for pattern, description in suspicious_patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                # Check if it's in a string literal (probably OK)
                if '"' in line[:match.start()] or "'" in line[:match.start()]:
                    continue
                
                findings.append({
                    'file': str(file_path),
                    'line': lineno,
                    'code': line.strip(),
                    'value': match.group(),
                    'description': description,
                })
    
    return findings


def find_default_parameters(file_path: Path) -> List[Dict]:
    """Find default parameter values that might be hardcoded assumptions."""
    findings = []
    
    try:
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception:
        return findings
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.defaults:
                if isinstance(arg, ast.Constant):
                    value = arg.value
                    if isinstance(value, (int, float)):
                        # Check if it's a suspicious default
                        if (isinstance(value, float) and 0 < value < 1) or value in [2.0, 10.0, 24.0, 28.0, 35.0]:
                            findings.append({
                                'file': str(file_path),
                                'function': node.name,
                                'default_value': value,
                                'type': 'function_default',
                            })


def main():
    """Run comprehensive audit."""
    project_root = Path('.')
    
    # Directories to search
    search_dirs = [
        'nfl_quant',
        'scripts',
    ]
    
    all_findings = {
        'hardcoded_values': [],
        'default_parameters': [],
    }
    
    print("="*80)
    print("COMPREHENSIVE HARDCODED VALUES AUDIT")
    print("="*80)
    print()
    
    for search_dir in search_dirs:
        dir_path = project_root / search_dir
        if not dir_path.exists():
            continue
        
        print(f"Searching {search_dir}/...")
        
        for py_file in dir_path.rglob('*.py'):
            # Skip __pycache__ and venv
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
            
            # Find hardcoded values
            findings = find_hardcoded_values(py_file)
            all_findings['hardcoded_values'].extend(findings)
            
            # Find default parameters
            defaults = find_default_parameters(py_file)
            if defaults:
                all_findings['default_parameters'].extend(defaults)
    
    # Report findings
    print(f"\n📊 Audit Results:")
    print(f"   Hardcoded values found: {len(all_findings['hardcoded_values'])}")
    print(f"   Default parameters found: {len(all_findings['default_parameters'])}")
    
    # Group by file
    by_file = {}
    for finding in all_findings['hardcoded_values']:
        file = finding['file']
        if file not in by_file:
            by_file[file] = []
        by_file[file].append(finding)
    
    print(f"\n📁 Files with hardcoded values:")
    for file, findings in sorted(by_file.items()):
        print(f"   {file}: {len(findings)} instances")
    
    # Show top 20 findings
    print(f"\n🔍 Top 20 Hardcoded Values:")
    for i, finding in enumerate(all_findings['hardcoded_values'][:20], 1):
        print(f"   {i}. {finding['file']}:{finding['line']}")
        print(f"      {finding['description']}: {finding['value']}")
        print(f"      {finding['code'][:80]}")
        print()
    
    # Save detailed report
    report_file = Path('reports/HARDCODED_VALUES_AUDIT.txt')
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE HARDCODED VALUES AUDIT\n")
        f.write("="*80 + "\n\n")
        
        for file, findings in sorted(by_file.items()):
            f.write(f"\n{file}:\n")
            f.write("-" * 80 + "\n")
            for finding in findings:
                f.write(f"  Line {finding['line']}: {finding['description']} = {finding['value']}\n")
                f.write(f"    {finding['code']}\n")
    
    print(f"✅ Detailed report saved to {report_file}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review findings in {report_file}")
    print(f"   2. Replace hardcoded values with data-driven calculations")
    print(f"   3. Use actual historical/current season data")


if __name__ == '__main__':
    main()

