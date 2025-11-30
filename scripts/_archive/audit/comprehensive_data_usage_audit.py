#!/usr/bin/env python3
"""
Comprehensive Data Usage Audit

Audits the entire codebase to ensure:
1. Both 2025 current season AND historical data are used correctly
2. No hardcoded defaults (everything calculated from data)
3. Proper data source prioritization (2025 first, historical fallback)
4. Consistent data loading patterns throughout framework
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set
import json

class DataUsageAuditor:
    """Audit data usage patterns across the codebase."""

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent.parent.parent
        self.issues = []
        self.findings = {
            'hardcoded_defaults': [],
            'missing_historical_data': [],
            'wrong_data_sources': [],
            'inconsistent_loading': [],
            'missing_integration': []
        }

    def audit_file(self, file_path: Path) -> Dict:
        """Audit a single Python file for data usage issues."""
        try:
            with open(file_path) as f:
                content = f.read()
        except Exception as e:
            return {'error': str(e)}

        issues = []

        # Check for hardcoded defaults
        hardcoded_patterns = [
            r'0\.05\s*#.*default',
            r'0\.10\s*#.*default',
            r'0\.08\s*#.*default',
            r'trailing_target_share\s*=\s*0\.',
            r'trailing_carry_share\s*=\s*0\.',
            r'weeks\s*\*\s*40',  # Hardcoded team targets
            r'weeks\s*\*\s*25',  # Hardcoded team carries
        ]

        for pattern in hardcoded_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    'type': 'hardcoded_default',
                    'line': line_num,
                    'pattern': pattern,
                    'match': match.group()
                })

        # Check for missing historical data usage
        if 'load_2025' in content or 'stats_week.*2025' in content:
            if 'load.*2024' not in content and 'historical' not in content.lower():
                issues.append({
                    'type': 'missing_historical_data',
                    'message': 'Uses 2025 data but no historical data fallback'
                })

        # Check for wrong data source priority
        if 'nflverse.*is_available' in content:
            # Should try Sleeper first for 2025
            if 'season.*2025' in content and 'sleeper' not in content.lower():
                issues.append({
                    'type': 'wrong_data_sources',
                    'message': '2025 season should prioritize Sleeper over NFLverse'
                })

        return {'issues': issues, 'file': str(file_path)}

    def audit_codebase(self) -> Dict:
        """Audit entire codebase."""
        scripts_dir = self.root_dir / 'scripts'
        nfl_quant_dir = self.root_dir / 'nfl_quant'

        all_files = []
        for directory in [scripts_dir, nfl_quant_dir]:
            if directory.exists():
                all_files.extend(directory.rglob('*.py'))

        results = {
            'files_audited': 0,
            'issues_found': 0,
            'by_category': {
                'hardcoded_defaults': [],
                'missing_historical_data': [],
                'wrong_data_sources': [],
                'inconsistent_loading': []
            }
        }

        for file_path in all_files:
            if '__pycache__' in str(file_path):
                continue

            audit_result = self.audit_file(file_path)
            if 'issues' in audit_result:
                results['files_audited'] += 1
                if audit_result['issues']:
                    results['issues_found'] += len(audit_result['issues'])
                    for issue in audit_result['issues']:
                        issue['file'] = audit_result['file']
                        results['by_category'][issue['type']].append(issue)

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate human-readable audit report."""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE DATA USAGE AUDIT REPORT")
        report.append("="*80)
        report.append("")
        report.append(f"Files Audited: {results['files_audited']}")
        report.append(f"Issues Found: {results['issues_found']}")
        report.append("")

        for category, issues in results['by_category'].items():
            if issues:
                report.append(f"\n{category.upper().replace('_', ' ')}: {len(issues)} issues")
                report.append("-" * 80)
                for issue in issues[:10]:  # Show first 10
                    report.append(f"  {issue['file']}:{issue.get('line', '?')}")
                    if 'message' in issue:
                        report.append(f"    {issue['message']}")
                    elif 'match' in issue:
                        report.append(f"    Found: {issue['match']}")

        return "\n".join(report)


def main():
    """Run comprehensive audit."""
    auditor = DataUsageAuditor()
    results = auditor.audit_codebase()
    report = auditor.generate_report(results)

    print(report)

    # Save to file
    output_file = Path('reports/DATA_USAGE_AUDIT_REPORT.md')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"# Data Usage Audit Report\n\n")
        f.write(f"Generated: {Path(__file__).stat().st_mtime}\n\n")
        f.write(report)

    print(f"\n✅ Audit complete. Report saved to {output_file}")


if __name__ == '__main__':
    main()
