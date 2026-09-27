from __future__ import annotations
import argparse
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT.parents[1]
DELTA_PATH = ROOT / 'artifacts' / 'submission5_recovery_delta.tsv'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT / 'FINAL_THIRD_SUBMISSION' / 'matching_results.tsv')
    parser.add_argument('--output', type=Path, default=PACKAGE_ROOT / 'output' / 'matching_results.tsv')
    args = parser.parse_args()
    if not args.baseline.exists():
        raise FileNotFoundError(f'Baseline matching file not found: {args.baseline}')
    if not DELTA_PATH.exists():
        raise FileNotFoundError(f'Recovery delta not found: {DELTA_PATH}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with DELTA_PATH.open('r', encoding='utf-8', newline='') as delta_file:
        recoveries = {row['source1_entity_id'].encode('utf-8'): row['matched_entity_ids'].encode('utf-8') for row in csv.DictReader(delta_file, delimiter='\t')}
    if len(recoveries) != 7847:
        raise ValueError(f'Recovery delta must contain exactly 7,847 unique rows, found {len(recoveries)}')
    temporary_output = args.output.with_suffix(args.output.suffix + '.tmp')
    matched_recoveries = 0
    row_count = 0
    with args.baseline.open('rb') as baseline_file, temporary_output.open('wb') as output_file:
        header = baseline_file.readline()
        if header.rstrip(b'\r\n').split(b'\t') != [b'source1_entity_id', b'matched_entity_ids']:
            raise ValueError('Unexpected baseline matching header')
        output_file.write(header.rstrip(b'\r\n') + b'\r\n')
        for line in baseline_file:
            row_count += 1
            content = line.rstrip(b'\r\n')
            line_ending = b'\r\n'
            source1_id, separator, matched_ids = content.partition(b'\t')
            if not separator:
                raise ValueError(f'Malformed baseline row {row_count + 1}')
            if source1_id in recoveries:
                if matched_ids:
                    raise ValueError(f'Refusing to overwrite non-empty baseline row: {source1_id.decode()}')
                output_file.write(source1_id + b'\t' + recoveries.pop(source1_id) + line_ending)
                matched_recoveries += 1
            else:
                output_file.write(content + line_ending)
    if row_count != 1732544 or matched_recoveries != 7847 or recoveries:
        temporary_output.unlink(missing_ok=True)
        raise ValueError(f'Recovery invariants failed: rows={row_count}, applied={matched_recoveries}, remaining={len(recoveries)}')
    temporary_output.replace(args.output)
    print(f'Applied {matched_recoveries:,} verified empty-to-singleton recoveries: {args.output}')
if __name__ == '__main__':
    main()
