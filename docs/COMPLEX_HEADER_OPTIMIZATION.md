# Complex Header and Table Filtering Optimization

## Overview

This document describes the enhancements made to the table extraction logic in `fiscal_mind/parser.py` to handle complex table headers and improve multi-table detection.

## Problem Statement (问题陈述)

优化现有的表格提取逻辑，使其能够处理复杂表头，例如超过两行的表头（同时也存在合并单元格）复杂表头，有些表头可能作为说明层级信息存在，可用连字符- 连接复杂信息。例如 华东-南京-浦口，华东-苏州-昆山等。 检测同一sheet中多个表格逻辑也需要优化，不要将离散的，较小的，临时的数据作为单独的表格。

**Translation:** Optimize the existing table extraction logic to handle complex headers, such as headers with more than two rows (also with merged cells). Some headers may exist as hierarchical information description, which can be connected with hyphens, for example: 华东-南京-浦口, 华东-苏州-昆山, etc. The logic for detecting multiple tables in the same sheet also needs to be optimized, not to treat discrete, small, temporary data as separate tables.

## Key Features Implemented

### 1. Multi-row Header Detection

**Feature:** Support for detecting and merging multi-row headers (up to 5 rows).

**How it works:**
- When a potential header row is detected, the algorithm checks up to `MAX_HEADER_ROWS` (5) subsequent rows
- Each row is evaluated to determine if it's part of the header
- Multi-row headers are merged using hyphen separators (e.g., "华东-南京-浦口")

**Configuration:**
```python
MAX_HEADER_ROWS = 5  # Maximum rows to consider for multi-row headers
```

**Example:**
```
Row 1:  |  区域  |   华东   |   华东   |   华北   |
Row 2:  |       |   南京   |   苏州   |   北京   |
Row 3:  |       |   浦口   |   昆山   |   朝阳   |
Data:   | ...   |   1200   |   980    |   1500   |

Result: ['区域', '华东-南京-浦口', '华东-苏州-昆山', '华北-北京-朝阳']
```

### 2. Merged Cell Handling

**Feature:** Correct handling of merged cells in headers.

**How it works:**
- Built a merged cell cache at the start of table detection for O(1) lookups
- When reading cell values, the algorithm checks if a cell is part of a merged range
- Returns the value from the top-left cell of the merged range

**Performance Optimization:**
- `_build_merged_cell_cache()`: Creates a dictionary mapping all cells in merged ranges to their values
- Prevents repeated O(n) iterations through merged cell ranges
- Significant performance improvement for worksheets with many merged cells

### 3. Table Size Filtering

**Feature:** Filters out small, temporary data that shouldn't be treated as tables.

**How it works:**
- Tables must meet minimum size requirements:
  - At least `MIN_TABLE_ROWS` (3) data rows
  - At least `MIN_TABLE_COLS` (2) columns
- Small data snippets are logged but not included in the detected tables

**Configuration:**
```python
MIN_TABLE_ROWS = 3  # Minimum data rows for a valid table
MIN_TABLE_COLS = 2  # Minimum columns for a valid table
```

**Example:**
```
Table 1:  5 rows x 5 cols  ✓ Detected (≥3 rows, ≥2 cols)
Table 2:  4 rows x 4 cols  ✓ Detected (≥3 rows, ≥2 cols)
Data 3:   2 rows x 2 cols  ✗ Filtered (< 3 rows)
Data 4:   1 row  x 2 cols  ✗ Filtered (< 3 rows)
```

## New Methods

### TableDetector._build_merged_cell_cache(ws)
Builds a cache dictionary for merged cell lookups.

**Args:**
- `ws`: openpyxl worksheet object

**Returns:**
- Dictionary mapping cell coordinates to merged cell values

**Purpose:** Performance optimization to avoid O(n) lookups for every cell access.

### TableDetector._get_merged_cell_value(ws, row_idx, col_idx, merged_cache=None)
Retrieves the value of a cell, accounting for merged cells.

**Args:**
- `ws`: openpyxl worksheet object
- `row_idx`: Row index (0-based)
- `col_idx`: Column index (0-based)
- `merged_cache`: Optional merged cell cache for performance

**Returns:**
- Cell value (from merged range top-left if applicable)

### TableDetector._detect_multi_row_headers(ws, data_array, start_row, start_col, end_col, merged_cache=None)
Detects and merges multi-row headers.

**Args:**
- `ws`: openpyxl worksheet object
- `data_array`: 2D array of cell values
- `start_row`: Starting row index
- `start_col`: Starting column index
- `end_col`: Ending column index
- `merged_cache`: Optional merged cell cache

**Returns:**
- Tuple of (merged_headers, header_row_count)

**Algorithm:**
1. Scan up to MAX_HEADER_ROWS rows starting from start_row
2. Check if each row looks like a header (text-based, not data)
3. For multi-row headers, merge values with hyphen separator
4. Avoid duplicates from vertically merged cells

### TableDetector._collect_potential_headers_with_merges(ws, data_array, row_idx, start_col, max_col, merged_cache=None)
Collects potential header cells in a row, considering merged cells.

**Args:**
- `ws`: openpyxl worksheet object
- `data_array`: 2D array of cell values
- `row_idx`: Row index to scan
- `start_col`: Starting column index
- `max_col`: Maximum column index
- `merged_cache`: Optional merged cell cache

**Returns:**
- Tuple of (potential_headers, end_col)

## Updated detect_tables() Method

The main `detect_tables()` method was updated to:

1. **Build merged cell cache** at the start for performance
2. **Use merged cell values** when scanning for table boundaries
3. **Call multi-row header detection** for all detected tables
4. **Filter tables** based on minimum size requirements
5. **Log detailed information** about detected and filtered tables

## Testing

### Test Files Created

1. **examples/create_complex_header_sample.py**
   - Script to create sample Excel files for testing
   - Creates two test files: complex_header_sample.xlsx and multi_table_with_small_data.xlsx

2. **examples/complex_header_sample.xlsx**
   - Contains a 3-row header with merged cells
   - Demonstrates hierarchical header structure (Region-City-District)
   - Example: "华东-南京-浦口", "华东-苏州-昆山", etc.

3. **examples/multi_table_with_small_data.xlsx**
   - Contains 2 valid tables and 2 small data snippets
   - Tests the table filtering functionality
   - Small tables are correctly filtered out

4. **tests/test_complex_headers.py**
   - Comprehensive test suite with 3 test cases:
     - Complex header detection
     - Small table filtering
     - Backward compatibility

### Test Results

All tests passing:
```
✓ 复杂表头检测 (Complex header detection)
✓ 小表格过滤 (Small table filtering)
✓ 向后兼容性 (Backward compatibility)
✓ 现有多表格测试 (Existing multi-table tests)
```

## Backward Compatibility

- Changes only affect code when `detect_multiple_tables=True` is set
- Default behavior (detect_multiple_tables=False) remains unchanged
- All existing tests continue to pass
- No breaking changes to API

## Performance Considerations

### Before Optimization
- O(n) iterations through merged cell ranges for every cell access
- Performance degradation proportional to number of merged cells

### After Optimization
- O(1) lookups using merged cell cache
- Cache built once at the start of table detection
- Significant performance improvement for worksheets with many merged cells

## Usage Example

```python
from fiscal_mind.parser import ExcelParser

# Create parser with multi-table detection enabled
parser = ExcelParser(detect_multiple_tables=True)

# Load Excel file with complex headers
doc = parser.load_document('complex_header_sample.xlsx')

# Get table information
sheet_name = doc.get_sheet_names()[0]
tables = doc.get_sheet_tables(sheet_name)

# Access hierarchical headers
for table in tables:
    print(f"Table at ({table.start_row}, {table.start_col})")
    print(f"Headers: {table.data.columns.tolist()}")
    # Output: ['区域', '华东-南京-浦口', '华东-苏州-昆山', ...]
```

## Security

CodeQL analysis completed with 0 alerts:
- No security vulnerabilities introduced
- All code follows best practices

## Future Enhancements

Possible future improvements:
- Configurable minimum table size thresholds via constructor parameters
- Support for different header merger separators (not just hyphen)
- Detection of headers with even more complex structures
- Performance profiling and further optimizations

## References

- Original issue: Optimize table extraction logic for complex headers
- Related: Multi-table detection (v2.1)
- File: `fiscal_mind/parser.py`
