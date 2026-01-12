# plot_mcp_server.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from fastmcp import FastMCP
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import logging
import numpy as np
from typing import List, Optional

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="PerovskiteSolarCellPlotServer")
    parser.add_argument('--port', type=int, default=50010, help='Server port (default: 50010)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50010
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = FastMCP("PerovskiteSolarCellPlotServer", host=args.host, port=args.port)

@mcp.tool()
async def plot_solar_cell_structure_vs_time(start_year: int, end_year: int) -> dict:
    """Analyze the development trend of perovskite solar cell structures over time
    
    This tool generates a normalized stacked bar chart showing the percentage distribution
    of different perovskite solar cell structures (n-i-p, p-i-n, tandem, other) across
    multiple years. Each year's data is normalized to 100% to show relative proportions.
    
    **Data Source**: Currently uses Excel file (20250623_crossref.xlsx) with temporary
    flag control, but can be configured to use process_search() function.
    
    **Chart Type**: Normalized stacked bar chart with percentage distribution
    
    **Workflow**:
    1. Validate input parameters (year range: 2000-2030)
    2. Read data from Excel file or database query
    3. Extract and validate year information from publication dates
    4. Classify structures into 4 categories: n-i-p, p-i-n, tandem, other
    5. Calculate yearly counts and normalize to percentages (sum = 100% per year)
    6. Generate interactive Plotly stacked bar chart
    7. Prepare detailed yearly analysis data for text summary
    
    **Structure Classification**:
    - n-i-p: Regular/conventional structures
    - p-i-n: Inverted structures  
    - tandem: Multi-junction/stacked structures
    - other: Unclassified or mixed structures
    
    Parameters:
    -----------
    start_year : int
        Starting year for analysis (inclusive). Must be >= 2000 and < end_year
    end_year : int
        Ending year for analysis (inclusive). Must be <= 2030 and > start_year
    
    Returns:
    --------
    dict
        A dictionary containing:
        - success (bool): Whether the operation succeeded
        - message (str): Success/error message
        - data_summary (str): Brief description of generated chart
        - total_records (int): Number of papers processed
        - structure_types (List[str]): List of structure categories used
        - yearly_analysis (dict): Detailed yearly breakdown with structure percentages
          Format: {
              "year": {
                  "total_papers": int,
                  "structure_percentages": {
                      "structure_name": {"count": int, "percentage": float}
                  }
              }
          }
        - plot (dict): Plotly figure dictionary for rendering
    
    Raises:
    -------
    Returns error dict if:
    - Invalid year range (start_year >= end_year)
    - Year range outside 2000-2030
    - Excel file not found
    - Required columns missing from data
    - No data found for specified time range
    - Data processing errors
    
    Example:
    --------
    >>> result = await plot_solar_cell_structure_vs_time(2020, 2023)
    >>> if result["success"]:
    ...     print(f"Generated chart with {result['total_records']} papers")
    ...     yearly_data = result["yearly_analysis"]
    """
    logging.info(f"🚀 Starting plot_solar_cell_structure_vs_time with start_year={start_year}, end_year={end_year}")
    
    try:
        # Validate input
        logging.info("🔍 Validating input parameters...")
        if start_year >= end_year:
            logging.error(f"❌ Validation failed: start_year ({start_year}) >= end_year ({end_year})")
            return {
                "success": False,
                "message": "❌ Error: Start year must be less than end year"
            }
        
        if start_year < 2000 or end_year > 2030:
            logging.error(f"❌ Validation failed: Year range {start_year}-{end_year} outside valid range 2000-2030")
            return {
                "success": False,
                "message": "❌ Error: Year range should be between 2000-2030"
            }
        
        logging.info("✅ Input validation passed")
        
        # Temporary condition: use Excel file as data source
        USE_EXCEL_FILE = True
        
        if USE_EXCEL_FILE:
            # Read Excel file
            logging.info("📂 Using Excel file as data source...")
            excel_file_path = '20250623_crossref.xlsx'
            
            if not os.path.exists(excel_file_path):
                logging.error(f"❌ Excel file not found: {excel_file_path}")
                return {
                    "success": False,
                    "message": "❌ Error: Excel file not found"
                }
            
            # Read Excel file
            df = pd.read_excel(excel_file_path)
            logging.info(f"📊 Excel file loaded successfully. Shape: {df.shape}")
            logging.info(f"📋 Columns: {df.columns.tolist()}")
            
            # Check if there are year and structure type related columns
            year_col = None
            structure_col = None
            
            for col in df.columns:
                if 'year' in col.lower() or 'date' in col.lower():
                    year_col = col
                    break
            
            for col in df.columns:
                if 'structure' in col.lower() or 'type' in col.lower():
                    structure_col = col
                    break
            
            if year_col is None or structure_col is None:
                logging.error(f"❌ Required columns not found. Year column: {year_col}, Structure column: {structure_col}")
                return {
                    "success": False,
                    "message": "❌ Error: Required columns (year/date and structure/type) not found in Excel file"
                }
            
            logging.info(f"📋 Using year column: {year_col}, structure column: {structure_col}")
            
            # Extract year - ensure type safety
            def safe_extract_year(date_value):
                """Safely extract year, ensure integer return"""
                try:
                    if pd.isna(date_value):
                        return None
                    
                    # If it's a date column, convert to date first then extract year
                    if 'date' in year_col.lower():
                        date_obj = pd.to_datetime(date_value, errors='coerce')
                        if pd.isna(date_obj):
                            return None
                        return int(date_obj.year)
                    else:
                        # If it's a numeric column, convert to integer year
                        year_val = pd.to_numeric(date_value, errors='coerce')
                        if pd.isna(year_val):
                            return None
                        year_int = int(float(year_val))
                        return year_int if 1900 <= year_int <= 2100 else None
                except (ValueError, TypeError, OverflowError):
                    return None
            
            # Apply year extraction
            logging.info(f"📅 Extracting years from column: {year_col}")
            df['Year'] = df[year_col].apply(safe_extract_year)
            
            # Show some year extraction examples
            if not df.empty:
                sample_years = df[[year_col, 'Year']].head(10)
                logging.info(f"📅 Year extraction samples:")
                for _, row in sample_years.iterrows():
                    logging.info(f"   '{row[year_col]}' -> {row['Year']}")
            
            # Filter out invalid years
            initial_count = len(df)
            df = df.dropna(subset=['Year'])
            df['Year'] = df['Year'].astype(int)
            logging.info(f"📊 Filtered out {initial_count - len(df)} rows with invalid years")
            
            # Record year range information
            if not df.empty:
                year_range = f"{df['Year'].min()} to {df['Year'].max()}"
                logging.info(f"📅 Available year range: {year_range}")
                logging.info(f"📅 Unique years: {sorted(df['Year'].unique())}")
            
            # Filter year range
            df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
            logging.info(f"📊 After year range filtering: {len(df)} rows remaining")
            
            # Standardize structure type names
            structure_mapping = {
                'nip': 'n-i-p',
                'n-i-p': 'n-i-p',
                'pin': 'p-i-n',
                'p-i-n': 'p-i-n',
                'tandem': 'tandem',
                'other': 'other'
            }
            
            def normalize_structure(structure_name):
                if pd.isna(structure_name):
                    return 'other'
                structure_name = str(structure_name).lower()
                for key, value in structure_mapping.items():
                    if key in structure_name:
                        return value
                return 'other'
            
            df['Structure_Type'] = df[structure_col].apply(normalize_structure)
            
            # Group by year and structure type for statistics
            yearly_stats = df.groupby(['Year', 'Structure_Type']).size().reset_index(name='Count')
            
            # Calculate total for each year
            yearly_totals = yearly_stats.groupby('Year')['Count'].sum().reset_index()
            yearly_totals.columns = ['Year', 'Total']
            
            # Merge data to calculate percentages
            yearly_stats = yearly_stats.merge(yearly_totals, on='Year')
            yearly_stats['Percentage'] = (yearly_stats['Count'] / yearly_stats['Total']) * 100
            
            logging.info(f"📊 Yearly statistics calculated: {yearly_stats.shape}")
            
        else:
            # Original logic (using process_search)
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            logging.info(f"📅 Querying data from {start_date} to {end_date}")
            
            summary, df, page_info = process_search(
                start_date_str=start_date,
                end_date_str=end_date,
                structure="All Structures",
                pce_min=0,
                pce_max=50,
                title_keyword="",
                page_size=100000,
                current_page=1
            )
            
            if df.empty:
                logging.warning(f"⚠️ No data found for {start_year}-{end_year}")
                return {
                    "success": False,
                    "message": f"❌ No data found for {start_year}-{end_year}"
                }
            
            # Process original data...
            df['Year'] = pd.to_datetime(df['Publication Date']).dt.year
            structure_counts = df.groupby(['Year', 'Solar Cell Structure']).size().reset_index(name='Count')
        
        # Create stacked bar chart
        logging.info("🎨 Creating stacked bar chart...")
        fig = go.Figure()
        
        # Define color mapping
        colors = {
            'n-i-p': '#1f77b4',
            'p-i-n': '#ff7f0e',
            'tandem': '#2ca02c',
            'other': '#d62728'
        }
        
        # Get all years - ensure they are integers
        years = sorted([int(year) for year in yearly_stats['Year'].unique()])
        structure_types = ['n-i-p', 'p-i-n', 'tandem', 'other']
        
        # Create stacked bar chart for each structure type
        for structure_type in structure_types:
            structure_data = yearly_stats[yearly_stats['Structure_Type'] == structure_type]
            
            # Create complete year-percentage mapping
            year_percentage_map = dict(zip(structure_data['Year'], structure_data['Percentage']))
            percentages = [year_percentage_map.get(year, 0) for year in years]
            
            fig.add_trace(go.Bar(
                x=years,
                y=percentages,
                name=structure_type,
                marker_color=colors.get(structure_type, '#666666'),
                text=[f'{p:.1f}%' if p > 0 else '' for p in percentages],
                textposition='inside',
                textfont=dict(size=10)
            ))
        
        # Configure chart layout
        fig.update_layout(
            title=f'Perovskite Solar Cell Structure Distribution ({start_year}-{end_year})',
            xaxis_title='Year',
            yaxis_title='Percentage (%)',
            barmode='stack',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(range=[0, 100]),
            xaxis=dict(
                dtick=1,  # Show one tick every 1 year
                tickmode='linear',
                tick0=start_year,  # Start from start year
                tickformat='d'  # Display as integer
            )
        )
        
        # Prepare yearly percentage analysis data
        yearly_analysis = {}
        for year in years:
            # Ensure year is integer
            year = int(year)
            year_data = yearly_stats[yearly_stats['Year'] == year]
            
            # Safely calculate yearly total
            try:
                year_total = int(year_data['Count'].sum())
            except (ValueError, TypeError):
                year_total = 0
            
            year_analysis = {
                'total_papers': year_total,
                'structure_percentages': {}
            }
            
            for structure in structure_types:
                structure_data = year_data[year_data['Structure_Type'] == structure]
                if not structure_data.empty:
                    try:
                        count = int(float(structure_data['Count'].iloc[0]))
                        percentage = (count / year_total * 100) if year_total > 0 else 0
                        year_analysis['structure_percentages'][structure] = {
                            'count': count,
                            'percentage': round(percentage, 1)
                        }
                    except (ValueError, TypeError, IndexError):
                        year_analysis['structure_percentages'][structure] = {
                            'count': 0,
                            'percentage': 0.0
                        }
                else:
                    year_analysis['structure_percentages'][structure] = {
                        'count': 0,
                        'percentage': 0.0
                    }
            
            # Use integer year as key
            yearly_analysis[str(year)] = year_analysis

        # Return result
        logging.info("✅ Chart creation completed successfully")
        result = {
            "success": True,
            "message": f"🎨 Successfully generated structure development trend chart for {start_year}-{end_year}!",
            "data_summary": f"Generated normalized stacked bar chart from Excel data",
            "total_records": len(df) if USE_EXCEL_FILE else len(df),
            "structure_types": structure_types,
            "yearly_analysis": yearly_analysis,
            "plot": fig.to_dict()
        }
        logging.info(f"📤 Returning result with normalized percentages and yearly analysis")
        return result
        
    except Exception as e:
        logging.error(f"❌ plot_solar_cell_structure_vs_time error: {str(e)}")
        logging.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "message": f"❌ Plotting failed: {str(e)}"
        }

# @mcp.tool()
async def plot_structure_count_vs_time(start_year: int, end_year: int, valid_structure_types: Optional[List[str]] = None) -> dict:
    return {
        "success": False,
        "message": "❌ This tool is temporarily disabled and will be available in future versions"
    }

@mcp.tool()
async def plot_pce_vs_time_from_excel(start_date: str, end_date: str) -> dict:
    """Generate PCE (Power Conversion Efficiency) vs publication time scatter plot from Excel data
    
    This tool creates an interactive scatter plot showing how perovskite solar cell
    Power Conversion Efficiency (PCE) values evolve over time, with different colors
    representing different device structure types. The plot helps visualize efficiency
    trends and compare performance across structure categories.
    
    **Data Source**: Reads from Excel file (20250623_crossref.xlsx) containing
    crossref publication data with PCE measurements and device structure information.
    
    **Chart Type**: Interactive scatter plot with color-coded structure types
    
    **Workflow**:
    1. Load and validate Excel file existence and required columns
    2. Convert publication_date (YYYY-MM-DD) to float format (year + fractional time)
    3. Filter data by date range and remove invalid/missing dates
    4. Classify device structures using intelligent pattern matching:
       - Regular_n-i-p_structure: Contains n-i-p, nip, regular, conventional, etc.
       - Inverted_p-i-n_structure: Contains p-i-n, pin, inverted, inverse, etc.
       - Tandem_structure: Contains tandem, stack, multi-junction, etc.
       - Other: Unclassified structures (excluded from plot)
    5. Filter valid PCE data (positive numerical values)
    6. Generate scatter plot with hover information and structure-based coloring
    7. Calculate yearly analysis including average and maximum PCE by structure type
    
    **Required Excel Columns**:
    - publication_date: Publication date in YYYY-MM-DD format
    - jv_reverse_scan_pce: PCE values (numerical, positive)
    - solar_cell_structure: Device structure description text
    - title: Paper title (optional, for hover info)
    
    **Structure Classification Logic**:
    Uses intelligent pattern matching with priority rules:
    1. Tandem patterns (highest priority): tandem, stack, multi-junction, etc.
    2. Strong p-i-n indicators: inverted, inverse, p-i-n 
    3. n-i-p patterns: n-i-p, nip, regular, conventional, normal
    4. Weak p-i-n patterns: pin (only if no n-i-p indicators)
    5. Default patterns: planar, mesoporous → Regular n-i-p
    
    Parameters:
    -----------
    start_date : str
        Starting date for analysis in YYYY-MM-DD format (e.g., "2020-01-01")
        Must be a valid date string that can be parsed by pandas.to_datetime()
    end_date : str
        Ending date for analysis in YYYY-MM-DD format (e.g., "2024-12-31")
        Must be a valid date string that can be parsed by pandas.to_datetime()
    
    Returns:
    --------
    dict
        A dictionary containing:
        - success (bool): Whether the operation succeeded
        - message (str): Success/error message
        - data_summary (str): Brief description of generated chart
        - total_records (int): Number of data points displayed in chart
        - total_points_including_other (int): Total data points including excluded "Other"
        - excluded_other_points (int): Number of "Other" category points excluded
        - structure_stats (dict): Statistical summary by structure type
          Format: {
              "structure_name": {
                  "count": int,    # Number of papers
                  "mean": float,   # Average PCE
                  "max": float,    # Maximum PCE
                  "min": float     # Minimum PCE
              }
          }
        - yearly_pce_analysis (dict): Detailed yearly analysis by structure
          Format: {
              "year": {
                  "total_papers": int,
                  "overall_avg_pce": float,
                  "structure_avg_pce": {
                      "structure_name": {
                          "avg_pce": float,
                          "max_pce": float, 
                          "count": int
                      }
                  }
              }
          }
        - date_range (List[str]): Input date range [start_date, end_date]
        - plot (dict): Plotly figure dictionary for rendering
    
    Raises:
    -------
    Returns error dict if:
    - Excel file not found at expected path
    - Missing required columns in Excel file
    - Invalid date range format or unparseable dates
    - No data found for specified date range
    - All PCE values are invalid/missing
    - Data processing errors during analysis
    
    Example:
    --------
    >>> result = await plot_pce_vs_time_from_excel("2020-01-01", "2024-12-31")
    >>> if result["success"]:
    ...     print(f"Chart generated with {result['total_records']} data points")
    ...     yearly_data = result["yearly_pce_analysis"]
    ...     for year, data in yearly_data.items():
    ...         print(f"{year}: Overall avg PCE = {data['overall_avg_pce']}%")
    
    >>> # Check structure-specific performance
    >>> stats = result["structure_stats"]  
    >>> for structure, metrics in stats.items():
    ...     print(f"{structure}: Avg {metrics['mean']:.1f}%, Max {metrics['max']:.1f}%")
    """
    logging.info(f"🚀 Starting plot_pce_vs_time_from_excel with start_date={start_date}, end_date={end_date}")
    
    try:
        # Read Excel file
        logging.info("📂 Reading Excel file...")
        excel_file_path = '20250623_crossref.xlsx'
        
        if not os.path.exists(excel_file_path):
            logging.error(f"❌ Excel file not found: {excel_file_path}")
            return {
                "success": False,
                "message": "❌ Error: Excel file not found"
            }
        
        # Read Excel file
        df = pd.read_excel(excel_file_path)
        logging.info(f"📊 Excel file loaded successfully. Shape: {df.shape}")
        logging.info(f"📋 Columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ['publication_date', 'jv_reverse_scan_pce', 'solar_cell_structure']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"❌ Missing required columns: {missing_columns}")
            return {
                "success": False,
                "message": f"❌ Error: Missing required columns: {missing_columns}"
            }
        
        # Convert date format and filter data
        logging.info("📅 Processing publication dates...")
        
        def convert_date_to_float(date_str):
            """Convert various date formats to float year (supports YYYY, YYYY-MM-DD, integer, etc.)"""
            try:
                if pd.isna(date_str):
                    return None
                
                # Convert to string for processing
                date_str = str(date_str).strip()
                
                # Handle empty strings
                if not date_str or date_str.lower() in ['nan', 'none', 'null', '']:
                    return None
                
                # Try to parse directly as integer year (e.g., 2025)
                try:
                    year_int = int(float(date_str))
                    if 1900 <= year_int <= 2100:  # Reasonable year range
                        return float(year_int)
                except (ValueError, OverflowError):
                    pass
                
                # Try to parse as date format
                try:
                    date_obj = pd.to_datetime(date_str, errors='coerce')
                    if not pd.isna(date_obj):
                        year = date_obj.year
                        month = date_obj.month
                        day = date_obj.day
                        
                        # Calculate if it's a leap year
                        import calendar
                        days_in_year = 366 if calendar.isleap(year) else 365
                        
                        float_date = year + (month - 1) / 12 + (day - 1) / days_in_year
                        return float_date
                except:
                    pass
                
                # Try to extract year from string (e.g., "2025-02-03" -> 2025)
                try:
                    import re
                    year_match = re.search(r'(\d{4})', date_str)
                    if year_match:
                        year_int = int(year_match.group(1))
                        if 1900 <= year_int <= 2100:
                            return float(year_int)
                except:
                    pass
                
                logging.warning(f"⚠️ Could not parse date: {date_str}")
                return None
                
            except Exception as e:
                logging.warning(f"⚠️ Failed to convert date {date_str}: {e}")
                return None
        
        # Apply date conversion
        logging.info("📅 Converting publication dates to float format...")
        df['date_float'] = df['publication_date'].apply(convert_date_to_float)
        
        # Ensure PCE column is numeric for subsequent comparisons
        df['jv_reverse_scan_pce'] = pd.to_numeric(df['jv_reverse_scan_pce'], errors='coerce')
        
        # Record conversion results statistics
        total_count = len(df)
        valid_dates = df['date_float'].notna().sum()
        invalid_dates = total_count - valid_dates
        
        logging.info(f"📊 Date conversion results: {valid_dates}/{total_count} valid dates, {invalid_dates} invalid dates")
        
        # Show some conversion examples
        if not df.empty:
            sample_data = df[['publication_date', 'date_float']].head(10)
            logging.info(f"📅 Date conversion samples:")
            for _, row in sample_data.iterrows():
                logging.info(f"   '{row['publication_date']}' -> {row['date_float']}")
        
        # Filter out data with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=['date_float'])
        logging.info(f"📊 Filtered out {initial_count - len(df)} rows with invalid dates")
        
        # Convert date range to float for filtering
        start_date_float = convert_date_to_float(start_date)
        end_date_float = convert_date_to_float(end_date)
        
        if start_date_float is None or end_date_float is None:
            logging.error(f"❌ Invalid date range: {start_date} to {end_date}")
            return {
                "success": False,
                "message": "❌ Error: Invalid date range format"
            }
        
        logging.info(f"📅 Date range converted: {start_date} -> {start_date_float}, {end_date} -> {end_date_float}")
        
        # Ensure date_float column is numeric type before comparison
        def safe_compare_date(date_val, start_val, end_val):
            """Safely compare date values"""
            try:
                if pd.isna(date_val):
                    return False
                date_float = float(date_val)
                return start_val <= date_float <= end_val
            except (ValueError, TypeError):
                return False
        
        # Filter data by date range
        date_mask = df['date_float'].apply(lambda x: safe_compare_date(x, start_date_float, end_date_float))
        df = df[date_mask]
        logging.info(f"📊 Filtered data by date range. Remaining rows: {len(df)}")
        
        if df.empty:
            logging.warning(f"⚠️ No data found for date range {start_date} to {end_date}")
            return {
                "success": False,
                "message": f"❌ No data found for date range {start_date} to {end_date}"
            }
        
        # Classify device structures
        logging.info("🔧 Classifying device structures...")
        
        def classify_structure(structure_str):
            """Classify device structure based on solar_cell_structure field (loose matching)"""
            if pd.isna(structure_str):
                return 'Other'
            
            structure_str = str(structure_str).lower().strip()
            
            # First check tandem related patterns (highest priority, as tandem might contain other keywords)
            tandem_patterns = ['tandem', 'stack', 'multi-junction', 'multijunction', 'double', 'triple', 'perovskite/silicon']
            for pattern in tandem_patterns:
                if pattern in structure_str:
                    return 'Tandem_structure'
            
            # Strong p-i-n indicators (if these words are included, even if n-i-p is present, classify as p-i-n)
            strong_pin_patterns = ['inverted', 'inverse', 'p-i-n']
            for pattern in strong_pin_patterns:
                if pattern in structure_str:
                    return 'Inverted_p-i-n_structure'
            
            # Weak p-i-n indicators (only match if no n-i-p indicators are present)
            weak_pin_patterns = ['pin']
            has_weak_pin = any(pattern in structure_str for pattern in weak_pin_patterns)
            
            # n-i-p matching (including various spellings)
            nip_patterns = ['n-i-p', 'nip', 'regular', 'conventional', 'normal', 'standard']
            has_nip = any(pattern in structure_str for pattern in nip_patterns)
            
            # Decision logic
            if has_nip:
                return 'Regular_n-i-p_structure'
            elif has_weak_pin:
                return 'Inverted_p-i-n_structure'
            
            # If no matches, check some other possible patterns
            if any(word in structure_str for word in ['planar', 'mesoporous', 'meso']):
                return 'Regular_n-i-p_structure'  # Default to regular
            
            return 'Other'
        
        df['structure_category'] = df['solar_cell_structure'].apply(classify_structure)
        
        # Record detailed classification results
        logging.info("🔧 Structure classification results:")
        classification_details = df.groupby(['solar_cell_structure', 'structure_category']).size().reset_index(name='count')
        for _, row in classification_details.head(20).iterrows():  # Show first 20 classification results
            logging.info(f"   '{row['solar_cell_structure']}' → {row['structure_category']} ({row['count']} records)")
        
        # Filter valid PCE data
        df = df.dropna(subset=['jv_reverse_scan_pce'])
        df = df[df['jv_reverse_scan_pce'] > 0]  # Filter out non-positive PCE values
        
        logging.info(f"📊 Final data shape after all filtering: {df.shape}")
        logging.info(f"🏗️ Structure categories: {df['structure_category'].value_counts().to_dict()}")
        
        # Create scatter plot
        logging.info("🎨 Creating scatter plot...")
        fig = go.Figure()
        
        # Define color mapping and standard names
        structure_info = {
            'Regular_n-i-p_structure': {
                'color': '#1f77b4',
                'display_name': 'Regular n-i-p Structure'
            },
            'Inverted_p-i-n_structure': {
                'color': '#ff7f0e', 
                'display_name': 'Inverted p-i-n Structure'
            },
            'Tandem_structure': {
                'color': '#2ca02c',
                'display_name': 'Tandem Structure'
            },
            'Other': {
                'color': '#d62728',
                'display_name': 'Other Structures'
            }
        }
        
        # Create scatter plot for each structure type (skipping Other category)
        for structure_type, info in structure_info.items():
            # Skip Other category
            if structure_type == 'Other':
                logging.info(f"⏭️ Skipping Other category ({len(df[df['structure_category'] == structure_type])} points)")
                continue
                
            structure_data = df[df['structure_category'] == structure_type]
            
            if not structure_data.empty:
                logging.info(f"📈 Adding {len(structure_data)} points for {structure_type}")
                
                fig.add_trace(go.Scatter(
                    x=structure_data['date_float'],
                    y=structure_data['jv_reverse_scan_pce'],
                    mode='markers',
                    name=info['display_name'],  # Use standard display name
                    marker=dict(
                        color=info['color'],
                        size=6,
                        opacity=0.7
                    ),
                    text=structure_data['title'].fillna('No title'),
                    hovertemplate=(
                        '<b>%{text}</b><br>'
                        'Date: %{x:.2f}<br>'
                        'PCE: %{y:.2f}%<br>'
                        'Structure: ' + info['display_name'] + '<br>'
                        '<extra></extra>'
                    )
                ))
        
        # Configure chart layout
        fig.update_layout(
            title=f'PCE vs Time ({start_date} to {end_date})',
            xaxis_title='Year',
            yaxis_title='PCE (%)',
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                dtick=1,  # Show one tick every 1 year
                tickmode='linear',
                tick0=int(float(start_date_float)),  # Ensure type safety for conversion
                tickformat='d'  # Display as integer
            )
        )
        
        # Calculate statistics (excluding Other category)
        df_for_stats = df[df['structure_category'] != 'Other']
        stats_by_structure = df_for_stats.groupby('structure_category')['jv_reverse_scan_pce'].agg(['count', 'mean', 'max', 'min']).round(2)
        
        # Calculate yearly average efficiency analysis
        # Ensure year column is integer type
        df_for_stats = df_for_stats.copy()  # Avoid modifying original dataframe
        
        # More secure year extraction: ensure all date_float are numeric type
        def safe_extract_year(date_float):
            """Safely extract year from float date"""
            try:
                if pd.isna(date_float):
                    return None
                year_val = int(float(date_float))
                return year_val if 1900 <= year_val <= 2100 else None
            except (ValueError, TypeError, OverflowError):
                return None
        
        df_for_stats['year'] = df_for_stats['date_float'].apply(safe_extract_year)
        
        # Filter out invalid years
        df_for_stats = df_for_stats.dropna(subset=['year'])
        df_for_stats['year'] = df_for_stats['year'].astype(int)
        
        yearly_pce_analysis = {}
        
        # Group and calculate average efficiency by year and structure type
        unique_years = sorted(df_for_stats['year'].unique())
        logging.info(f"📅 Unique years found: {unique_years}")
        
        for year in unique_years:
            # Ensure year is integer type
            year = int(year)
            logging.info(f"📊 Processing year {year} (type: {type(year)})")
            
            # Use type-safe year filtering
            year_data = df_for_stats[df_for_stats['year'].astype(int) == year]
            # Safely calculate yearly statistics
            try:
                total_papers = len(year_data)
                overall_avg_pce = round(float(year_data['jv_reverse_scan_pce'].mean()), 2) if total_papers > 0 else 0.0
            except (ValueError, TypeError):
                total_papers = 0
                overall_avg_pce = 0.0
            
            year_analysis = {
                'total_papers': total_papers,
                'structure_avg_pce': {},
                'overall_avg_pce': overall_avg_pce
            }
            
            for structure_type in ['Regular_n-i-p_structure', 'Inverted_p-i-n_structure', 'Tandem_structure']:
                structure_data = year_data[year_data['structure_category'] == structure_type]
                if not structure_data.empty:
                    try:
                        avg_pce = round(float(structure_data['jv_reverse_scan_pce'].mean()), 2)
                        max_pce = round(float(structure_data['jv_reverse_scan_pce'].max()), 2)
                        count = len(structure_data)
                        year_analysis['structure_avg_pce'][structure_type] = {
                            'avg_pce': avg_pce,
                            'max_pce': max_pce,
                            'count': count
                        }
                    except (ValueError, TypeError):
                        year_analysis['structure_avg_pce'][structure_type] = {
                            'avg_pce': 0.0,
                            'max_pce': 0.0,
                            'count': 0
                        }
                else:
                    year_analysis['structure_avg_pce'][structure_type] = {
                        'avg_pce': 0.0,
                        'max_pce': 0.0,
                        'count': 0
                    }
            
            # Use string as dictionary key to ensure consistency
            yearly_pce_analysis[str(year)] = year_analysis
        
        # Record data for Other category but do not display in chart
        other_count = len(df[df['structure_category'] == 'Other'])
        if other_count > 0:
            logging.info(f"📊 Other category excluded from chart: {other_count} data points")
        
        # Return result
        logging.info("✅ Scatter plot creation completed successfully")
        displayed_points = len(df_for_stats)
        total_points = len(df)
        
        result = {
            "success": True,
            "message": f"🎨 Successfully generated PCE vs time scatter plot for {start_date} to {end_date}!",
            "data_summary": f"Generated scatter plot with {displayed_points} data points from Excel file (excluded {other_count} Other category points)",
            "total_records": displayed_points,
            "total_points_including_other": total_points,
            "excluded_other_points": other_count,
            "structure_stats": stats_by_structure.to_dict(),
            "yearly_pce_analysis": yearly_pce_analysis,
            "date_range": [start_date, end_date],
            "plot": fig.to_dict()
        }
        logging.info(f"📤 Returning result with {displayed_points} displayed data points ({other_count} Other points excluded)")
        return result
        
    except Exception as e:
        logging.error(f"❌ plot_pce_vs_time_from_excel error: {str(e)}")
        logging.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "message": f"❌ Plotting failed: {str(e)}"
        }

def run_server():
    """Start MCP server for Perovskite Solar Cell Data Analysis
    
    Initializes and runs the FastMCP server providing three main analytical tools
    for perovskite solar cell research data visualization and trend analysis.
    """
    print("🚀 Starting Perovskite Solar Cell Data Analysis MCP Server...")
    print(f"📍 Server address: http://{args.host}:{args.port}")
    print(f"📝 Log level: {args.log_level}")
    print("🛠️  Available MCP Tools:")
    print("   📊 plot_solar_cell_structure_vs_time:")
    print("      └─ Generate normalized stacked bar chart of structure type percentages over time")
    print("   🔬 plot_pce_vs_time_from_excel:")
    print("      └─ Generate PCE scatter plot over time with structure-based color coding")
    print("   ✨ Features: Interactive charts, detailed analysis, type-safe data processing")
    print("   📚 Data Sources: Excel files + database queries with intelligent structure classification")
    print("   ⚠️  Note: Some tools may be temporarily disabled for maintenance")
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)

if __name__ == "__main__":
    # Configure logging based on command line arguments
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test new functions
    import asyncio
    run_server()

