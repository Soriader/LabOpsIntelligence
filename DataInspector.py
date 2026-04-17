import pandas as pd
import numpy as np


class DataInspector:

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, name: str = "Dataset") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            raise ValueError(f"{name} is empty!")

    @staticmethod
    def _validate_column_exists(df: pd.DataFrame, column_name: str) -> None:
        if column_name is None:
            raise ValueError("column_name must be provided")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

    @staticmethod
    def _is_missing_value(series: pd.Series) -> pd.Series:
        """
        Returns a boolean mask for missing values:
        - NaN / None
        - empty string ""
        - whitespace-only strings
        """
        return series.isna() | series.astype(str).str.strip().eq("")

    @staticmethod
    def display_dimensions(df: pd.DataFrame, name: str = "Dataset") -> None:
        DataInspector._validate_dataframe(df, name)

        print("=" * 50)
        print(f"📊 {name}")
        print("=" * 50)
        print(f"📈 Rows: {df.shape[0]:,}")
        print(f"📑 Columns: {df.shape[1]}")
        print(f"🧩 Shape: {df.shape[0]:,} × {df.shape[1]}")
        print("=" * 50)

    @staticmethod
    def display_null_value(df: pd.DataFrame, name: str = "Dataset", column_name: str = None) -> None:
        DataInspector._validate_dataframe(df, name)
        DataInspector._validate_column_exists(df, column_name)

        print("=" * 50)
        print(f"📊 MISSING VALUES ANALYSIS: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)

        total_rows = len(df)
        series = df[column_name]

        nan_count = series.isna().sum()
        empty_string_count = 0
        whitespace_count = 0

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            empty_string_count = series.fillna("__NA__").eq("").sum()
            whitespace_count = (
                series.fillna("__NA__").astype(str).str.strip().eq("").sum()
                - empty_string_count
            )

        total_missing = nan_count + empty_string_count + whitespace_count

        print(f"📈 Total rows: {total_rows:,}")
        print("🔍 Missing values breakdown:")
        print(f"   • NaN/None: {nan_count:,} ({nan_count / total_rows * 100:.2f}%)")

        if empty_string_count > 0:
            print(f"   • Empty strings '': {empty_string_count:,} ({empty_string_count / total_rows * 100:.2f}%)")

        if whitespace_count > 0:
            print(f"   • Whitespace strings: {whitespace_count:,} ({whitespace_count / total_rows * 100:.2f}%)")

        print(f"\n📊 TOTAL MISSING: {total_missing:,} ({total_missing / total_rows * 100:.2f}%)")

        if total_missing == 0:
            print(f"\n✅ No missing values found in column '{column_name}'!")

        print("=" * 50)

    @staticmethod
    def display_unique_value(df: pd.DataFrame, name: str = "Dataset", column_name: str = None) -> None:
        DataInspector._validate_dataframe(df, name)
        DataInspector._validate_column_exists(df, column_name)

        unique_values = df[column_name].unique()
        unique_count = df[column_name].nunique(dropna=False)
        total_rows = len(df)

        print("=" * 50)
        print(f"📊 UNIQUE VALUES ANALYSIS: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)
        print(f"📈 Total unique values: {unique_count:,} / {total_rows:,} ({unique_count / total_rows * 100:.2f}%)")

        print(f"\n🔍 Unique values (first {min(20, unique_count)}):")
        for i, value in enumerate(unique_values[:20], start=1):
            print(f"   {i}. {value}")

        if unique_count > 20:
            print(f"   ... and {unique_count - 20} more")

        print("=" * 50)

    @staticmethod
    def get_shape_info(df: pd.DataFrame) -> tuple[int, int, float]:
        DataInspector._validate_dataframe(df)

        return (
            df.shape[0],
            df.shape[1],
            df.memory_usage(deep=True).sum() / 1024 ** 2
        )

    @staticmethod
    def display_correct_value(
        df: pd.DataFrame,
        name: str = "Dataset",
        column_name: str = None,
        allowed_values: list = None
    ) -> None:
        DataInspector._validate_dataframe(df, name)
        DataInspector._validate_column_exists(df, column_name)

        if allowed_values is None or len(allowed_values) == 0:
            raise ValueError("allowed_values must be provided and cannot be empty")

        unique_values = df[column_name].dropna().unique()
        invalid_values = [val for val in unique_values if val not in allowed_values]

        invalid_mask = ~df[column_name].isin(allowed_values)
        invalid_count = invalid_mask.sum()

        print("=" * 50)
        print(f"📊 VALUE VALIDATION: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)

        print(f"📈 Allowed values: {', '.join(map(str, allowed_values))}")
        print(f"🔍 Found unique values: {', '.join(map(str, unique_values))}")

        if invalid_values:
            print(f"\n❌ Incorrect unique values found: {', '.join(map(str, invalid_values))}")
            print(f"📊 Count of incorrect records: {invalid_count:,}")
        else:
            print("\n✅ All values are correct! Only allowed values present.")

        print("=" * 50)

    @staticmethod
    def units_correctness(
        df: pd.DataFrame,
        name: str = "Dataset",
        parameter_col: str = None,
        unit_col: str = "unit",
        correct_units: dict = None
    ) -> None:
        DataInspector._validate_dataframe(df, name)
        DataInspector._validate_column_exists(df, parameter_col)
        DataInspector._validate_column_exists(df, unit_col)

        if correct_units is None or len(correct_units) == 0:
            raise ValueError("correct_units must be provided and cannot be empty")

        print("=" * 60)
        print(f"📊 UNIT CORRECTNESS ANALYSIS: {name}")
        print("=" * 60)

        total_incorrect = 0

        for param, expected_unit in correct_units.items():
            param_df = df[df[parameter_col] == param]

            if param_df.empty:
                continue

            incorrect = param_df[param_df[unit_col] != expected_unit]

            if len(incorrect) > 0:
                total_incorrect += len(incorrect)

                print(f"\n❌ PARAMETER: {param}")
                print(f"   Expected unit: {expected_unit}")
                print(f"   Incorrect records: {len(incorrect):,}")

                for _, row in incorrect.iterrows():
                    print(
                        f"      • ID: {row.get('test_id', 'N/A')} → actual unit: {row[unit_col]}"
                    )

        if total_incorrect == 0:
            print("✅ All units are correct!")

        print(f"\n📊 TOTAL INCORRECT UNIT RECORDS: {total_incorrect:,}")
        print("=" * 60)

    @staticmethod
    def unlogical_combination(
        df: pd.DataFrame,
        name: str = "Dataset",
        status_col: str = None,
        result_col: str = None
    ) -> None:
        DataInspector._validate_dataframe(df, name)
        DataInspector._validate_column_exists(df, status_col)
        DataInspector._validate_column_exists(df, result_col)

        print("=" * 60)
        print(f"📊 LOGICAL COMBINATION VALIDATION: {name}")
        print("=" * 60)

        missing_result_mask = DataInspector._is_missing_value(df[result_col])

        ok_but_missing = df[(df[status_col] == "OK") & missing_result_mask]
        error_but_has_result = df[(df[status_col] == "ERROR") & ~missing_result_mask]

        errors_found = 0

        if not ok_but_missing.empty:
            print("❌ Records with status='OK' but missing result_value:")
            for _, row in ok_but_missing.iterrows():
                print(f"   • ID: {row.get('test_id', 'N/A')}")
            errors_found += len(ok_but_missing)

        if not error_but_has_result.empty:
            print("\n❌ Records with status='ERROR' but result_value exists:")
            for _, row in error_but_has_result.iterrows():
                print(f"   • ID: {row.get('test_id', 'N/A')} → result_value: {row[result_col]}")
            errors_found += len(error_but_has_result)

        if errors_found == 0:
            print("✅ All combinations are logical!")
        else:
            print(f"\n📈 Total logical errors: {errors_found:,}")

        print("=" * 60)

    @staticmethod
    def duplicated_in_tests(
        df: pd.DataFrame,
        name: str = "Dataset",
        sample_id: str = "sample_id",
        parameter: str = "parameter",
        date_col: str = None
    ) -> None:
        DataInspector._validate_dataframe(df, name)

        required_cols = [sample_id, parameter]
        for col in required_cols:
            DataInspector._validate_column_exists(df, col)

        print("=" * 60)
        print(f"📊 DUPLICATE ANALYSIS: {name}")
        print("=" * 60)
        print(f"🔍 Checking duplicates by: {sample_id} + {parameter}")

        duplicates_mask = df.duplicated(subset=[sample_id, parameter], keep=False)
        duplicate_rows = df[duplicates_mask]

        if len(duplicate_rows) == 0:
            print("\n✅ No duplicates found!")
        else:
            duplicate_groups = duplicate_rows.groupby([sample_id, parameter])

            print(f"\n❌ Found {len(duplicate_rows):,} duplicate rows")
            print(f"📊 {len(duplicate_groups):,} parameter group(s) have duplicates:\n")

            for (s_id, param), group in duplicate_groups:
                print(f"   📌 {s_id} → {param}: {len(group)} occurrences")

                if date_col and date_col in df.columns:
                    dates = group[date_col].tolist()
                    print(f"      Dates: {dates}")

                if "test_id" in df.columns:
                    test_ids = group["test_id"].tolist()
                    print(f"      Test IDs: {test_ids}")

                print()

        duplicate_percent = (len(duplicate_rows) / len(df)) * 100 if len(df) > 0 else 0

        print("\n📈 Summary:")
        print(f"   Total rows: {len(df):,}")
        print(f"   Duplicate rows: {len(duplicate_rows):,} ({duplicate_percent:.2f}%)")
        print("=" * 60)

    @staticmethod
    def find_outliers(
        df: pd.DataFrame,
        name: str = "Dataset",
        param_col: str = "parameter",
        result_col: str = "result_value",
        unit_col: str = "unit",
        custom_thresholds: dict = None
    ) -> None:
        DataInspector._validate_dataframe(df, name)

        required_cols = [param_col, result_col, unit_col]
        for col in required_cols:
            DataInspector._validate_column_exists(df, col)

        default_param_thresholds = {
            "sulfur": {"min": 0, "max": 50, "unit": "mg/kg"},
            "water": {"min": 0, "max": 600, "unit": "mg/kg"},
            "density": {"min": 600, "max": 1100, "unit": "kg/m3"},
            "viscosity": {"min": 0, "max": 150, "unit": "cSt"},
            "chloride": {"min": 0, "max": 100, "unit": "mg/kg"},
            "flash_point": {"min": -10, "max": 100, "unit": "C"},
        }

        default_unit_thresholds = {
            "mg/kg": {"min": 0, "max": 1000},
            "kg/m3": {"min": 600, "max": 1100},
            "cSt": {"min": 0, "max": 500},
            "C": {"min": -50, "max": 300},
            "ppm": {"min": 0, "max": 1000},
            "mg/L": {"min": 0, "max": 1000},
            "%": {"min": 0, "max": 100},
        }

        param_thresholds = default_param_thresholds.copy()
        if custom_thresholds:
            param_thresholds.update(custom_thresholds)

        print("=" * 60)
        print(f"📊 OUTLIER DETECTION: {name}")
        print("=" * 60)

        outliers_found = 0
        invalid_values = 0
        parameters_checked = set()

        for idx, row in df.iterrows():
            param = row[param_col]
            unit = row[unit_col]
            result = row[result_col]

            if DataInspector._is_missing_value(pd.Series([result])).iloc[0]:
                continue

            parameters_checked.add(param)

            if param in param_thresholds:
                thresholds = param_thresholds[param]
                source = f"parameter-specific ({param})"

                try:
                    value = float(result)

                    if value > thresholds["max"]:
                        print(
                            f"❌ {param}: ID {row.get('test_id', idx)} - {value} {thresholds.get('unit', unit)} > {thresholds['max']} (too high) [based on: {source}]"
                        )
                        outliers_found += 1
                    elif value < thresholds["min"]:
                        print(
                            f"❌ {param}: ID {row.get('test_id', idx)} - {value} {thresholds.get('unit', unit)} < {thresholds['min']} (too low) [based on: {source}]"
                        )
                        outliers_found += 1

                except (ValueError, TypeError):
                    print(f"⚠️ {param}: ID {row.get('test_id', idx)} - Invalid numeric value: '{result}'")
                    invalid_values += 1

            elif unit in default_unit_thresholds:
                thresholds = default_unit_thresholds[unit]
                source = f"unit-based ({unit})"

                try:
                    value = float(result)

                    if value > thresholds["max"]:
                        print(
                            f"❌ {param}: ID {row.get('test_id', idx)} - {value} {unit} > {thresholds['max']} (too high) [based on: {source}]"
                        )
                        outliers_found += 1
                    elif value < thresholds["min"]:
                        print(
                            f"❌ {param}: ID {row.get('test_id', idx)} - {value} {unit} < {thresholds['min']} (too low) [based on: {source}]"
                        )
                        outliers_found += 1

                except (ValueError, TypeError):
                    print(f"⚠️ {param}: ID {row.get('test_id', idx)} - Invalid numeric value: '{result}'")
                    invalid_values += 1

            else:
                print(f"⚠️ {param}: No thresholds defined (unit: {unit}) - skipping validation")

        print("=" * 60)
        print(f"📊 OUTLIER SUMMARY: {name}")
        print("=" * 60)
        print(f"   Parameters analyzed: {len(parameters_checked)}")
        print(f"   Parameters with thresholds: {len(param_thresholds)}")
        print(f"   Outliers found: {outliers_found}")

        if invalid_values > 0:
            print(f"   Invalid numeric values: {invalid_values}")

        if outliers_found == 0 and invalid_values == 0:
            print("\n✅ No outliers or invalid values detected!")

        print("=" * 60)