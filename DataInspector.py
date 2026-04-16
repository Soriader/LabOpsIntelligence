import pandas as pd
import numpy as np

class DataInspector:

    @staticmethod
    def display_dimensions(df: pd.DataFrame, name: str = "Dataset") -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        print("=" * 50)
        print(f"📊 {name}")
        print("=" * 50)
        print(f"📈 Rows: {df.shape[0]:,}")
        print(f"📑 Columns: {df.shape[1]}")
        print(f"💾 Size: {df.shape[0]:,} × {df.shape[1]}")
        print("=" * 50)

    @staticmethod
    def display_null_value(df: pd.DataFrame, name: str = "Dataset",
                           column_name: str = None) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if column_name is None:
            raise ValueError("column_name must be provided")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        print("=" * 50)
        print(f"📊 MISSING VALUES ANALYSIS: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)

        total_rows = len(df)

        nan_count = df[column_name].isnull().sum()

        empty_string_count = 0
        whitespace_count = 0

        if df[column_name].dtype == 'object':  # Tylko dla kolumn tekstowych
            empty_string_count = (df[column_name] == "").sum()

            whitespace_count = df[column_name].astype(str).str.strip().eq("").sum() - empty_string_count

        total_missing = nan_count + empty_string_count + whitespace_count

        print(f"📈 Total rows: {total_rows:,}")
        print(f"🔍 Missing values breakdown:")
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
    def display_unique_value(df: pd.DataFrame, name: str = "Unique value", column_name: str = None) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if column_name is None:
            raise ValueError("column_name must be provided")

        unique_values = df[column_name].unique()
        unique_count = df[column_name].nunique()
        total_rows = len(df)

        print("=" * 50)
        print(f"📊 UNIQUE VALUES ANALYSIS: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)
        print(f"📈 Total unique values: {unique_count:,} / {total_rows:,} ({unique_count / total_rows * 100:.2f}%)")

        print(f"\n🔍 Unique values (first {min(20, unique_count)}):")
        for i, value in enumerate(unique_values[:20]):
            print(f"   {i + 1}. {value}")

        if unique_count > 20:
            print(f"   ... and {unique_count - 20} more")

        print("=" * 50)

    @staticmethod
    def get_shape_info(df: pd.DataFrame, name: str = "Dataset") -> tuple:

        return (df.shape[0], df.shape[1],
                df.memory_usage(deep=True).sum() / 1024 ** 2)

    @staticmethod
    def display_correct_value(df: pd.DataFrame, name: str = "Dataset",
                              column_name: str = None, allowed_values: list = None) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if allowed_values is None:
            raise ValueError("allowed_values must be provided")

        if column_name is None:
            raise ValueError("column_name must be provided")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        invalid_mask = ~df[column_name].isin(allowed_values)
        invalid_records = df[invalid_mask]
        invalid_count = len(invalid_records)

        unique_invalid_values = invalid_records[column_name].unique()

        total_rows = len(df)
        valid_count = total_rows - invalid_count

        print("=" * 50)
        print(f"📊 COLUMN VALIDATION: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)

        print(f"✅ Allowed values: {', '.join(map(str, allowed_values))}")
        print(f"📈 Total rows: {total_rows:,}")
        print(f"✔️  Valid records: {valid_count:,} ({valid_count / total_rows * 100:.2f}%)")

        if invalid_count > 0:
            print(f"❌ Invalid records: {invalid_count:,} ({invalid_count / total_rows * 100:.2f}%)")
            print(f"\n⚠️  Invalid values found: {', '.join(map(str, unique_invalid_values))}")

            print(f"\n📊 Invalid values distribution:")
            invalid_distribution = invalid_records[column_name].value_counts()
            for val, count in invalid_distribution.items():
                print(f"   • '{val}': {count} times")
        else:
            print(f"\n✅ All values are correct!")

        print("=" * 50)

    @staticmethod
    def units_correctness(df: pd.DataFrame, name: str = "Dataset",
                          column_name: str = None, correct_units: dict = None) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if correct_units is None:
            raise ValueError("correct_units dictionary must be provided")  # Poprawiony komunikat

        if column_name is None:
            raise ValueError("column_name must be provided")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        # Sprawdź czy kolumna 'unit' istnieje
        if 'unit' not in df.columns:
            raise ValueError("Column 'unit' not found in DataFrame")

        print("=" * 50)
        print(f"📊 CORRECT UNITS ANALYSIS: {name}")
        print("=" * 50)

        errors_found = 0

        for param, expected_unit in correct_units.items():
            param_df = df[df[column_name] == param]

            incorrect = param_df[param_df['unit'] != expected_unit]

            if len(incorrect) > 0:
                errors_found += len(incorrect)
                print(f"\n❌ PARAMETER: {param}")
                print(f"   Expected unit: {expected_unit}")
                print(f"   Found {len(incorrect)} incorrect record(s):")

                for _, row in incorrect.iterrows():
                    print(f"      • ID: {row['test_id']} → wrong unit: '{row['unit']}' (should be: '{expected_unit}')")

        if errors_found == 0:
            print("\n✅ All units are correct!")
        else:
            print(f"\n📈 Total incorrect units: {errors_found}")

        print("=" * 50)

    @staticmethod
    def unlogical_combination(df: pd.DataFrame, name: str = "Dataset",
                              status_col: str = None, result_col: str = None):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if name is None:
            print(f"⚠️ {name} is empty!")
            return

        if status_col is None:
            raise ValueError("column_name must be provided")

        if result_col is None:
            raise ValueError("column_name must be provided")

        if status_col not in df.columns:
            raise ValueError(f"Column '{status_col}' not found")

        if result_col not in df.columns:
            raise ValueError(f"Column '{result_col}' not found")

        print("=" * 50)
        print(f"📊 LOGICAL COMBINATION VALIDATION: {name}")
        print("=" * 50)

        errors_found = 0

        for idx, row in df.iterrows():
            status = row[status_col]
            result = row[result_col]

            if status == "OK" and pd.isna(result):
                print(f"❌ ID: {row.get('test_id', idx)} - Status 'OK' but result_value is missing")
                errors_found += 1

            elif status == "ERROR" and pd.notna(result):
                print(f"❌ ID: {row.get('test_id', idx)} - Status 'ERROR' but result_value exists: {result}")
                errors_found += 1

        if errors_found == 0:
            print("✅ All combinations are logical!")
        else:
            print(f"\n📈 Total logical errors: {errors_found}")

        print("=" * 50)

    @staticmethod
    def duplicated_in_tests(df: pd.DataFrame, name: str = "Dataset", test_id: str = 'test_id', parameter: str = 'parameter') -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if name is None:
            print(f"⚠️ {name} is empty!")
            return

        if test_id is None:
            raise ValueError("column_name must be provided")

        if parameter is None:
            raise ValueError("column_name must be provided")

        if test_id not in df.columns:
            raise ValueError(f"Column '{test_id}' not found")

        if parameter not in df.columns:
            raise ValueError(f"Column '{parameter}' not found")

        duplicates = df.duplicated(subset=[test_id, parameter], keep=False)

        print("=" * 50)
        print(f"📊 DUPLICATE ANALYSIS")
        print(duplicates)
        print("=" * 50)

        pass

    @staticmethod
    def find_outliers(df: pd.DataFrame, name: str = "Dataset",
                      unit_col: str = 'unit', result_col: str = 'result_value') -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if name is None:
            print(f"⚠️ {name} is empty!")
            return

        if unit_col is None:
            raise ValueError("column_name must be provided")

        if result_col is None:
            raise ValueError("column_name must be provided")

        if unit_col not in df.columns:
            raise ValueError(f"Column '{unit_col}' not found")

        if result_col not in df.columns:
            raise ValueError(f"Column '{result_col}' not found")

        thresholds = {
            'mg/kg': {'max': 1000, 'min': 0},
            'kg/m3': {'max': 1000, 'min': 0},
            'cSt': {'max': 500, 'min': 0},
            'C': {'max': 300, 'min': -50}  # temperatura może być ujemna!
        }

        print("=" * 50)
        print(f"📊 OUTLIER DETECTION: {name}")
        print("=" * 50)

        outliers_found = 0

        for idx, row in df.iterrows():
            unit = row[unit_col]
            result = row[result_col]

            if pd.isna(result):
                continue

            if unit not in thresholds:
                continue

            try:
                value = float(result)
                threshold = thresholds[unit]

                if value > threshold['max']:
                    print(f"❌ ID: {row.get('test_id', idx)} - {value} {unit} is too high (max: {threshold['max']})")
                    outliers_found += 1
                elif value < threshold['min']:
                    print(f"❌ ID: {row.get('test_id', idx)} - {value} {unit} is too low (min: {threshold['min']})")
                    outliers_found += 1

            except (ValueError, TypeError):
                print(f"⚠️ ID: {row.get('test_id', idx)} - Invalid numeric value: '{result}'")
                outliers_found += 1

        if outliers_found == 0:
            print("✅ No outliers found!")

        print("=" * 50)



