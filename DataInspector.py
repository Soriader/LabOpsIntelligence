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
    def display_null_value(df: pd.DataFrame, name: str = "Null value", column_name: str = None) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        print("=" * 50)
        print(f"📊 NULL ANALYSIS: {name}")
        print("=" * 50)

        nulls_per_column = df[column_name].isnull().sum()

        if nulls_per_column == 0:
            print(f"✅ No null values in column '{column_name}'!")
        else:
            print(f"📈 Nulls in '{column_name}': {nulls_per_column:,} ({nulls_per_column / len(df) * 100:.2f}%)")

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
    def display_correct_value(df: pd.DataFrame, name: str = "Dataset", column_name: str = None, allowed_values: list = None) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if allowed_values is None:
            print(f"⚠️ {allowed_values} is empty!")
            return

        if column_name is None:
            raise ValueError("column_name must be provided")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        unique_values = df[column_name].unique()

        invalid_values = [val for val in unique_values if val not in allowed_values]

        print("=" * 50)
        print(f"📊 PRIORITY VALIDATION: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)

        print(f"📈 Allowed values: {', '.join(allowed_values)}")
        print(f"🔍 Found unique values: {', '.join(map(str, unique_values))}")

        if invalid_values:
            print(f"\n❌ INCORRECT values found: {', '.join(map(str, invalid_values))}")
            print(f"📊 Count of incorrect records: {len(invalid_values)}")
        else:
            print(f"\n✅ All values are correct! Only allowed values present.")

        print("=" * 50)

    @staticmethod
    def units_correctness(df: pd.DataFrame, name: str = "Null value", column_name: str = None, correct_units: dict = None) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if correct_units is None:
            print(f"⚠️ {name} is empty!")
            return

        if column_name is None:
            raise ValueError("column_name must be provided")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        print("=" * 50)
        print(f"📊 CORRECT UNITS ANALYSIS: {name}")
        print("=" * 50)

        for param, expected_unit in correct_units.items():
            param_df = df[df[column_name] == param]

            incorrect = param_df[param_df['unit'] != expected_unit]

            if len(incorrect) > 0:
                print(f"\n❌ PARAMETER: {param}")
                print(f"   Expected unit: {expected_unit}")
                print(f"   Find wrong units:")

                for _, row in incorrect.iterrows():
                    print(f"      • ID: {row['test_id']} → unit: {row[column_name]}")

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

    @staticmethod
    def validate_required_events_per_sample(df: pd.DataFrame,
                                            name: str = "Dataset",
                                            sample_id: str = "sample_id",
                                            column_name: str = "event_type") -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        if sample_id not in df.columns:
            raise ValueError(f"Column '{sample_id}' not found")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        required_events = {"received", "testing_started", "testing_finished", "validated"}

        print("=" * 50)
        print(f"📊 EVENT VALIDATION: {name}")
        print("=" * 50)

        group_by_id = df.groupby(sample_id)
        samples_with_issues = 0

        for sample, group in group_by_id:
            has_events = set(group[column_name].unique())

            missing_events = required_events - has_events

            if missing_events:
                samples_with_issues += 1
                print(f"\n❌ Sample: {sample}")
                print(f"   Has: {sorted(has_events)}")
                print(f"   Missing: {sorted(missing_events)}")

        print("\n" + "=" * 50)

        if samples_with_issues == 0:
            print("✅ Every sample_id contains all required events")
        else:
            print(f"⚠️ {samples_with_issues} sample(s) have missing events")

        print("=" * 50)