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
    def display_data(df: pd.DataFrame, name: str = "Dataset") -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        print("=" * 50)
        print(f"📊 UNIQUE VALUES ANALYSIS: {name}")

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

