import datetime

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
            'C': {'max': 300, 'min': -50}
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
            sample_events = set(group[column_name].unique())

            missing_events = required_events - sample_events

            if missing_events:
                samples_with_issues += 1
                print(f"\n❌ Sample: {sample}")
                print(f"   Has: {sorted(sample_events)}")
                print(f"   Missing: {sorted(missing_events)}")

        print("\n" + "=" * 50)

        if samples_with_issues == 0:
            print("✅ Every sample_id contains all required events")
        else:
            print(f"⚠️ {samples_with_issues} sample(s) have missing events")

        print("=" * 50)

    @staticmethod
    def check_event_order(
            df: pd.DataFrame,
            name: str = "Dataset",
            sample_col: str = "sample_id",
            event_col: str = "event_type",
            timestamp_col: str = "event_timestamp"
    ) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            print(f"⚠️ {name} is empty!")
            return

        for col in [sample_col, event_col, timestamp_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        expected_events = ["received", "testing_started", "testing_finished", "validated"]

        print("=" * 60)
        print(f"📊 EVENT ORDER VALIDATION: {name}")
        print("=" * 60)

        missing_event_errors = 0
        duplicate_event_errors = 0
        order_errors_count = 0
        invalid_timestamp_errors = 0

        total_samples = df[sample_col].nunique()

        for sample_value, group in df.groupby(sample_col):
            existing_events = set(group[event_col].unique())
            missing_events = [event for event in expected_events if event not in existing_events]

            if missing_events:
                print(f"❌ {sample_value}: missing events -> {missing_events}")
                missing_event_errors += 1
                continue

            duplicate_found = False

            for event in expected_events:
                event_count = (group[event_col] == event).sum()

                if event_count > 1:
                    print(f"❌ {sample_value}: duplicate event '{event}' appears {event_count} times")
                    duplicate_event_errors += 1
                    duplicate_found = True

            if duplicate_found:
                continue

            timestamps = {}
            invalid_timestamp_for_sample = False

            for event in expected_events:
                event_time = group.loc[group[event_col] == event, timestamp_col].iloc[0]

                if pd.isna(event_time):
                    print(f"❌ {sample_value}: invalid timestamp for event '{event}'")
                    invalid_timestamp_errors += 1
                    invalid_timestamp_for_sample = True
                else:
                    timestamps[event] = event_time

            if invalid_timestamp_for_sample:
                continue

            if not (timestamps["received"] < timestamps["testing_started"]):
                print(
                    f"❌ {sample_value}: received ({timestamps['received']}) >= "
                    f"testing_started ({timestamps['testing_started']})"
                )
                order_errors_count += 1

            if not (timestamps["testing_started"] < timestamps["testing_finished"]):
                print(
                    f"❌ {sample_value}: testing_started ({timestamps['testing_started']}) >= "
                    f"testing_finished ({timestamps['testing_finished']})"
                )
                order_errors_count += 1

            if not (timestamps["testing_finished"] < timestamps["validated"]):
                print(
                    f"❌ {sample_value}: testing_finished ({timestamps['testing_finished']}) >= "
                    f"validated ({timestamps['validated']})"
                )
                order_errors_count += 1

        print("\n" + "=" * 60)
        print("📊 SUMMARY:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Missing event errors: {missing_event_errors:,}")
        print(f"   Duplicate event errors: {duplicate_event_errors:,}")
        print(f"   Invalid timestamp errors: {invalid_timestamp_errors:,}")
        print(f"   Order errors: {order_errors_count:,}")

        total_errors = (
                missing_event_errors
                + duplicate_event_errors
                + invalid_timestamp_errors
                + order_errors_count
        )

        if total_errors == 0:
            print("   ✅ All samples have correct event sequence!")

        print("=" * 60)

    @staticmethod
    def check_samples_lab_events_date(
            samples_df: pd.DataFrame,
            events_df: pd.DataFrame,
            name: str = "Dataset",
            sample_col: str = "sample_id",
            samples_date_col: str = "date_received",
            event_type_col: str = "event_type",
            event_timestamp_col: str = "event_timestamp"
    ) -> None:

        if not isinstance(samples_df, pd.DataFrame):
            raise TypeError("samples_df must be a pandas DataFrame")

        if not isinstance(events_df, pd.DataFrame):
            raise TypeError("events_df must be a pandas DataFrame")

        if samples_df.empty:
            print(f"⚠️ samples_df is empty!")
            return

        if events_df.empty:
            print(f"⚠️ events_df is empty!")
            return

        required_samples_cols = [sample_col, samples_date_col]
        required_events_cols = [sample_col, event_type_col, event_timestamp_col]

        for col in required_samples_cols:
            if col not in samples_df.columns:
                raise ValueError(f"Column '{col}' not found in samples_df")

        for col in required_events_cols:
            if col not in events_df.columns:
                raise ValueError(f"Column '{col}' not found in events_df")

        print("=" * 70)
        print(f"📊 SAMPLES VS LAB_EVENTS DATE VALIDATION: {name}")
        print("=" * 70)

        # 1. Prepare samples subset
        samples_subset = samples_df[[sample_col, samples_date_col]].copy()
        samples_subset[samples_date_col] = pd.to_datetime(
            samples_subset[samples_date_col],
            errors="coerce"
        )

        # 2. Prepare only 'received' events
        received_events = events_df[events_df[event_type_col] == "received"].copy()
        received_events = received_events[[sample_col, event_timestamp_col]].copy()
        received_events = received_events.rename(
            columns={event_timestamp_col: "received_timestamp"}
        )
        received_events["received_timestamp"] = pd.to_datetime(
            received_events["received_timestamp"],
            errors="coerce"
        )

        # 3. Check duplicates in received events
        duplicate_received = received_events[
            received_events.duplicated(subset=[sample_col], keep=False)
        ].copy()

        duplicate_sample_ids = set(duplicate_received[sample_col].unique())

        if len(duplicate_received) > 0:
            print(f"❌ Duplicate 'received' events found for {len(duplicate_sample_ids)} sample(s):")
            for sample_value in sorted(duplicate_sample_ids):
                count = (received_events[sample_col] == sample_value).sum()
                print(f"   • {sample_value}: {count} 'received' events")

        # 4. Exclude duplicate sample_ids from comparison
        received_events_unique = received_events[
            ~received_events[sample_col].isin(duplicate_sample_ids)
        ].copy()

        # 5. Merge samples with received events
        merged = samples_subset.merge(
            received_events_unique,
            on=sample_col,
            how="left"
        )

        # 6. Check missing received event
        missing_received = merged[merged["received_timestamp"].isna()].copy()

        if len(missing_received) > 0:
            print(f"\n❌ Missing 'received' event for {len(missing_received)} sample(s):")
            for _, row in missing_received.iterrows():
                print(f"   • {row[sample_col]}")

        # 7. Keep only comparable rows
        comparable = merged.dropna(subset=[samples_date_col, "received_timestamp"]).copy()

        # 8. Compute time difference
        comparable["time_diff"] = comparable["received_timestamp"] - comparable[samples_date_col]
        comparable["time_diff_minutes"] = comparable["time_diff"].dt.total_seconds() / 60
        comparable["abs_time_diff_minutes"] = comparable["time_diff_minutes"].abs()

        # 9. Categorize differences
        def classify_difference(minutes: float) -> str:
            if minutes <= 5:
                return "OK"
            elif minutes <= 60:
                return "SUSPICIOUS"
            else:
                return "ERROR"

        comparable["comparison_status"] = comparable["abs_time_diff_minutes"].apply(classify_difference)

        # 10. Show suspicious and error cases
        suspicious_rows = comparable[comparable["comparison_status"] == "SUSPICIOUS"]
        error_rows = comparable[comparable["comparison_status"] == "ERROR"]

        if len(suspicious_rows) > 0:
            print(f"\n⚠️ Suspicious time differences ({len(suspicious_rows)} sample(s)):")
            for _, row in suspicious_rows.iterrows():
                print(
                    f"   • {row[sample_col]}: "
                    f"date_received={row[samples_date_col]}, "
                    f"received_timestamp={row['received_timestamp']}, "
                    f"diff={row['time_diff_minutes']:.2f} min"
                )

        if len(error_rows) > 0:
            print(f"\n❌ Incorrect time differences ({len(error_rows)} sample(s)):")
            for _, row in error_rows.iterrows():
                print(
                    f"   • {row[sample_col]}: "
                    f"date_received={row[samples_date_col]}, "
                    f"received_timestamp={row['received_timestamp']}, "
                    f"diff={row['time_diff_minutes']:.2f} min"
                )

        # 11. Invalid dates
        invalid_samples_dates = samples_subset[samples_subset[samples_date_col].isna()]
        invalid_received_dates = received_events_unique[received_events_unique["received_timestamp"].isna()]

        # 12. Summary
        total_samples = samples_subset[sample_col].nunique()
        ok_count = (comparable["comparison_status"] == "OK").sum()
        suspicious_count = (comparable["comparison_status"] == "SUSPICIOUS").sum()
        error_count = (comparable["comparison_status"] == "ERROR").sum()

        print("\n" + "=" * 70)
        print("📊 SUMMARY:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Duplicate 'received' sample_ids: {len(duplicate_sample_ids):,}")
        print(f"   Missing 'received' events: {len(missing_received):,}")
        print(f"   Invalid dates in samples: {len(invalid_samples_dates):,}")
        print(f"   Invalid received timestamps: {len(invalid_received_dates):,}")
        print(f"   Comparable rows: {len(comparable):,}")
        print(f"   OK: {ok_count:,}")
        print(f"   SUSPICIOUS: {suspicious_count:,}")
        print(f"   ERROR: {error_count:,}")
        print("=" * 70)

    @staticmethod
    def check_tests_vs_events(
            tests_df: pd.DataFrame,
            events_df: pd.DataFrame,
            name: str = "Dataset",
            sample_col: str = "sample_id",
            test_date_col: str = "test_date",
            event_type_col: str = "event_type",
            event_timestamp_col: str = "event_timestamp"
    ) -> None:

        if not isinstance(tests_df, pd.DataFrame):
            raise TypeError("tests_df must be a pandas DataFrame")

        if not isinstance(events_df, pd.DataFrame):
            raise TypeError("events_df must be a pandas DataFrame")

        if tests_df.empty:
            print("⚠️ tests_df is empty!")
            return

        if events_df.empty:
            print("⚠️ events_df is empty!")
            return

        required_tests_cols = [sample_col, test_date_col]
        required_events_cols = [sample_col, event_type_col, event_timestamp_col]

        for col in required_tests_cols:
            if col not in tests_df.columns:
                raise ValueError(f"Column '{col}' not found in tests_df")

        for col in required_events_cols:
            if col not in events_df.columns:
                raise ValueError(f"Column '{col}' not found in events_df")

        print("=" * 70)
        print(f"📊 TESTS VS LAB_EVENTS VALIDATION: {name}")
        print("=" * 70)

        # 1. Prepare tests
        tests = tests_df[[sample_col, test_date_col]].copy()
        tests[test_date_col] = pd.to_datetime(tests[test_date_col], errors="coerce")

        # 2. Prepare events (pivot: received, started, finished)
        events = events_df.copy()
        events[event_timestamp_col] = pd.to_datetime(events[event_timestamp_col], errors="coerce")

        required_events = ["received", "testing_started", "testing_finished"]

        # only important events
        events = events[events[event_type_col].isin(required_events)]

        # pivot -> one line for sample_id
        events_pivot = events.pivot_table(
            index=sample_col,
            columns=event_type_col,
            values=event_timestamp_col,
            aggfunc="first"
        ).reset_index()

        # change column name
        events_pivot = events_pivot.rename(columns={
            "received": "received_ts",
            "testing_started": "testing_started_ts",
            "testing_finished": "testing_finished_ts"
        })

        # 3. Merge
        merged = tests.merge(events_pivot, on=sample_col, how="left")

        # 4. Missing events
        missing_events = merged[
            merged[["received_ts", "testing_started_ts", "testing_finished_ts"]].isna().any(axis=1)
        ]

        if len(missing_events) > 0:
            print(f"❌ {len(missing_events)} tests with missing event timestamps")

        # 5. We remove incomplete ones for further validation
        valid = merged.dropna(
            subset=["received_ts", "testing_started_ts", "testing_finished_ts", test_date_col]
        ).copy()

        # 6. Logical validations

        # test before received
        before_received = valid[valid[test_date_col] < valid["received_ts"]]

        # test before start
        before_start = valid[valid[test_date_col] < valid["testing_started_ts"]]

        # post-test
        after_finish = valid[valid[test_date_col] > valid["testing_finished_ts"]]

        correct = valid[
            (valid[test_date_col] >= valid["testing_started_ts"]) &
            (valid[test_date_col] <= valid["testing_finished_ts"])
            ]

        # 7. Result

        if len(before_received) > 0:
            print(f"\n❌ Tests before 'received': {len(before_received)}")
            for _, row in before_received.iterrows():
                print(f"   • {row[sample_col]} | test_date={row[test_date_col]} < received={row['received_ts']}")

        if len(before_start) > 0:
            print(f"\n⚠️ Tests before 'testing_started': {len(before_start)}")

        if len(after_finish) > 0:
            print(f"\n❌ Tests after 'testing_finished': {len(after_finish)}")

        # 8. Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY:")
        print(f"   Total tests: {len(tests):,}")
        print(f"   Missing event data: {len(missing_events):,}")
        print(f"   Valid for comparison: {len(valid):,}")
        print(f"   Correct tests: {len(correct):,}")
        print(f"   Before received: {len(before_received):,}")
        print(f"   Before testing_started: {len(before_start):,}")
        print(f"   After testing_finished: {len(after_finish):,}")
        print("=" * 70)