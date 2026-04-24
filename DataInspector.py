import pandas as pd

class DataInspector:

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, name: str = "Dataset") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")

        if df.empty:
            raise ValueError(f"{name} is empty")

    @staticmethod
    def _validate_column_exists(df: pd.DataFrame, column_name: str) -> None:
        if column_name is None:
            raise ValueError("column_name must be provided")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

    @staticmethod
    def _is_missing_value(series: pd.Series) -> pd.Series:
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
    def display_null_value(
        df: pd.DataFrame,
        name: str = "Dataset",
        column_name: str = None
    ) -> None:
        DataInspector._validate_dataframe(df, name)
        DataInspector._validate_column_exists(df, column_name)

        missing_mask = DataInspector._is_missing_value(df[column_name])
        missing_count = missing_mask.sum()
        total_rows = len(df)

        print("=" * 50)
        print(f"📊 MISSING VALUES ANALYSIS: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)
        print(f"📈 Total rows: {total_rows:,}")
        print(f"📊 Missing values: {missing_count:,} ({missing_count / total_rows * 100:.2f}%)")

        if missing_count == 0:
            print(f"✅ No missing values found in '{column_name}'")

        print("=" * 50)

    @staticmethod
    def display_unique_value(
        df: pd.DataFrame,
        name: str = "Dataset",
        column_name: str = None
    ) -> None:
        DataInspector._validate_dataframe(df, name)
        DataInspector._validate_column_exists(df, column_name)

        unique_values = df[column_name].unique()
        unique_count = df[column_name].nunique(dropna=False)
        total_rows = len(df)

        print("=" * 50)
        print(f"📊 UNIQUE VALUES ANALYSIS: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)
        print(f"📈 Unique values: {unique_count:,} / {total_rows:,}")

        print(f"\n🔍 First {min(20, unique_count)} unique values:")
        for i, value in enumerate(unique_values[:20], start=1):
            print(f"   {i}. {value}")

        if unique_count > 20:
            print(f"   ... and {unique_count - 20} more")

        print("=" * 50)

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

        invalid_mask = ~df[column_name].isin(allowed_values)
        invalid_rows = df[invalid_mask]
        invalid_unique_values = invalid_rows[column_name].unique()

        print("=" * 50)
        print(f"📊 VALUE VALIDATION: {name}")
        print(f"📍 Column: {column_name}")
        print("=" * 50)
        print(f"📈 Allowed values: {', '.join(map(str, allowed_values))}")
        print(f"🔍 Found values: {', '.join(map(str, df[column_name].unique()))}")

        if len(invalid_rows) > 0:
            print(f"\n❌ Incorrect unique values: {', '.join(map(str, invalid_unique_values))}")
            print(f"📊 Incorrect records: {len(invalid_rows):,}")
        else:
            print("\n✅ All values are correct")

        print("=" * 50)

    @staticmethod
    def units_correctness(
        df: pd.DataFrame,
        name: str = "Dataset",
        parameter_col: str = "parameter",
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
            incorrect = param_df[param_df[unit_col] != expected_unit]

            if len(incorrect) > 0:
                total_incorrect += len(incorrect)

                print(f"\n❌ PARAMETER: {param}")
                print(f"   Expected unit: {expected_unit}")
                print(f"   Incorrect records: {len(incorrect):,}")

                for _, row in incorrect.iterrows():
                    print(
                        f"      • ID: {row.get('test_id', 'N/A')} "
                        f"→ actual unit: {row[unit_col]}"
                    )

        if total_incorrect == 0:
            print("✅ All units are correct")

        print(f"\n📊 TOTAL INCORRECT UNIT RECORDS: {total_incorrect:,}")
        print("=" * 60)

    @staticmethod
    def unlogical_combination(
        df: pd.DataFrame,
        name: str = "Dataset",
        status_col: str = "status",
        result_col: str = "result_value"
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

        if len(ok_but_missing) > 0:
            print(f"❌ Status='OK' but missing result_value: {len(ok_but_missing):,}")

        if len(error_but_has_result) > 0:
            print(f"❌ Status='ERROR' but result_value exists: {len(error_but_has_result):,}")

        total_errors = len(ok_but_missing) + len(error_but_has_result)

        if total_errors == 0:
            print("✅ All status/result combinations are logical")
        else:
            print(f"\n📈 Total logical errors: {total_errors:,}")

        print("=" * 60)

    @staticmethod
    def duplicated_in_tests(
        df: pd.DataFrame,
        name: str = "Dataset",
        sample_col: str = "sample_id",
        parameter_col: str = "parameter",
        date_col: str = "test_date"
    ) -> None:
        DataInspector._validate_dataframe(df, name)

        for col in [sample_col, parameter_col]:
            DataInspector._validate_column_exists(df, col)

        print("=" * 60)
        print(f"📊 DUPLICATE ANALYSIS: {name}")
        print("=" * 60)
        print(f"🔍 Checking duplicates by: {sample_col} + {parameter_col}")

        duplicates_mask = df.duplicated(subset=[sample_col, parameter_col], keep=False)
        duplicate_rows = df[duplicates_mask]

        if len(duplicate_rows) == 0:
            print("✅ No duplicates found")
        else:
            duplicate_groups = duplicate_rows.groupby([sample_col, parameter_col])

            print(f"❌ Duplicate rows: {len(duplicate_rows):,}")
            print(f"📊 Duplicate groups: {len(duplicate_groups):,}")

            for (sample_id, parameter), group in duplicate_groups:
                print(f"\n   📌 {sample_id} → {parameter}: {len(group)} occurrences")

                if "test_id" in df.columns:
                    print(f"      Test IDs: {group['test_id'].tolist()}")

                if date_col in df.columns:
                    print(f"      Dates: {group[date_col].tolist()}")

        duplicate_percent = len(duplicate_rows) / len(df) * 100

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

        for col in [param_col, result_col, unit_col]:
            DataInspector._validate_column_exists(df, col)

        thresholds = {
            "sulfur": {"min": 0, "max": 50, "unit": "mg/kg"},
            "water": {"min": 0, "max": 600, "unit": "mg/kg"},
            "density": {"min": 600, "max": 1100, "unit": "kg/m3"},
            "viscosity": {"min": 0, "max": 150, "unit": "cSt"},
            "chloride": {"min": 0, "max": 100, "unit": "mg/kg"},
            "flash_point": {"min": -10, "max": 100, "unit": "C"},
        }

        if custom_thresholds:
            thresholds.update(custom_thresholds)

        print("=" * 60)
        print(f"📊 OUTLIER DETECTION: {name}")
        print("=" * 60)

        outliers_found = 0
        invalid_values = 0

        for idx, row in df.iterrows():
            param = row[param_col]
            result = row[result_col]

            if DataInspector._is_missing_value(pd.Series([result])).iloc[0]:
                continue

            if param not in thresholds:
                continue

            try:
                value = float(result)
            except (ValueError, TypeError):
                print(f"⚠️ ID: {row.get('test_id', idx)} - invalid numeric value: {result}")
                invalid_values += 1
                continue

            min_value = thresholds[param]["min"]
            max_value = thresholds[param]["max"]
            expected_unit = thresholds[param]["unit"]

            if value < min_value:
                print(
                    f"❌ {param}: ID {row.get('test_id', idx)} "
                    f"→ {value} {expected_unit} < {min_value}"
                )
                outliers_found += 1

            elif value > max_value:
                print(
                    f"❌ {param}: ID {row.get('test_id', idx)} "
                    f"→ {value} {expected_unit} > {max_value}"
                )
                outliers_found += 1

        print("\n📊 OUTLIER SUMMARY:")
        print(f"   Outliers found: {outliers_found:,}")
        print(f"   Invalid numeric values: {invalid_values:,}")

        if outliers_found == 0 and invalid_values == 0:
            print("✅ No outliers or invalid numeric values found")

        print("=" * 60)

    @staticmethod
    def validate_required_events_per_sample(
        df: pd.DataFrame,
        name: str = "Dataset",
        sample_col: str = "sample_id",
        event_col: str = "event_type"
    ) -> None:
        DataInspector._validate_dataframe(df, name)

        for col in [sample_col, event_col]:
            DataInspector._validate_column_exists(df, col)

        required_events = {"received", "testing_started", "testing_finished", "validated"}

        print("=" * 60)
        print(f"📊 REQUIRED EVENTS VALIDATION: {name}")
        print("=" * 60)

        samples_with_issues = 0

        for sample_id, group in df.groupby(sample_col):
            sample_events = set(group[event_col].unique())
            missing_events = required_events - sample_events

            if missing_events:
                samples_with_issues += 1
                print(f"❌ {sample_id}: missing {sorted(missing_events)}")

        if samples_with_issues == 0:
            print("✅ Every sample has all required events")
        else:
            print(f"⚠️ Samples with missing events: {samples_with_issues:,}")

        print("=" * 60)

    @staticmethod
    def check_event_order(
        df: pd.DataFrame,
        name: str = "Dataset",
        sample_col: str = "sample_id",
        event_col: str = "event_type",
        timestamp_col: str = "event_timestamp"
    ) -> None:
        DataInspector._validate_dataframe(df, name)

        for col in [sample_col, event_col, timestamp_col]:
            DataInspector._validate_column_exists(df, col)

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        expected_events = ["received", "testing_started", "testing_finished", "validated"]

        print("=" * 60)
        print(f"📊 EVENT ORDER VALIDATION: {name}")
        print("=" * 60)

        missing_event_errors = 0
        duplicate_event_errors = 0
        invalid_timestamp_errors = 0
        order_errors = 0

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

            for event in expected_events:
                event_time = group.loc[group[event_col] == event, timestamp_col].iloc[0]

                if pd.isna(event_time):
                    print(f"❌ {sample_value}: invalid timestamp for '{event}'")
                    invalid_timestamp_errors += 1
                    timestamps = None
                    break

                timestamps[event] = event_time

            if timestamps is None:
                continue

            if not (timestamps["received"] < timestamps["testing_started"]):
                print(f"❌ {sample_value}: received >= testing_started")
                order_errors += 1

            if not (timestamps["testing_started"] < timestamps["testing_finished"]):
                print(f"❌ {sample_value}: testing_started >= testing_finished")
                order_errors += 1

            if not (timestamps["testing_finished"] < timestamps["validated"]):
                print(f"❌ {sample_value}: testing_finished >= validated")
                order_errors += 1

        total_errors = (
            missing_event_errors
            + duplicate_event_errors
            + invalid_timestamp_errors
            + order_errors
        )

        print("\n📊 SUMMARY:")
        print(f"   Total samples: {df[sample_col].nunique():,}")
        print(f"   Missing event errors: {missing_event_errors:,}")
        print(f"   Duplicate event errors: {duplicate_event_errors:,}")
        print(f"   Invalid timestamp errors: {invalid_timestamp_errors:,}")
        print(f"   Order errors: {order_errors:,}")

        if total_errors == 0:
            print("✅ All samples have correct event sequence")

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
        DataInspector._validate_dataframe(samples_df, "samples_df")
        DataInspector._validate_dataframe(events_df, "events_df")

        for col in [sample_col, samples_date_col]:
            DataInspector._validate_column_exists(samples_df, col)

        for col in [sample_col, event_type_col, event_timestamp_col]:
            DataInspector._validate_column_exists(events_df, col)

        print("=" * 70)
        print(f"📊 SAMPLES VS LAB_EVENTS DATE VALIDATION: {name}")
        print("=" * 70)

        samples_subset = samples_df[[sample_col, samples_date_col]].copy()
        samples_subset[samples_date_col] = pd.to_datetime(
            samples_subset[samples_date_col],
            errors="coerce"
        )

        received_events = events_df[events_df[event_type_col] == "received"].copy()
        received_events = received_events[[sample_col, event_timestamp_col]].copy()
        received_events = received_events.rename(
            columns={event_timestamp_col: "received_timestamp"}
        )
        received_events["received_timestamp"] = pd.to_datetime(
            received_events["received_timestamp"],
            errors="coerce"
        )

        duplicate_received = received_events[
            received_events.duplicated(subset=[sample_col], keep=False)
        ]

        duplicate_sample_ids = set(duplicate_received[sample_col].unique())

        if duplicate_sample_ids:
            print(f"❌ Duplicate 'received' events for {len(duplicate_sample_ids)} sample(s)")

        received_events_unique = received_events[
            ~received_events[sample_col].isin(duplicate_sample_ids)
        ].copy()

        merged = samples_subset.merge(
            received_events_unique,
            on=sample_col,
            how="left"
        )

        missing_received = merged[merged["received_timestamp"].isna()]

        comparable = merged.dropna(subset=[samples_date_col, "received_timestamp"]).copy()

        comparable["time_diff"] = comparable["received_timestamp"] - comparable[samples_date_col]
        comparable["time_diff_minutes"] = comparable["time_diff"].dt.total_seconds() / 60
        comparable["abs_time_diff_minutes"] = comparable["time_diff_minutes"].abs()

        def classify_difference(minutes: float) -> str:
            if minutes <= 5:
                return "OK"
            if minutes <= 60:
                return "SUSPICIOUS"
            return "ERROR"

        comparable["comparison_status"] = comparable["abs_time_diff_minutes"].apply(classify_difference)

        suspicious_rows = comparable[comparable["comparison_status"] == "SUSPICIOUS"]
        error_rows = comparable[comparable["comparison_status"] == "ERROR"]

        if len(missing_received) > 0:
            print(f"❌ Missing 'received' event: {len(missing_received):,}")

        if len(suspicious_rows) > 0:
            print(f"⚠️ Suspicious time differences: {len(suspicious_rows):,}")

        if len(error_rows) > 0:
            print(f"❌ Incorrect time differences: {len(error_rows):,}")

        print("\n📊 SUMMARY:")
        print(f"   Total samples: {samples_subset[sample_col].nunique():,}")
        print(f"   Duplicate 'received' sample_ids: {len(duplicate_sample_ids):,}")
        print(f"   Missing 'received' events: {len(missing_received):,}")
        print(f"   Comparable rows: {len(comparable):,}")
        print(f"   OK: {(comparable['comparison_status'] == 'OK').sum():,}")
        print(f"   SUSPICIOUS: {(comparable['comparison_status'] == 'SUSPICIOUS').sum():,}")
        print(f"   ERROR: {(comparable['comparison_status'] == 'ERROR').sum():,}")
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
        DataInspector._validate_dataframe(tests_df, "tests_df")
        DataInspector._validate_dataframe(events_df, "events_df")

        for col in [sample_col, test_date_col]:
            DataInspector._validate_column_exists(tests_df, col)

        for col in [sample_col, event_type_col, event_timestamp_col]:
            DataInspector._validate_column_exists(events_df, col)

        print("=" * 70)
        print(f"📊 TESTS VS LAB_EVENTS VALIDATION: {name}")
        print("=" * 70)

        tests = tests_df[[sample_col, test_date_col]].copy()
        tests[test_date_col] = pd.to_datetime(tests[test_date_col], errors="coerce")

        events = events_df.copy()
        events[event_timestamp_col] = pd.to_datetime(events[event_timestamp_col], errors="coerce")

        required_events = ["received", "testing_started", "testing_finished"]
        events = events[events[event_type_col].isin(required_events)]

        events_pivot = events.pivot_table(
            index=sample_col,
            columns=event_type_col,
            values=event_timestamp_col,
            aggfunc="first"
        ).reset_index()

        events_pivot = events_pivot.rename(columns={
            "received": "received_ts",
            "testing_started": "testing_started_ts",
            "testing_finished": "testing_finished_ts"
        })

        merged = tests.merge(events_pivot, on=sample_col, how="left")

        missing_events = merged[
            merged[["received_ts", "testing_started_ts", "testing_finished_ts"]].isna().any(axis=1)
        ]

        valid = merged.dropna(
            subset=["received_ts", "testing_started_ts", "testing_finished_ts", test_date_col]
        ).copy()

        before_received = valid[valid[test_date_col] < valid["received_ts"]]
        before_start = valid[valid[test_date_col] < valid["testing_started_ts"]]
        after_finish = valid[valid[test_date_col] > valid["testing_finished_ts"]]

        correct = valid[
            (valid[test_date_col] >= valid["testing_started_ts"]) &
            (valid[test_date_col] <= valid["testing_finished_ts"])
        ]

        print("\n📊 SUMMARY:")
        print(f"   Total tests: {len(tests):,}")
        print(f"   Missing event data: {len(missing_events):,}")
        print(f"   Valid for comparison: {len(valid):,}")
        print(f"   Correct tests: {len(correct):,}")
        print(f"   Before received: {len(before_received):,}")
        print(f"   Before testing_started: {len(before_start):,}")
        print(f"   After testing_finished: {len(after_finish):,}")
        print("=" * 70)